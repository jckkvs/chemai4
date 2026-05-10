"""
backend/chem/xtb_ml_features.py

xTB最適化構造からの機械学習用特徴量抽出モジュール。

既存の xtb_adapter.py が出力する記述子辞書 (xtb_HomoEnergy 等) を入力とし、
化学的硬さ・軟らかさ・親電子性指数・3D幾何特徴量などの **派生** ML特徴量を計算する。

既存の xtb_adapter.py および _XTB_DESCRIPTORS は一切変更しない。
このモジュールは「xTBの出力を受け取って追加の特徴量を返す」後段処理に徹する。

Implements: 量子化学量 → 派生指標 → 統計的特徴 の変換パイプライン
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 設定
# ============================================================

@dataclass
class XTBFeatureConfig:
    """特徴量抽出の設定。"""
    # 基本量子化学量
    include_energy: bool = True
    include_orbitals: bool = True           # HOMO/LUMO/gap
    include_dipole: bool = True
    include_polarizability: bool = False

    # 電荷統計
    charge_statistics: list[str] | None = None  # ['max','min','mean','std','skew']

    # 派生指標
    include_hardness_softness: bool = True   # η=(IP-EA)/2, S=1/η
    include_electrophilicity: bool = True    # ω=μ²/2η
    include_fukui_approx: bool = False       # 簡易福井関数（原子電荷差分）

    # 3D幾何
    include_3d_geom: bool = True             # 体積, 表面積, 慣性モーメント

    # 出力形式
    feature_prefix: str = "xtb_ml_"


# ============================================================
# 抽出エンジン
# ============================================================

class XTBMLFeatureExtractor:
    """
    xTB出力辞書から機械学習向けの派生特徴量を抽出するクラス。

    使い方::

        extractor = XTBMLFeatureExtractor()
        features = extractor.extract_from_xtb_output(xtb_result_dict)
        # features = {"xtb_ml_Hardness": 3.14, "xtb_ml_Softness": 0.318, ...}
    """

    def __init__(self, config: XTBFeatureConfig | None = None):
        self.config = config or XTBFeatureConfig()

    # ────────────────────────────────────────────────────────
    # メイン API
    # ────────────────────────────────────────────────────────

    def extract_from_xtb_output(
        self,
        xtb_result: dict,
        xyz_coordinates: np.ndarray | None = None,
        atomic_numbers: list[int] | None = None,
    ) -> dict[str, float]:
        """
        xTBの計算結果辞書から派生ML特徴量を生成する。

        Args:
            xtb_result: xtb_adapter の _parse_xtb_output() が返す辞書。
                        キー例: ``xtb_HomoEnergy``, ``xtb_LumoEnergy``, ...
            xyz_coordinates: 最適化後の3D座標 (N_atoms x 3)。
                             Noneの場合は3D幾何特徴量をスキップ。
            atomic_numbers: 原子番号リスト (長さ N_atoms)。
                            元素別統計を計算する場合に使用。

        Returns:
            ``{feature_name: value}`` の辞書。
        """
        features: dict[str, float] = {}
        prefix = self.config.feature_prefix

        # === 基本量子化学量 ===
        if self.config.include_energy:
            self._extract_energy_features(xtb_result, features, prefix)

        if self.config.include_dipole and "xtb_DipoleMoment" in xtb_result:
            features[f"{prefix}Dipole"] = float(xtb_result["xtb_DipoleMoment"])

        if self.config.include_polarizability and "xtb_Polarizability" in xtb_result:
            features[f"{prefix}Polarizability"] = float(xtb_result["xtb_Polarizability"])

        # === 電荷統計 ===
        if self.config.charge_statistics:
            self._extract_charge_statistics(xtb_result, features, prefix)

        # === 派生指標（化学的硬さ・親電子性など）===
        ip, ea = self._extract_conceptual_dft(xtb_result, features, prefix)

        # === 3D幾何学的特徴量 ===
        if self.config.include_3d_geom and xyz_coordinates is not None:
            features.update(
                self._compute_3d_geometry_features(xyz_coordinates, atomic_numbers)
            )

        # === 簡易福井関数近似 ===
        if self.config.include_fukui_approx:
            self._extract_fukui_approx(xtb_result, features, prefix)

        return features

    def batch_extract(
        self,
        xtb_results: list[dict],
        xyz_list: list[np.ndarray] | None = None,
        atomic_numbers_list: list[list[int]] | None = None,
    ) -> pd.DataFrame:
        """
        複数分子分のxTB結果をバッチ処理し、DataFrameとして返す。

        Args:
            xtb_results: 各分子の xtb_result 辞書のリスト。
            xyz_list: 各分子の最適化後座標のリスト（省略可）。
            atomic_numbers_list: 各分子の原子番号リスト（省略可）。

        Returns:
            行=分子, 列=特徴量 の DataFrame。
        """
        all_features: list[dict[str, float]] = []
        for i, result in enumerate(xtb_results):
            xyz = xyz_list[i] if xyz_list else None
            atoms = atomic_numbers_list[i] if atomic_numbers_list else None
            try:
                feats = self.extract_from_xtb_output(result, xyz, atoms)
            except Exception as e:
                logger.warning("XTB ML特徴量抽出エラー (idx=%d): %s", i, e)
                feats = {}
            all_features.append(feats)
        return pd.DataFrame(all_features)

    # ────────────────────────────────────────────────────────
    # 内部ヘルパー
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_energy_features(
        xtb_result: dict,
        features: dict[str, float],
        prefix: str,
    ) -> None:
        """基本エネルギー量を抽出する。"""
        key_map = {
            "xtb_TotalEnergy": "TotalEnergy",
            "xtb_HomoEnergy": "HomoEnergy",
            "xtb_LumoEnergy": "LumoEnergy",
        }
        for src, dst in key_map.items():
            if src in xtb_result and xtb_result[src] is not None:
                try:
                    features[f"{prefix}{dst}"] = float(xtb_result[src])
                except (TypeError, ValueError):
                    pass

        # HOMO-LUMOギャップ
        if "xtb_HomoLumoGap" in xtb_result:
            try:
                features[f"{prefix}Gap"] = float(xtb_result["xtb_HomoLumoGap"])
            except (TypeError, ValueError):
                pass
        elif (
            "xtb_HomoEnergy" in xtb_result
            and "xtb_LumoEnergy" in xtb_result
            and xtb_result["xtb_HomoEnergy"] is not None
            and xtb_result["xtb_LumoEnergy"] is not None
        ):
            features[f"{prefix}Gap"] = float(
                xtb_result["xtb_LumoEnergy"] - xtb_result["xtb_HomoEnergy"]
            )

    @staticmethod
    def _extract_charge_statistics(
        xtb_result: dict,
        features: dict[str, float],
        prefix: str,
    ) -> None:
        """Mulliken電荷の統計量を抽出する。"""
        stat_keys = {
            "max": "xtb_MullikenChargeMax",
            "min": "xtb_MullikenChargeMin",
            "mean": "xtb_MullikenChargeMean",
            "std": "xtb_MullikenChargeStd",
        }
        for stat_name, src_key in stat_keys.items():
            if src_key in xtb_result and xtb_result[src_key] is not None:
                try:
                    features[f"{prefix}Charge{stat_name.capitalize()}"] = float(
                        xtb_result[src_key]
                    )
                except (TypeError, ValueError):
                    pass

    def _extract_conceptual_dft(
        self,
        xtb_result: dict,
        features: dict[str, float],
        prefix: str,
    ) -> tuple[float | None, float | None]:
        """
        概念DFT指標（化学的硬さ η, 軟らかさ S, 親電子性 ω）を
        Koopmans近似で算出する。

        IP ≈ -E_HOMO,  EA ≈ -E_LUMO  (Koopmans' theorem)
        η  = (IP - EA) / 2
        S  = 1 / (2η)
        μ  = -(IP + EA) / 2  (chemical potential)
        ω  = μ² / (2η)       (electrophilicity index, Parr et al. 1999)

        Returns:
            (ip, ea) タプル。算出不能の場合は (None, None)。
        """
        homo = xtb_result.get("xtb_HomoEnergy")
        lumo = xtb_result.get("xtb_LumoEnergy")

        if homo is None or lumo is None:
            return None, None

        try:
            ip = -float(homo)   # ionization potential
            ea = -float(lumo)   # electron affinity
        except (TypeError, ValueError):
            return None, None

        if self.config.include_hardness_softness:
            hardness = (ip - ea) / 2.0
            features[f"{prefix}Hardness"] = hardness
            if abs(hardness) > 1e-8:
                features[f"{prefix}Softness"] = 1.0 / (2.0 * hardness)
            else:
                features[f"{prefix}Softness"] = 0.0

        if self.config.include_electrophilicity:
            hardness = (ip - ea) / 2.0
            mu = -(ip + ea) / 2.0  # chemical potential
            if abs(hardness) > 1e-8:
                features[f"{prefix}Electrophilicity"] = (mu ** 2) / (2.0 * hardness)

        return ip, ea

    @staticmethod
    def _extract_fukui_approx(
        xtb_result: dict,
        features: dict[str, float],
        prefix: str,
    ) -> None:
        """
        簡易福井関数近似: Mulliken電荷の極値を反応性指標として使う。

        - FukuiElectrophilic ≈ |q_min| (最も負の電荷 → 求核攻撃を受けやすい)
        - FukuiNucleophilic  ≈ |q_max| (最も正の電荷 → 求電子攻撃を受けやすい)
        """
        q_min = xtb_result.get("xtb_MullikenChargeMin")
        q_max = xtb_result.get("xtb_MullikenChargeMax")
        if q_min is not None:
            features[f"{prefix}FukuiElectrophilic"] = abs(float(q_min))
        if q_max is not None:
            features[f"{prefix}FukuiNucleophilic"] = abs(float(q_max))

    def _compute_3d_geometry_features(
        self,
        coords: np.ndarray,
        atomic_numbers: list[int] | None = None,
    ) -> dict[str, float]:
        """
        3D座標から幾何学的特徴量を計算する。

        計算内容:
        - 慣性テンソルの固有値（分子形状の特徴付け）
        - 非球面性パラメータ (asphericity)
        - 分子サイズ（最大・平均原子間距離）
        - 元素別の重心距離統計（原子番号リスト提供時）
        """
        features: dict[str, float] = {}
        prefix = self.config.feature_prefix + "3D_"

        if coords.shape[0] < 2:
            return features

        # 重心
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid

        # 慣性テンソル（質量=1 の簡易版）
        # I_ab = Σ_i (|r_i|² δ_ab - r_ia * r_ib)
        n_atoms = coords.shape[0]
        inertia = np.zeros((3, 3))
        for c in centered:
            r2 = np.dot(c, c)
            inertia += r2 * np.eye(3) - np.outer(c, c)

        eigenvalues = np.sort(np.linalg.eigvalsh(inertia))

        features[f"{prefix}InertiaMin"] = float(eigenvalues[0])
        features[f"{prefix}InertiaMid"] = float(eigenvalues[1])
        features[f"{prefix}InertiaMax"] = float(eigenvalues[2])

        # 非球面性: 大→棒状, 小→球状
        features[f"{prefix}Asphericity"] = float(
            eigenvalues[2] - (eigenvalues[0] + eigenvalues[1]) / 2.0
        )

        # 分子サイズ
        # 原子間距離行列の上三角部分のみ計算（メモリ節約）
        dists = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dists.append(float(np.linalg.norm(coords[i] - coords[j])))

        if dists:
            features[f"{prefix}MaxDistance"] = max(dists)
            features[f"{prefix}MeanDistance"] = float(np.mean(dists))
            features[f"{prefix}MinDistance"] = min(dists)

        # 元素別統計
        if atomic_numbers and len(atomic_numbers) == n_atoms:
            for elem in set(atomic_numbers):
                mask = np.array(atomic_numbers) == elem
                elem_coords = coords[mask]
                if elem_coords.shape[0] >= 2:
                    elem_centroid = np.mean(elem_coords, axis=0)
                    radii = np.linalg.norm(elem_coords - elem_centroid, axis=1)
                    features[f"{prefix}Elem{elem}_RadiusMean"] = float(np.mean(radii))

        return features
