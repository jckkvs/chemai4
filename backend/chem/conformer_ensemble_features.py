"""
backend/chem/conformer_ensemble_features.py

複数conformer（立体配座）のアンサンブルから統計的ML特徴量を抽出するモジュール。

単一構造ではなく「構造アンサンブル」から以下の統計量を特徴量化:
- エネルギー分布: mean, std, range, Boltzmann重み
- 軌道エネルギーの揺らぎ: HOMO/LUMOのconformer間標準偏差
- 電荷分布の構造依存性: 原子別電荷のconformer間変動
- 3D記述子のロバストネス: 形状特徴の安定性指標

化学的根拠:
- 分子の柔軟性・立体異性体効果をMLモデルに反映可能
- 単一構造の記述子が立体配座に強く依存する問題を軽減
- Boltzmann加重平均による熱力学的に代表的な特徴量

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConformerEnsembleConfig:
    """アンサンブル特徴量抽出の設定。"""
    # エネルギー統計
    include_energy_stats: bool = True
    # 軌道エネルギーの揺らぎ
    include_orbital_fluctuation: bool = True
    # 電荷分布の揺らぎ
    include_charge_fluctuation: bool = True
    # 3D形状の揺らぎ
    include_shape_stability: bool = True
    # Boltzmann重み付き平均
    use_boltzmann_weights: bool = True
    # 温度 (K) — Boltzmann分布用
    temperature: float = 298.15
    # 特徴量プレフィックス
    feature_prefix: str = "ens_"


class ConformerEnsembleExtractor:
    """
    複数conformerのxTB結果から統計的ML特徴量を抽出する。

    使い方::

        extractor = ConformerEnsembleExtractor()
        features = extractor.extract(conformer_xtb_results)
        # features = {"ens_energy_mean": -10.2, "ens_energy_std": 0.003, ...}

    Note:
        入力は「同一分子の複数conformer」に対するxTB結果のリスト。
        分子ごとに1回呼び出す。
    """

    # Boltzmann定数 (Hartree/K)
    _KB_HARTREE = 3.1668114e-6

    def __init__(self, config: ConformerEnsembleConfig | None = None):
        self.config = config or ConformerEnsembleConfig()

    def extract(
        self,
        conformer_results: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        同一分子の複数conformer xTB結果から統計的特徴量を抽出する。

        Args:
            conformer_results: 各conformerの ``_parse_xtb_output()`` 出力辞書リスト。
                最低2個のconformerが必要。

        Returns:
            統計的特徴量の辞書。
        """
        features: dict[str, float] = {}
        prefix = self.config.feature_prefix
        n_conf = len(conformer_results)

        if n_conf < 2:
            logger.debug("conformer数 < 2: アンサンブル特徴量はスキップ")
            return features

        features[f"{prefix}n_conformers"] = float(n_conf)

        # Boltzmann重みの計算
        weights = self._compute_boltzmann_weights(conformer_results)
        if weights is not None:
            features[f"{prefix}boltzmann_entropy"] = float(
                -np.sum(weights * np.log(weights + 1e-30))
            )

        # 各カテゴリの特徴量を抽出
        if self.config.include_energy_stats:
            features.update(
                self._extract_energy_stats(conformer_results, weights, prefix)
            )

        if self.config.include_orbital_fluctuation:
            features.update(
                self._extract_orbital_fluctuation(conformer_results, weights, prefix)
            )

        if self.config.include_charge_fluctuation:
            features.update(
                self._extract_charge_fluctuation(conformer_results, weights, prefix)
            )

        if self.config.include_shape_stability:
            features.update(
                self._extract_shape_stability(conformer_results, weights, prefix)
            )

        return features

    # ────────────────────────────────────────────────────────
    # Boltzmann重み
    # ────────────────────────────────────────────────────────

    def _compute_boltzmann_weights(
        self,
        results: list[dict],
    ) -> np.ndarray | None:
        """
        conformerのエネルギーからBoltzmann重みを計算する。

        p_i = exp(-E_i / kT) / Σ exp(-E_j / kT)
        """
        energies = []
        for r in results:
            e = r.get("xtb_TotalEnergy")
            if e is not None:
                try:
                    energies.append(float(e))
                except (TypeError, ValueError):
                    return None
            else:
                return None

        if len(energies) < 2:
            return None

        arr = np.array(energies)
        kT = self._KB_HARTREE * self.config.temperature

        if kT < 1e-15:
            return None

        # 数値安定性のためエネルギーをシフト (最小値を0に)
        shifted = arr - np.min(arr)
        boltz = np.exp(-shifted / kT)
        total = np.sum(boltz)

        if total < 1e-30:
            return np.ones(len(energies)) / len(energies)

        return boltz / total

    # ────────────────────────────────────────────────────────
    # 特徴量抽出メソッド群
    # ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_energy_stats(
        results: list[dict],
        weights: np.ndarray | None,
        prefix: str,
    ) -> dict[str, float]:
        """conformerエネルギーの統計量を抽出する。"""
        features: dict[str, float] = {}

        energies = []
        for r in results:
            e = r.get("xtb_TotalEnergy")
            if e is not None:
                try:
                    energies.append(float(e))
                except (TypeError, ValueError):
                    pass

        if len(energies) < 2:
            return features

        arr = np.array(energies)
        features[f"{prefix}energy_mean"] = float(np.mean(arr))
        features[f"{prefix}energy_std"] = float(np.std(arr))
        features[f"{prefix}energy_range"] = float(np.max(arr) - np.min(arr))
        features[f"{prefix}energy_min"] = float(np.min(arr))

        # Boltzmann加重平均
        if weights is not None and len(weights) == len(energies):
            features[f"{prefix}energy_boltz_mean"] = float(np.sum(weights * arr))

        # kcal/mol換算のエネルギー範囲（化学的に解釈しやすい）
        hartree_to_kcal = 627.509
        features[f"{prefix}energy_range_kcal"] = float(
            (np.max(arr) - np.min(arr)) * hartree_to_kcal
        )

        return features

    @staticmethod
    def _extract_orbital_fluctuation(
        results: list[dict],
        weights: np.ndarray | None,
        prefix: str,
    ) -> dict[str, float]:
        """HOMO/LUMOエネルギーのconformer間変動を抽出する。"""
        features: dict[str, float] = {}

        for key, label in [
            ("xtb_HomoEnergy", "homo"),
            ("xtb_LumoEnergy", "lumo"),
            ("xtb_HomoLumoGap", "gap"),
        ]:
            values = []
            for r in results:
                v = r.get(key)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if len(values) >= 2:
                arr = np.array(values)
                features[f"{prefix}{label}_mean"] = float(np.mean(arr))
                features[f"{prefix}{label}_std"] = float(np.std(arr))
                features[f"{prefix}{label}_range"] = float(
                    np.max(arr) - np.min(arr)
                )

                if weights is not None and len(weights) == len(values):
                    features[f"{prefix}{label}_boltz_mean"] = float(
                        np.sum(weights * arr)
                    )

        return features

    @staticmethod
    def _extract_charge_fluctuation(
        results: list[dict],
        weights: np.ndarray | None,
        prefix: str,
    ) -> dict[str, float]:
        """Mulliken電荷統計のconformer間変動を抽出する。"""
        features: dict[str, float] = {}

        for key, label in [
            ("xtb_MullikenChargeMax", "qmax"),
            ("xtb_MullikenChargeMin", "qmin"),
            ("xtb_MullikenChargeStd", "qstd"),
        ]:
            values = []
            for r in results:
                v = r.get(key)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if len(values) >= 2:
                arr = np.array(values)
                features[f"{prefix}{label}_mean"] = float(np.mean(arr))
                features[f"{prefix}{label}_std"] = float(np.std(arr))

        return features

    @staticmethod
    def _extract_shape_stability(
        results: list[dict],
        weights: np.ndarray | None,
        prefix: str,
    ) -> dict[str, float]:
        """双極子モーメント・分極率のconformer間変動を抽出する。"""
        features: dict[str, float] = {}

        for key, label in [
            ("xtb_DipoleMoment", "dipole"),
            ("xtb_Polarizability", "polar"),
        ]:
            values = []
            for r in results:
                v = r.get(key)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if len(values) >= 2:
                arr = np.array(values)
                features[f"{prefix}{label}_mean"] = float(np.mean(arr))
                features[f"{prefix}{label}_std"] = float(np.std(arr))
                features[f"{prefix}{label}_cv"] = float(
                    np.std(arr) / (np.mean(arr) + 1e-10)
                )  # 変動係数 (coefficient of variation)

        return features
