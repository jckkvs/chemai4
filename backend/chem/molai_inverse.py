"""
backend/chem/molai_inverse.py

MolAI 潜在空間での逆解析・分子探索モジュール。

MolAIのエンコード/デコード能力を活用し、
PCA低次元空間上で目標物性値を満たす分子を探索する。

ワークフロー:
  既知SMILES → Encoder → 潜在ベクトル → PCA圧縮
                                          ↓
                               PCA空間で探索（BO/ランダム/補間/グリッド）
                                          ↓
                               PCA逆変換 → 潜在ベクトル
                                          ↓
                               Decoder → 新規SMILES候補
                                          ↓
                               RDKit検証（有効SMILES？）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── データクラス ──────────────────────────────────────────────────────────
@dataclass
class MolCandidate:
    """逆解析で発見された候補分子"""
    smiles: str
    pc_vector: np.ndarray          # PCA空間上の座標
    predicted_value: float | None = None  # モデルによる予測物性値
    is_valid: bool = False         # RDKitで有効なSMILESか
    is_novel: bool = False         # 学習データに無い新規分子か
    similarity_to_nearest: float = 0.0  # 最近傍学習分子との類似度


@dataclass
class DegeneracyMap:
    """PCA空間の縮退マップ: 異なるPC値が同一分子にデコードされる領域"""
    grid_points: np.ndarray        # グリッド点 (N, n_pca_dims)
    decoded_smiles: list[str]      # 各点のデコード結果
    unique_smiles: list[str]       # ユニーク分子一覧
    smiles_to_label: dict[str, int]  # SMILES→ラベル番号
    labels: np.ndarray             # 各点のラベル番号 (N,)


# ── 逆解析エンジン ────────────────────────────────────────────────────────
class MolAIInverseAnalyzer:
    """MolAI潜在空間での逆解析・分子探索

    Args:
        adapter: 学習済みMolAIAdapter（compute + train_autoencoder済み）
    """

    def __init__(self, adapter: Any):
        from backend.chem.molai_adapter import MolAIAdapter
        if not isinstance(adapter, MolAIAdapter):
            raise TypeError("MolAIAdapter のインスタンスが必要です")
        self.adapter = adapter
        self._training_smiles: list[str] = []
        self._training_pc: np.ndarray | None = None

    def set_training_data(self, smiles_list: list[str], pc_vectors: np.ndarray):
        """学習データを登録（探索範囲の決定とnovelty判定用）"""
        self._training_smiles = list(smiles_list)
        self._training_pc = pc_vectors.copy()

    # ─── 探索メソッド ─────────────────────────────────────────────────
    def explore_random(
        self,
        model: Any,
        target_value: float,
        n_candidates: int = 100,
        n_samples: int = 5000,
        maximize: bool = False,
    ) -> list[MolCandidate]:
        """ランダムサンプリング + モデル予測 → 上位N件を返す。

        PCA空間上で学習データの分布範囲内からランダムにサンプリングし、
        学習済みモデルで物性値を予測 → target_valueに最も近い候補を返す。
        """
        if self._training_pc is None:
            raise RuntimeError("先に set_training_data() で学習データを登録してください。")

        # 学習データの分布からサンプリング範囲を決定
        pc_mean = np.nanmean(self._training_pc, axis=0)
        pc_std = np.nanstd(self._training_pc, axis=0)
        pc_min = pc_mean - 3 * pc_std
        pc_max = pc_mean + 3 * pc_std

        n_dims = self._training_pc.shape[1]
        samples = np.random.uniform(
            pc_min, pc_max, size=(n_samples, n_dims)
        ).astype(np.float32)

        # デコード
        decoded = self.adapter.decode_from_pca(samples)

        # RDKit有効性チェック + 予測
        candidates = self._evaluate_candidates(
            samples, decoded, model, target_value, maximize
        )

        # ターゲットに近い順にソート
        candidates.sort(
            key=lambda c: abs(c.predicted_value - target_value)
            if c.predicted_value is not None else float("inf")
        )

        return candidates[:n_candidates]

    def explore_interpolation(
        self,
        smiles_a: str,
        smiles_b: str,
        n_steps: int = 20,
        model: Any = None,
    ) -> list[MolCandidate]:
        """2分子間の潜在空間補間。

        分子Aと分子Bの潜在ベクトルを線形補間し、
        各中間点をデコードして新しい分子を生成。
        """
        # エンコード → PCA
        raw_a = self.adapter.encode_raw([smiles_a])
        raw_b = self.adapter.encode_raw([smiles_b])

        if self.adapter._pca is None or self.adapter._scaler is None:
            raise RuntimeError("先に compute() を実行してください")

        pc_a = self.adapter._pca.transform(
            self.adapter._scaler.transform(raw_a)
        )[0]
        pc_b = self.adapter._pca.transform(
            self.adapter._scaler.transform(raw_b)
        )[0]

        # 線形補間
        alphas = np.linspace(0, 1, n_steps)
        pc_interp = np.array([
            pc_a * (1 - a) + pc_b * a for a in alphas
        ], dtype=np.float32)

        # デコード
        decoded = self.adapter.decode_from_pca(pc_interp)

        target_value = 0.0  # 補間ではターゲット不要
        candidates = self._evaluate_candidates(
            pc_interp, decoded, model, target_value, False
        )

        return candidates

    def explore_bayesian(
        self,
        model: Any,
        target_value: float,
        n_candidates: int = 50,
        n_iterations: int = 100,
        maximize: bool = False,
    ) -> list[MolCandidate]:
        """ベイズ最適化で目標物性値の分子を探索。

        PCA空間上でガウス過程回帰を用いた
        Expected Improvement獲得関数で最適点を逐次探索する。
        """
        if self._training_pc is None:
            raise RuntimeError("先に set_training_data() で学習データを登録してください。")

        from scipy.optimize import minimize

        pc_mean = np.nanmean(self._training_pc, axis=0)
        pc_std = np.nanstd(self._training_pc, axis=0)
        bounds = list(zip(
            (pc_mean - 3 * pc_std).tolist(),
            (pc_mean + 3 * pc_std).tolist(),
        ))
        n_dims = len(bounds)

        # 収集した候補点
        all_pc = []
        all_pred = []

        # 初期ランダム探索
        n_init = min(20, n_iterations // 3)
        init_samples = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(n_init, n_dims),
        ).astype(np.float32)

        for pc in init_samples:
            decoded = self.adapter.decode_from_pca(pc.reshape(1, -1))
            if decoded and decoded[0]:
                try:
                    pred = float(model.predict(
                        pd.DataFrame([pc], columns=[f"molai_pc{k+1}" for k in range(n_dims)])
                    )[0])
                    all_pc.append(pc)
                    all_pred.append(pred)
                except Exception:
                    pass

        # 逐次最適化
        for it in range(n_init, n_iterations):
            if len(all_pc) < 2:
                # 追加ランダムサンプル
                pc = np.random.uniform(
                    [b[0] for b in bounds],
                    [b[1] for b in bounds],
                    size=(n_dims,),
                ).astype(np.float32)
            else:
                # 簡易サロゲートモデル（RBFカーネルによる補間）
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, ConstantKernel

                X_obs = np.array(all_pc)
                y_obs = np.array(all_pred)
                # 目標との距離を最小化したい
                y_dist = np.abs(y_obs - target_value)

                kernel = ConstantKernel() * RBF()
                gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
                try:
                    gpr.fit(X_obs, y_dist)
                except Exception:
                    # フォールバック: ランダム
                    pc = np.random.uniform(
                        [b[0] for b in bounds],
                        [b[1] for b in bounds],
                        size=(n_dims,),
                    ).astype(np.float32)
                    decoded = self.adapter.decode_from_pca(pc.reshape(1, -1))
                    if decoded and decoded[0]:
                        try:
                            pred = float(model.predict(
                                pd.DataFrame([pc], columns=[f"molai_pc{k+1}" for k in range(n_dims)])
                            )[0])
                            all_pc.append(pc)
                            all_pred.append(pred)
                        except Exception:
                            pass
                    continue

                # EI(Expected Improvement)相当: 距離が最小になる点を探索
                def _acquisition(x):
                    mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)
                    return float(mu[0])  # 距離の予測値を最小化

                # 多点初期化で局所最適を回避
                best_x = None
                best_val = float("inf")
                for _ in range(5):
                    x0 = np.random.uniform(
                        [b[0] for b in bounds],
                        [b[1] for b in bounds],
                    )
                    try:
                        res = minimize(_acquisition, x0, bounds=bounds, method="L-BFGS-B")
                        if res.fun < best_val:
                            best_val = res.fun
                            best_x = res.x
                    except Exception:
                        pass
                pc = best_x if best_x is not None else np.random.uniform(
                    [b[0] for b in bounds], [b[1] for b in bounds]
                )
                pc = pc.astype(np.float32)

            decoded = self.adapter.decode_from_pca(pc.reshape(1, -1))
            if decoded and decoded[0]:
                try:
                    pred = float(model.predict(
                        pd.DataFrame([pc], columns=[f"molai_pc{k+1}" for k in range(n_dims)])
                    )[0])
                    all_pc.append(pc)
                    all_pred.append(pred)
                except Exception:
                    pass

        # 結果をMolCandidateに変換
        if not all_pc:
            return []

        all_pc_arr = np.array(all_pc, dtype=np.float32)
        decoded_all = self.adapter.decode_from_pca(all_pc_arr)
        candidates = self._evaluate_candidates(
            all_pc_arr, decoded_all, model, target_value, maximize
        )
        candidates.sort(
            key=lambda c: abs(c.predicted_value - target_value)
            if c.predicted_value is not None else float("inf")
        )
        return candidates[:n_candidates]

    # ─── 縮退可視化 ───────────────────────────────────────────────────
    def compute_degeneracy_map(
        self,
        pc_dim1: int = 0,
        pc_dim2: int = 1,
        n_grid: int = 30,
        fixed_dims: np.ndarray | None = None,
    ) -> DegeneracyMap:
        """PCA空間の縮退マップを計算。

        2つのPC軸を格子状にスキャンし、各点をデコード。
        同じSMILESにデコードされる領域を可視化用にマッピングする。

        Args:
            pc_dim1: X軸に使うPC次元 (0-indexed)
            pc_dim2: Y軸に使うPC次元 (0-indexed)
            n_grid: 各軸のグリッド分割数
            fixed_dims: 固定するPC次元の値（Noneの場合は平均値）
        Returns:
            DegeneracyMap: 各グリッド点のデコード結果と縮退ラベル
        """
        if self._training_pc is None:
            raise RuntimeError("先に set_training_data() で学習データを登録してください。")

        pc_mean = np.nanmean(self._training_pc, axis=0)
        pc_std = np.nanstd(self._training_pc, axis=0)
        self._training_pc.shape[1]

        # 基準ベクトル（固定次元の値）
        if fixed_dims is not None:
            base = fixed_dims.copy()
        else:
            base = pc_mean.copy()

        # 2軸のスキャン範囲
        range1 = np.linspace(
            pc_mean[pc_dim1] - 2 * pc_std[pc_dim1],
            pc_mean[pc_dim1] + 2 * pc_std[pc_dim1],
            n_grid,
        )
        range2 = np.linspace(
            pc_mean[pc_dim2] - 2 * pc_std[pc_dim2],
            pc_mean[pc_dim2] + 2 * pc_std[pc_dim2],
            n_grid,
        )

        # グリッド生成
        grid_points = []
        for v1 in range1:
            for v2 in range2:
                pt = base.copy()
                pt[pc_dim1] = v1
                pt[pc_dim2] = v2
                grid_points.append(pt)
        grid_arr = np.array(grid_points, dtype=np.float32)

        # デコード
        decoded = self.adapter.decode_from_pca(grid_arr)

        # ユニーク分子のラベル付け
        unique_smiles = list(dict.fromkeys(decoded))  # 順序保持
        smiles_to_label = {s: i for i, s in enumerate(unique_smiles)}
        labels = np.array([smiles_to_label.get(s, -1) for s in decoded])

        logger.info(
            f"縮退マップ: {n_grid}x{n_grid}={len(grid_arr)}点 → "
            f"{len(unique_smiles)}種類のユニーク分子"
        )

        return DegeneracyMap(
            grid_points=grid_arr,
            decoded_smiles=decoded,
            unique_smiles=unique_smiles,
            smiles_to_label=smiles_to_label,
            labels=labels,
        )

    # ─── 内部ヘルパー ─────────────────────────────────────────────────
    def _evaluate_candidates(
        self,
        pc_vectors: np.ndarray,
        decoded_smiles: list[str],
        model: Any,
        target_value: float,
        maximize: bool,
    ) -> list[MolCandidate]:
        """デコード結果をMolCandidateリストに変換し、有効性・予測値を付与。"""
        candidates = []
        n_dims = pc_vectors.shape[1]
        col_names = [f"molai_pc{k+1}" for k in range(n_dims)]

        for i, (smi, pc) in enumerate(zip(decoded_smiles, pc_vectors)):
            cand = MolCandidate(smiles=smi, pc_vector=pc.copy())

            # RDKit有効性チェック
            if smi:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smi)
                    cand.is_valid = mol is not None
                except ImportError:
                    cand.is_valid = bool(smi)
            else:
                cand.is_valid = False

            # 新規性チェック
            if self._training_smiles:
                cand.is_novel = smi not in self._training_smiles

            # モデル予測
            if model is not None and cand.is_valid:
                try:
                    pred = float(model.predict(
                        pd.DataFrame([pc], columns=col_names)
                    )[0])
                    cand.predicted_value = pred
                except Exception:
                    pass

            # 最近傍類似度
            if self._training_smiles and cand.is_valid:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import DataStructs, AllChem
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        max_sim = 0.0
                        for tr_smi in self._training_smiles[:100]:  # 最大100件で計算
                            tr_mol = Chem.MolFromSmiles(tr_smi)
                            if tr_mol:
                                tr_fp = AllChem.GetMorganFingerprintAsBitVect(tr_mol, 2, nBits=2048)
                                sim = DataStructs.TanimotoSimilarity(fp, tr_fp)
                                max_sim = max(max_sim, sim)
                        cand.similarity_to_nearest = max_sim
                except ImportError:
                    pass

            if cand.is_valid:
                candidates.append(cand)

        return candidates
