"""ベイズ最適化コアエンジン.

Implements: F-A01〜A10
    BayesianOptimizer: GPフィッティング、獲得関数、KB法バッチ、ParEGO多目的
    獲得関数: EI, PI, UCB, PTR
    バッチ戦略: Kriging Believer, DoE+BOハイブリッド
    多目的: ParEGO (Chebyshev scalarization + EI)

参考文献:
    - Kriging Believer: Ginsbourger et al. (2010) "Kriging is well-suited
      to parallelize optimization"
    - ParEGO: Knowles (2006) "ParEGO: A hybrid algorithm with on-line
      landscape approximation for expensive multiobjective optimization"
    - PTR: Bayesian target-range optimization via CDF difference
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    Kernel,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler


# ─── 設定 dataclass ──────────────────────────────────────
@dataclass
class BOConfig:
    """ベイズ最適化の設定.

    Attributes:
        objective: "maximize" / "minimize" / "target_range"
        acquisition: "ei" / "pi" / "ucb" / "ptr"
        xi: EI/PI の探索パラメータ (ξ)
        kappa: UCB のパラメータ (κ)
        target_lo: PTR 目標範囲下限
        target_hi: PTR 目標範囲上限
        kernel_type: "default" / "matern" / "dotproduct" / "custom"
        matern_nu: Maternカーネルの平滑度 (0.5, 1.5, 2.5)
        batch_strategy: "single" / "kriging_believer" / "doe_then_bo" / "bo_then_doe"
        n_candidates: バッチ候補数
        multi_objective: 多目的最適化を使用するか
        objective_columns: 多目的用: 目的変数列リスト
        objective_directions: 多目的用: 各目的の方向 ("max" / "min")
        parego_rho: ParEGO の ρ パラメータ
    """

    objective: str = "minimize"
    acquisition: str = "ei"
    xi: float = 0.01
    kappa: float = 2.0
    target_lo: float | None = None
    target_hi: float | None = None
    kernel_type: str = "default"
    matern_nu: float = 2.5
    batch_strategy: str = "kriging_believer"
    n_candidates: int = 5
    multi_objective: bool = False
    objective_columns: list[str] = field(default_factory=list)
    objective_directions: list[str] = field(default_factory=list)
    parego_rho: float = 0.05


class BayesianOptimizer:
    """ベイズ最適化エンジン.

    Usage::
        bo = BayesianOptimizer(config=BOConfig(objective="minimize"))
        bo.fit(X_train, y_train)
        candidates = bo.suggest(X_candidates, n=5)
    """

    def __init__(self, config: BOConfig | None = None) -> None:
        self.config = config or BOConfig()
        self._gp: GaussianProcessRegressor | None = None
        self._gps: list[GaussianProcessRegressor] = []  # 多目的用
        self._scaler_X: StandardScaler | None = None
        self._scaler_y: StandardScaler | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._y_best: float | None = None
        self._is_fitted = False

    # ── カーネル構築 ──
    def _build_kernel(self) -> Kernel:
        """設定に基づいてカーネルを構築."""
        cfg = self.config
        if cfg.kernel_type == "matern":
            return Matern(nu=cfg.matern_nu) + WhiteKernel()
        elif cfg.kernel_type == "dotproduct":
            return DotProduct() + WhiteKernel()
        else:  # default: DotProduct + Matern + White
            return DotProduct() + Matern(nu=cfg.matern_nu) + WhiteKernel()

    # ── フィッティング ──
    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | pd.DataFrame,
    ) -> "BayesianOptimizer":
        """既存データでGPを学習.

        Args:
            X: 特徴量 (n_samples, n_features)
            y: 目的変数。単目的: (n_samples,)、多目的: (n_samples, n_objectives)

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        # スケーリング
        self._scaler_X = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X_arr)

        if self.config.multi_objective and y_arr.ndim == 2:
            self._fit_multi(X_scaled, y_arr)
        else:
            self._fit_single(X_scaled, y_arr.ravel())

        self._X_train = X_scaled
        self._is_fitted = True
        return self

    def _fit_single(self, X: np.ndarray, y: np.ndarray) -> None:
        """単目的GPフィッティング."""
        self._scaler_y = StandardScaler()
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        kernel = self._build_kernel()
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=False,
            alpha=1e-6,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(X, y_scaled)

        self._y_train = y_scaled
        # y_bestの設定（最小化用にスケーリングされた値）
        if self.config.objective == "maximize":
            self._y_best = float(np.max(y_scaled))
        else:
            self._y_best = float(np.min(y_scaled))

    def _fit_multi(self, X: np.ndarray, Y: np.ndarray) -> None:
        """多目的GPフィッティング（各目的に独立GP）."""
        self._gps = []
        self._scaler_y = None  # 多目的では個別
        n_obj = Y.shape[1]

        for j in range(n_obj):
            kernel = self._build_kernel()
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                alpha=1e-6,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, Y[:, j])
            self._gps.append(gp)

        self._y_train = Y

    # ── 獲得関数 ──
    def _acquisition(
        self,
        X_cand: np.ndarray,
        gp: GaussianProcessRegressor | None = None,
        y_best: float | None = None,
    ) -> np.ndarray:
        """獲得関数を計算.

        Args:
            X_cand: 候補点 (n_candidates, n_features) — スケーリング済
            gp: GPモデル（Noneの場合はself._gp）
            y_best: 現在の最良値（Noneの場合はself._y_best）

        Returns:
            獲得関数値 (n_candidates,)
        """
        if gp is None:
            gp = self._gp
        if y_best is None:
            y_best = self._y_best

        mu, sigma = gp.predict(X_cand, return_std=True)  # type: ignore[union-attr]
        sigma = np.maximum(sigma, 1e-9)

        acq = self.config.acquisition.lower()

        if acq == "ei":
            return self._ei(mu, sigma, y_best)  # type: ignore[arg-type]
        elif acq == "pi":
            return self._pi(mu, sigma, y_best)  # type: ignore[arg-type]
        elif acq == "ucb":
            return self._ucb(mu, sigma)
        elif acq == "ptr":
            return self._ptr(mu, sigma)
        else:
            raise ValueError(f"不明な獲得関数: {acq}")

    def _ei(self, mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
        """Expected Improvement.

        EI(x) = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
        minimize: (y_best - mu - xi) * Phi(Z) + sigma * phi(Z)
        """
        xi = self.config.xi
        if self.config.objective == "maximize":
            improvement = mu - y_best - xi
        else:
            improvement = y_best - mu - xi

        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _pi(self, mu: np.ndarray, sigma: np.ndarray, y_best: float) -> np.ndarray:
        """Probability of Improvement.

        PI(x) = Phi(Z), Z = (mu - y_best - xi) / sigma
        """
        xi = self.config.xi
        if self.config.objective == "maximize":
            Z = (mu - y_best - xi) / sigma
        else:
            Z = (y_best - mu - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma < 1e-9] = 0.0
        return pi

    def _ucb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound.

        UCB(x) = mu + kappa * sigma (maximize)
        LCB(x) = -mu + kappa * sigma (minimize)
        """
        kappa = self.config.kappa
        if self.config.objective == "maximize":
            return mu + kappa * sigma
        else:
            return -mu + kappa * sigma

    def _ptr(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Probability of Target Range.

        PTR(x) = Phi((hi - mu) / sigma) - Phi((lo - mu) / sigma)
        """
        lo = self.config.target_lo
        hi = self.config.target_hi
        if lo is None or hi is None:
            raise ValueError("PTR には target_lo と target_hi の設定が必要。")

        # スケーリングされた目標範囲
        if self._scaler_y is not None:
            lo_s = (lo - self._scaler_y.mean_[0]) / self._scaler_y.scale_[0]
            hi_s = (hi - self._scaler_y.mean_[0]) / self._scaler_y.scale_[0]
        else:
            lo_s, hi_s = lo, hi

        ptr = norm.cdf((hi_s - mu) / sigma) - norm.cdf((lo_s - mu) / sigma)
        ptr[sigma < 1e-9] = 0.0
        return ptr

    # ── 候補提案 ──
    def suggest(
        self,
        X_candidates: np.ndarray | pd.DataFrame,
        n: int | None = None,
    ) -> pd.DataFrame | np.ndarray:
        """次の実験候補を提案.

        Args:
            X_candidates: 探索候補点
            n: 提案数（Noneの場合はconfig.n_candidates）

        Returns:
            上位n候補（DataFrame or ndarray）、獲得関数値付き
        """
        if not self._is_fitted:
            raise RuntimeError("fit() を先に呼んでください。")
        if n is None:
            n = self.config.n_candidates

        is_df = isinstance(X_candidates, pd.DataFrame)
        list(X_candidates.columns) if is_df else None
        X_cand = np.asarray(X_candidates, dtype=np.float64)
        X_cand_scaled = self._scaler_X.transform(X_cand)  # type: ignore[union-attr]

        strategy = self.config.batch_strategy

        if self.config.multi_objective and self._gps:
            result_idx, acq_vals = self._suggest_parego(X_cand_scaled, n)
        elif strategy == "single" or n == 1:
            result_idx, acq_vals = self._suggest_single(X_cand_scaled, n)
        elif strategy == "kriging_believer":
            result_idx, acq_vals = self._kriging_believer(X_cand_scaled, n)
        elif strategy == "doe_then_bo":
            result_idx, acq_vals = self._doe_then_bo(X_cand_scaled, n)
        elif strategy == "bo_then_doe":
            result_idx, acq_vals = self._bo_then_doe(X_cand_scaled, n)
        else:
            result_idx, acq_vals = self._suggest_single(X_cand_scaled, n)

        # 結果構築
        if is_df:
            result = X_candidates.iloc[result_idx].copy().reset_index(drop=True)
            result["_acq_value"] = acq_vals
            result["_rank"] = range(1, len(result) + 1)
        else:
            result = X_cand[result_idx]
        return result

    def _suggest_single(
        self, X_cand: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """単純にtop-nを返す."""
        acq_vals = self._acquisition(X_cand)
        top_idx = np.argsort(acq_vals)[::-1][:n]
        return top_idx, acq_vals[top_idx]

    def _kriging_believer(
        self, X_cand: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kriging Believer法でバッチ候補を生成.

        逐次的にGP予測平均を仮想観測として追加し、
        多様な候補を生成する。

        参考: Ginsbourger et al. (2010)
        """
        gp_copy = copy.deepcopy(self._gp)
        X_obs = self._X_train.copy()  # type: ignore[union-attr]
        y_obs = self._y_train.copy()  # type: ignore[union-attr]
        y_best = self._y_best

        selected_idx: list[int] = []
        acq_values: list[float] = []
        remaining = set(range(len(X_cand)))

        for _ in range(min(n, len(X_cand))):
            if not remaining:
                break

            remaining_list = sorted(remaining)
            X_rem = X_cand[remaining_list]

            # 獲得関数を計算
            acq = self._acquisition(X_rem, gp=gp_copy, y_best=y_best)  # type: ignore[arg-type]
            best_local = np.argmax(acq)
            best_global = remaining_list[best_local]

            selected_idx.append(best_global)
            acq_values.append(float(acq[best_local]))
            remaining.discard(best_global)

            # 仮想観測を追加してGPを更新
            x_new = X_cand[best_global:best_global + 1]
            y_fake = gp_copy.predict(x_new)  # 予測平均を仮想観測とする
            X_obs = np.vstack([X_obs, x_new])
            y_obs = np.append(y_obs, y_fake)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_copy.fit(X_obs, y_obs)

            # y_best更新
            if self.config.objective == "maximize":
                y_best = float(np.max(y_obs))
            else:
                y_best = float(np.min(y_obs))

        return np.array(selected_idx), np.array(acq_values)

    def _doe_then_bo(
        self, X_cand: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """DoE→BO ハイブリッド.

        Step 1: 候補からmaximin距離で多様な部分集合を選出
        Step 2: その部分集合から獲得関数の高い順にn個を選ぶ
        """
        # Step 1: maximin ベースで多様な候補を3n個選出
        n_diverse = min(n * 3, len(X_cand))
        diverse_idx = self._maximin_select(X_cand, n_diverse)

        # Step 2: その中から獲得関数top-n
        X_diverse = X_cand[diverse_idx]
        acq_vals = self._acquisition(X_diverse)
        top_local = np.argsort(acq_vals)[::-1][:n]

        result_idx = diverse_idx[top_local]
        return result_idx, acq_vals[top_local]

    def _bo_then_doe(
        self, X_cand: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """BO→DoE ハイブリッド.

        Step 1: 獲得関数top 3nを選出
        Step 2: その中からmaximin距離でn個の多様な候補を選ぶ
        """
        # Step 1: 獲得関数top 3n
        n_top = min(n * 3, len(X_cand))
        acq_vals = self._acquisition(X_cand)
        top_idx = np.argsort(acq_vals)[::-1][:n_top]

        # Step 2: maximin
        X_top = X_cand[top_idx]
        diverse_local = self._maximin_select(X_top, n)

        result_idx = top_idx[diverse_local]
        return result_idx, acq_vals[result_idx]

    def _suggest_parego(
        self, X_cand: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """ParEGO: 多目的→単目的スカラ化 + EI.

        ランダム重みでChebyshev scalarizationを行い、
        各重みベクトルでの最良候補を集める。

        参考: Knowles (2006)
        """
        n_obj = len(self._gps)
        rng = np.random.RandomState(42)

        # 各GPから予測
        predictions = []
        for gp in self._gps:
            mu_j, _ = gp.predict(X_cand, return_std=True)
            predictions.append(mu_j)
        predictions = np.column_stack(predictions)

        # 既存データのY値で正規化
        Y_train = self._y_train
        Y_min = Y_train.min(axis=0)  # type: ignore[union-attr]
        Y_max = Y_train.max(axis=0)  # type: ignore[union-attr]
        Y_range = Y_max - Y_min
        Y_range[Y_range < 1e-9] = 1.0

        # 方向を統一（最小化に変換）
        dirs = self.config.objective_directions
        if not dirs:
            dirs = ["min"] * n_obj

        selected_idx: list[int] = []
        acq_values: list[float] = []
        remaining = set(range(len(X_cand)))

        for i in range(n):
            if not remaining:
                break

            # Dirichlet からランダム重み
            weights = rng.dirichlet(np.ones(n_obj))

            remaining_list = sorted(remaining)
            X_rem = X_cand[remaining_list]

            # 各GPから予測
            preds_rem = []
            for gp in self._gps:
                mu_j, _ = gp.predict(X_rem, return_std=True)
                preds_rem.append(mu_j)
            preds_rem = np.column_stack(preds_rem)

            # 正規化
            preds_norm = (preds_rem - Y_min) / Y_range

            # 方向変換（最大化→最小化: 1 - val）
            for j, d in enumerate(dirs):
                if d == "max":
                    preds_norm[:, j] = 1.0 - preds_norm[:, j]

            # Chebyshev scalarization
            rho = self.config.parego_rho
            scalarized = np.max(weights * preds_norm, axis=1) + rho * np.sum(
                weights * preds_norm, axis=1
            )

            # 最小値が最良（最小化に統一済み）
            best_local = np.argmin(scalarized)
            best_global = remaining_list[best_local]

            selected_idx.append(best_global)
            acq_values.append(float(-scalarized[best_local]))  # 負にして大きいほど良くする
            remaining.discard(best_global)

        return np.array(selected_idx), np.array(acq_values)

    # ── ヘルパー ──
    @staticmethod
    def _maximin_select(X: np.ndarray, n: int) -> np.ndarray:
        """Maximin距離ベースで多様なn個を選出.

        貪欲法: 最初にランダム1点、以降は「既選択点群との最小距離が最大」の点を追加。
        """
        if n >= len(X):
            return np.arange(len(X))

        selected: list[int] = [0]
        remaining = set(range(1, len(X)))

        for _ in range(n - 1):
            if not remaining:
                break
            rem_list = np.array(sorted(remaining))
            X_sel = X[selected]
            X_rem = X[rem_list]

            # 各候補と既選択点群の距離を計算
            # (n_rem, n_sel) の距離行列
            dists = np.linalg.norm(
                X_rem[:, None, :] - X_sel[None, :, :], axis=2
            )
            min_dists = dists.min(axis=1)  # 各候補の最近接距離
            best = np.argmax(min_dists)

            selected.append(rem_list[best])
            remaining.discard(rem_list[best])

        return np.array(selected)

    def predict(
        self, X: np.ndarray | pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """GPの予測平均と標準偏差を返す.

        Returns:
            (mu, sigma) — 元のスケール
        """
        if not self._is_fitted:
            raise RuntimeError("fit() を先に呼んでください。")

        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler_X.transform(X_arr)  # type: ignore[union-attr]

        if self.config.multi_objective and self._gps:
            mus, sigmas = [], []
            for gp in self._gps:
                mu, sigma = gp.predict(X_scaled, return_std=True)
                mus.append(mu)
                sigmas.append(sigma)
            return np.column_stack(mus), np.column_stack(sigmas)

        mu_s, sigma_s = self._gp.predict(X_scaled, return_std=True)  # type: ignore[union-attr]
        # 逆スケーリング
        if self._scaler_y is not None:
            mu = mu_s * self._scaler_y.scale_[0] + self._scaler_y.mean_[0]
            sigma = sigma_s * self._scaler_y.scale_[0]
        else:
            mu, sigma = mu_s, sigma_s
        return mu, sigma

    def get_gp_info(self) -> dict[str, Any]:
        """GPモデルの情報を取得."""
        if not self._is_fitted:
            return {}
        if self._gp is not None:
            return {
                "kernel": str(self._gp.kernel_),
                "log_marginal_likelihood": float(self._gp.log_marginal_likelihood_value_),
                "n_train": len(self._X_train) if self._X_train is not None else 0,
            }
        if self._gps:
            return {
                "n_objectives": len(self._gps),
                "kernels": [str(gp.kernel_) for gp in self._gps],
                "n_train": len(self._X_train) if self._X_train is not None else 0,
            }
        return {}
