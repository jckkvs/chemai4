"""
backend/optim/composition_sampler.py

ディリクレ分布を用いた組成系サンプリング。

Implements: F-DIR01〜DIR05
    DirichletSampler: αパラメータ適応型ディリクレサンプラー
    sample: 組成制約付きサンプリング
    update_alpha: Thompson sampling によるα更新
    constrained_sample: RangeConstraint 対応版

設計思想:
    - 組成変数（合計=定数）に特化したサンプリング戦略
    - ベイズ的にαを更新し、良い候補の領域に探索を集中
    - 既存の SumConstraint / RangeConstraint と統合可能

参考文献:
    Aitchison (1986) "The Statistical Analysis of Compositional Data"
    原文: "The sample space for compositional data is the simplex ...
           The Dirichlet distribution is the natural conjugate prior
           for multinomial sampling on the simplex."
    訳: 組成データの標本空間はシンプレックスであり、
        ディリクレ分布はシンプレックス上の多項分布の自然共役事前分布である。

    Frigyik et al. (2010) "Introduction to the Dirichlet Distribution
           and Related Processes"
    原文: "The Dirichlet distribution ... parameterized by a vector α
           of positive reals, is a distribution over the open standard
           (K-1)-simplex."
    訳: ディリクレ分布はα（正実数ベクトル）でパラメータ化され、
        開標準(K-1)-シンプレックス上の分布である。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DirichletSampler:
    """ディリクレ分布ベースの組成系サンプラー。

    Attributes:
        columns: 組成変数の列名リスト
        target_sum: 合計の目標値（デフォルト100.0 → wt%）
        alpha: ディリクレ分布のαパラメータ（各成分の集中度）
        min_values: 各成分の下限値（Noneなら制約なし）
        max_values: 各成分の上限値（Noneなら制約なし）
        seed: 乱数シード
        _rng: numpy RandomGenerator (内部管理)

    Complexity: 4
    Description: 適応型ディリクレ分布サンプリングエンジン。
                 αをBayesian Thompson samplingで更新する。
    """

    columns: list[str]
    target_sum: float = 100.0
    alpha: np.ndarray | None = None
    min_values: dict[str, float] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    _rng: Any = field(default=None, repr=False)

    # 更新履歴
    _update_count: int = field(default=0, repr=False)
    _alpha_history: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """初期化: αをデフォルト設定、RNG生成。"""
        k = len(self.columns)
        if self.alpha is None:
            # 均一ディリクレ（全成分等確率）
            self.alpha = np.ones(k, dtype=np.float64)
        else:
            self.alpha = np.asarray(self.alpha, dtype=np.float64)
            if len(self.alpha) != k:
                raise ValueError(
                    f"αの長さ({len(self.alpha)})と列数({k})が不一致"
                )
        if np.any(self.alpha <= 0):
            raise ValueError("αはすべて正の実数でなければなりません")

        self._rng = np.random.default_rng(self.seed)
        self._alpha_history.append(self.alpha.copy())

    @property
    def k(self) -> int:
        """成分数。"""
        return len(self.columns)

    def sample(self, n: int) -> pd.DataFrame:
        """ディリクレ分布からn点をサンプリングする。

        Args:
            n: サンプリング数

        Returns:
            DataFrame[columns] — 各行の合計が target_sum に等しい

        Implements: F-DIR01
        論文: Aitchison (1986) Theorem 2.1 — ディリクレ分布の基本性質
        """
        raw = self._rng.dirichlet(self.alpha, size=n)
        # 合計を target_sum にスケーリング
        scaled = raw * self.target_sum
        return pd.DataFrame(scaled, columns=self.columns)

    def constrained_sample(
        self,
        n: int,
        max_attempts: int = 100,
    ) -> pd.DataFrame:
        """範囲制約付きサンプリング（棄却法）。

        min_values / max_values で指定された範囲制約を満たすサンプルのみを返す。
        棄却率が高い場合は警告を出す。

        Args:
            n: 必要なサンプル数
            max_attempts: 最大試行倍率（n × max_attempts まで生成）

        Returns:
            DataFrame — 制約を満たす n 行（不足時は得られた分だけ）

        Implements: F-DIR02
        論文: Frigyik et al. (2010) Section 4 — Truncated Dirichlet
        """
        collected: list[pd.DataFrame] = []
        total_generated = 0
        total_accepted = 0

        batch_size = max(n * 5, 1000)

        while total_accepted < n and total_generated < n * max_attempts:
            batch = self.sample(batch_size)
            total_generated += len(batch)

            # 範囲制約適用
            mask = pd.Series(True, index=batch.index)
            for col in self.columns:
                if col in self.min_values:
                    mask &= batch[col] >= self.min_values[col]
                if col in self.max_values:
                    mask &= batch[col] <= self.max_values[col]

            accepted = batch[mask]
            if len(accepted) > 0:
                collected.append(accepted)
                total_accepted += len(accepted)

        if not collected:
            logger.warning(
                f"DirichletSampler: {total_generated}点生成したが"
                f"制約を満たすサンプルが0点。制約を緩和してください。"
            )
            return pd.DataFrame(columns=self.columns)

        result = pd.concat(collected, ignore_index=True).head(n)

        accept_rate = total_accepted / max(total_generated, 1)
        if accept_rate < 0.01:
            logger.warning(
                f"DirichletSampler: 採択率が非常に低い({accept_rate:.2%})。"
                f"制約が厳しすぎる可能性があります。α調整を推奨。"
            )
        elif accept_rate < 0.1:
            logger.info(
                f"DirichletSampler: 採択率={accept_rate:.2%} "
                f"({total_accepted}/{total_generated})"
            )

        return result

    def update_alpha(
        self,
        good_samples: pd.DataFrame,
        learning_rate: float = 0.3,
        min_alpha: float = 0.1,
    ) -> np.ndarray:
        """良い候補に基づきαをベイズ的に更新する。

        Thompson sampling の考え方に基づき、性能の良かったサンプルの
        組成比率をα更新の「観測データ」として利用する。

        更新式:
            α_new = (1 - lr) * α_old + lr * (α_pseudo + n_obs * x̄)
        ここで x̄ は good_samples の組成平均（target_sum正規化後）

        Args:
            good_samples: 性能が良かったサンプル群（DataFrame[columns]）
            learning_rate: 更新の強度 (0〜1)
            min_alpha: αの下限（ゼロ回避）

        Returns:
            更新後のα

        Implements: F-DIR03
        論文: Aitchison (1986) Chapter 11 — Bayesian analysis on the simplex
        原文: "The posterior Dirichlet is Dir(α + n₁, ..., α + nₖ) where
               nᵢ are the observed category counts."
        訳: 事後ディリクレ分布は Dir(α + n₁, ..., α + nₖ) であり、
            nᵢ は観測カテゴリカウント。
        """
        if good_samples.empty:
            logger.debug("DirichletSampler.update_alpha: 空のgood_samples、更新スキップ")
            return self.alpha.copy()

        # 組成比率に正規化
        good_arr = good_samples[self.columns].values
        row_sums = good_arr.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-15)
        proportions = good_arr / row_sums

        # 平均組成比率
        mean_proportions = proportions.mean(axis=0)
        n_good = len(good_samples)

        # ベイズ更新: 事前(α_old) + 尤度(n * x̄)
        alpha_pseudo = mean_proportions * n_good
        alpha_new = (1 - learning_rate) * self.alpha + learning_rate * (
            self.alpha + alpha_pseudo
        )

        # 下限クリップ
        alpha_new = np.maximum(alpha_new, min_alpha)

        self.alpha = alpha_new
        self._update_count += 1
        self._alpha_history.append(alpha_new.copy())

        logger.info(
            f"DirichletSampler: α更新 #{self._update_count} "
            f"(good_samples={n_good}, lr={learning_rate})"
        )
        return alpha_new.copy()

    def get_concentration_summary(self) -> dict[str, Any]:
        """現在のα状態のサマリーを返す。

        Returns:
            {
                "alpha": dict[col, float],
                "concentration": float (αの合計 — 大きいほど集中),
                "expected_proportions": dict[col, float],
                "update_count": int,
                "entropy": float (分布のエントロピー),
            }

        Implements: F-DIR04
        """
        alpha_sum = float(self.alpha.sum())
        expected = self.alpha / alpha_sum

        # ディリクレ分布のエントロピー
        from scipy.special import gammaln, digamma

        entropy = (
            sum(gammaln(a) for a in self.alpha)
            - gammaln(alpha_sum)
            + (alpha_sum - self.k) * digamma(alpha_sum)
            - sum((a - 1) * digamma(a) for a in self.alpha)
        )

        return {
            "alpha": dict(zip(self.columns, self.alpha.tolist())),
            "concentration": alpha_sum,
            "expected_proportions": dict(
                zip(self.columns, (expected * self.target_sum).tolist())
            ),
            "update_count": self._update_count,
            "entropy": float(entropy),
        }

    def reset_alpha(self, uniform: bool = True) -> None:
        """αをリセットする。

        Args:
            uniform: True なら均一ディリクレ (α=1) に、
                     False なら初期状態に。

        Implements: F-DIR05
        """
        if uniform:
            self.alpha = np.ones(self.k, dtype=np.float64)
        elif self._alpha_history:
            self.alpha = self._alpha_history[0].copy()
        self._update_count = 0
        self._alpha_history = [self.alpha.copy()]
        logger.info("DirichletSampler: αリセット完了")
"""
Composition sampler module for Dirichlet distribution based sampling.
"""
