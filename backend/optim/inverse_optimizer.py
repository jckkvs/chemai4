"""
backend/optim/inverse_optimizer.py

逆解析エンジン — 学習済みモデルから最適な説明変数値を逆探索する。

Implements: F-INV01〜INV04
    ランダムサンプリング / グリッドサーチ / ベイズ最適化 / 遺伝的アルゴリズム

設計思想:
    - AutoMLResult.best_pipeline.predict() を目的関数として使う
    - 制約(constraints)で各変数の探索範囲を定義
    - 結果はDataFrameで返す(rank, 各変数値, predicted)

参考文献:
    - ベイズ最適化: Jones et al. (1998) "Efficient Global Optimization"
    - 遺伝的アルゴリズム: Goldberg (1989) "Genetic Algorithms in Search"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── 設定 ────────────────────────────────────────────────
@dataclass
class InverseConfig:
    """逆解析の設定。

    Attributes:
        method: 最適化手法
        target_mode: "range" / "maximize" / "minimize"
        target_min: 目標最小値 (rangeモード)
        target_max: 目標最大値 (rangeモード)
        constraints: {col: {min, max, fixed, fixed_val, active}}
        method_params: 手法固有パラメータ
    """

    method: Literal["random", "grid", "bayesian", "ga", "dirichlet"] = "random"
    target_mode: Literal["range", "maximize", "minimize"] = "range"
    target_min: float | None = None
    target_max: float | None = None
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class InverseResult:
    """逆解析の結果。"""

    candidates: pd.DataFrame
    n_evaluated: int
    best_predicted: float
    method: str
    elapsed_seconds: float = 0.0


# ─── メイン ──────────────────────────────────────────────
def run_inverse_optimization(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    feature_names: list[str],
    config: InverseConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> InverseResult:
    """逆解析を実行する。

    Args:
        predict_fn: X(DataFrame) -> y(array) の予測関数
        feature_names: 説明変数名リスト
        config: 逆解析設定
        progress_callback: (step, total, msg) コールバック

    Returns:
        InverseResult
    """
    import time

    start = time.time()
    cb = progress_callback or (lambda s, t, m: None)

    # 探索変数と固定変数を分離
    search_cols: list[str] = []
    fixed_vals: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}

    for col in feature_names:
        c = config.constraints.get(col, {})
        if not c.get("active", True):
            continue
        if c.get("fixed", False):
            fixed_vals[col] = c.get("fixed_val", 0.0)
        else:
            lo = c.get("min", 0.0)
            hi = c.get("max", 1.0)
            if lo >= hi:
                hi = lo + 1e-6
            bounds[col] = (lo, hi)
            search_cols.append(col)

    if not search_cols:
        raise ValueError("探索対象の変数がありません。固定されていない変数を1つ以上設定してください。")

    n_dim = len(search_cols)
    logger.info(f"逆解析: method={config.method}, dims={n_dim}, fixed={len(fixed_vals)}")

    # 目的関数: X候補→スコア(最大化したい値)
    def objective(X_candidates: np.ndarray) -> np.ndarray:
        """候補行列 (n, n_dim) -> スコア配列 (n,)"""
        df = _build_full_df(X_candidates, search_cols, fixed_vals, feature_names)
        preds = predict_fn(df)
        return _score_predictions(preds, config)

    # 手法別の最適化
    if config.method == "random":
        candidates, n_eval = _optimize_random(
            objective, search_cols, bounds, config.method_params, cb,
        )
    elif config.method == "grid":
        candidates, n_eval = _optimize_grid(
            objective, search_cols, bounds, config.method_params, cb,
        )
    elif config.method == "bayesian":
        candidates, n_eval = _optimize_bayesian(
            objective, search_cols, bounds, config.method_params, cb,
        )
    elif config.method == "ga":
        candidates, n_eval = _optimize_ga(
            objective, search_cols, bounds, config.method_params, cb,
        )
    elif config.method == "dirichlet":
        candidates, n_eval = _optimize_dirichlet(
            objective, search_cols, bounds, config.method_params, cb,
        )
    else:
        raise ValueError(f"未対応の最適化手法: {config.method}")

    # 結果DataFrame構築
    result_df = _build_result_df(
        candidates, search_cols, fixed_vals, feature_names, predict_fn, config,
    )

    elapsed = time.time() - start
    best_pred = result_df["predicted"].iloc[0] if len(result_df) > 0 else 0.0

    return InverseResult(
        candidates=result_df,
        n_evaluated=n_eval,
        best_predicted=float(best_pred),
        method=config.method,
        elapsed_seconds=elapsed,
    )


# ─── ヘルパー ────────────────────────────────────────────
def _build_full_df(
    X_search: np.ndarray,
    search_cols: list[str],
    fixed_vals: dict[str, float],
    feature_names: list[str],
) -> pd.DataFrame:
    """探索変数 + 固定値から完全なDataFrameを構築。"""
    n = X_search.shape[0]
    data: dict[str, np.ndarray] = {}

    search_dict = {col: X_search[:, i] for i, col in enumerate(search_cols)}

    for col in feature_names:
        if col in search_dict:
            data[col] = search_dict[col]
        elif col in fixed_vals:
            data[col] = np.full(n, fixed_vals[col])
        else:
            data[col] = np.zeros(n)

    return pd.DataFrame(data, columns=feature_names)


def _score_predictions(preds: np.ndarray, config: InverseConfig) -> np.ndarray:
    """予測値をスコアに変換（大きいほど良い）。"""
    preds = np.asarray(preds, dtype=np.float64).ravel()

    if config.target_mode == "maximize":
        return preds
    elif config.target_mode == "minimize":
        return -preds
    else:  # range
        lo = config.target_min if config.target_min is not None else -np.inf
        hi = config.target_max if config.target_max is not None else np.inf
        center = (lo + hi) / 2.0
        half_width = max((hi - lo) / 2.0, 1e-9)
        # 範囲の中心に近いほどスコアが高い（ガウシアン型）
        return np.exp(-0.5 * ((preds - center) / half_width) ** 2)


def _build_result_df(
    candidates: np.ndarray,
    search_cols: list[str],
    fixed_vals: dict[str, float],
    feature_names: list[str],
    predict_fn: Callable,
    config: InverseConfig,
    top_n: int = 20,
) -> pd.DataFrame:
    """候補行列から上位N件の結果DataFrameを構築。"""
    df_full = _build_full_df(candidates, search_cols, fixed_vals, feature_names)
    preds = predict_fn(df_full)
    preds = np.asarray(preds).ravel()
    scores = _score_predictions(preds, config)

    # スコア降順でソート
    order = np.argsort(scores)[::-1][:top_n]

    result_rows = []
    for rank, idx in enumerate(order, 1):
        row: dict[str, Any] = {"rank": rank}
        for col in search_cols:
            col_idx = search_cols.index(col)
            row[col] = round(float(candidates[idx, col_idx]), 6)
        row["predicted"] = round(float(preds[idx]), 6)
        row["score"] = round(float(scores[idx]), 6)
        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ═══════════════════════════════════════════════════════════
# 手法1: ランダムサンプリング
# ═══════════════════════════════════════════════════════════
def _optimize_random(
    objective: Callable,
    search_cols: list[str],
    bounds: dict[str, tuple[float, float]],
    params: dict,
    cb: Callable,
) -> tuple[np.ndarray, int]:
    """制約範囲内から一様ランダムにサンプリング。"""
    n_samples = params.get("n_samples", 1000)
    seed = params.get("seed", 42)
    rng = np.random.RandomState(seed)

    n_dim = len(search_cols)
    cb(1, 3, f"ランダムサンプリング: {n_samples}点生成中...")

    X = np.empty((n_samples, n_dim))
    for i, col in enumerate(search_cols):
        lo, hi = bounds[col]
        X[:, i] = rng.uniform(lo, hi, n_samples)

    cb(2, 3, "予測値を計算中...")
    _ = objective(X)  # スコア計算（内部でpredict）

    cb(3, 3, "完了")
    return X, n_samples


# ═══════════════════════════════════════════════════════════
# 手法2: グリッドサーチ
# ═══════════════════════════════════════════════════════════
def _optimize_grid(
    objective: Callable,
    search_cols: list[str],
    bounds: dict[str, tuple[float, float]],
    params: dict,
    cb: Callable,
) -> tuple[np.ndarray, int]:
    """各変数を等間隔に分割し全組み合わせを評価。"""
    n_points = params.get("n_points", 10)
    n_dim = len(search_cols)

    # 次元が多すぎる場合は制限
    max_total = 500_000
    actual_points = n_points
    if n_points ** n_dim > max_total:
        actual_points = max(2, int(max_total ** (1.0 / n_dim)))
        logger.warning(
            f"グリッド総数が{max_total}を超えるため、分割数を{actual_points}に縮小"
        )

    cb(1, 3, f"グリッド生成: {actual_points}^{n_dim}点...")

    # meshgrid構築
    grids = []
    for col in search_cols:
        lo, hi = bounds[col]
        grids.append(np.linspace(lo, hi, actual_points))

    mesh = np.meshgrid(*grids, indexing="ij")
    X = np.column_stack([m.ravel() for m in mesh])
    n_total = X.shape[0]

    cb(2, 3, f"予測値を計算中... ({n_total}点)")
    _ = objective(X)

    cb(3, 3, "完了")
    return X, n_total


# ═══════════════════════════════════════════════════════════
# 手法3: ベイズ最適化
# ═══════════════════════════════════════════════════════════
def _optimize_bayesian(
    objective: Callable,
    search_cols: list[str],
    bounds: dict[str, tuple[float, float]],
    params: dict,
    cb: Callable,
) -> tuple[np.ndarray, int]:
    """ガウス過程回帰に基づくベイズ最適化。

    参考: Jones et al. (1998) "Efficient Global Optimization"
    """
    n_trials = params.get("n_trials", 100)
    seed = params.get("seed", 42)
    acq_func = params.get("acq_func", "EI").lower()
    n_init = max(5, min(20, n_trials // 5))
    n_dim = len(search_cols)

    rng = np.random.RandomState(seed)

    # Phase 1: 初期サンプリング (Latin Hypercube風)
    cb(1, 4, f"初期サンプリング: {n_init}点...")
    X_init = np.empty((n_init, n_dim))
    for i, col in enumerate(search_cols):
        lo, hi = bounds[col]
        X_init[:, i] = rng.uniform(lo, hi, n_init)

    scores_init = objective(X_init)

    # Phase 2: GP + 獲得関数で逐次探索
    X_all = X_init.copy()
    scores_all = scores_init.copy()
    n_bo_iter = n_trials - n_init

    cb(2, 4, f"ベイズ最適化: {n_bo_iter}回のイテレーション...")

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from scipy.stats import norm

        scaler_mean = X_all.mean(axis=0)
        scaler_std = X_all.std(axis=0) + 1e-9

        for iteration in range(n_bo_iter):
            if iteration % 10 == 0:
                cb(2, 4, f"ベイズ最適化: {iteration}/{n_bo_iter}...")

            # GPフィッティング
            X_scaled = (X_all - scaler_mean) / scaler_std
            y_scaled = (scores_all - scores_all.mean()) / (scores_all.std() + 1e-9)

            kernel = Matern(nu=2.5) + WhiteKernel()
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2, normalize_y=False, alpha=1e-6,
            )
            try:
                gp.fit(X_scaled, y_scaled)
            except Exception:
                # GPフィッティング失敗時はランダム候補で代替
                x_new = np.empty(n_dim)
                for j, col in enumerate(search_cols):
                    lo, hi = bounds[col]
                    x_new[j] = rng.uniform(lo, hi)
                X_all = np.vstack([X_all, x_new])
                new_score = objective(x_new.reshape(1, -1))
                scores_all = np.append(scores_all, new_score)
                continue

            # ランダム候補を生成して獲得関数で選択
            n_cand = min(1000, max(100, n_trials * 5))
            X_cand = np.empty((n_cand, n_dim))
            for j, col in enumerate(search_cols):
                lo, hi = bounds[col]
                X_cand[:, j] = rng.uniform(lo, hi, n_cand)

            X_cand_scaled = (X_cand - scaler_mean) / scaler_std
            mu, sigma = gp.predict(X_cand_scaled, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            y_best = y_scaled.max()

            # 獲得関数EI
            if acq_func == "ucb":
                kappa = 2.0
                acq_vals = mu + kappa * sigma
            elif acq_func == "pi":
                Z = (mu - y_best - 0.01) / sigma
                acq_vals = norm.cdf(Z)
            else:  # EI
                improvement = mu - y_best - 0.01
                Z = improvement / sigma
                acq_vals = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                acq_vals[sigma < 1e-9] = 0.0

            best_idx = np.argmax(acq_vals)
            x_new = X_cand[best_idx]

            X_all = np.vstack([X_all, x_new])
            new_score = objective(x_new.reshape(1, -1))
            scores_all = np.append(scores_all, new_score)

            # スケーラー更新
            scaler_mean = X_all.mean(axis=0)
            scaler_std = X_all.std(axis=0) + 1e-9

    except ImportError as e:
        logger.warning(f"ベイズ最適化の依存パッケージ不足: {e}。ランダム探索にフォールバック。")
        X_extra = np.empty((n_bo_iter, n_dim))
        for j, col in enumerate(search_cols):
            lo, hi = bounds[col]
            X_extra[:, j] = rng.uniform(lo, hi, n_bo_iter)
        X_all = np.vstack([X_all, X_extra])
        _ = objective(X_extra)

    cb(3, 4, "最終評価中...")
    cb(4, 4, "完了")
    return X_all, len(X_all)


# ═══════════════════════════════════════════════════════════
# 手法4: 遺伝的アルゴリズム
# ═══════════════════════════════════════════════════════════
def _optimize_ga(
    objective: Callable,
    search_cols: list[str],
    bounds: dict[str, tuple[float, float]],
    params: dict,
    cb: Callable,
) -> tuple[np.ndarray, int]:
    """実数値遺伝的アルゴリズム。

    参考: Goldberg (1989), Deb et al. (2002) "SBX crossover"
    """
    pop_size = params.get("pop_size", 50)
    n_gen = params.get("n_generations", 100)
    mut_rate = params.get("mutation_rate", 0.1)
    cx_rate = params.get("crossover_rate", 0.8)
    seed = params.get("seed", 42)
    n_dim = len(search_cols)

    rng = np.random.RandomState(seed)

    # 下限・上限配列
    lo_arr = np.array([bounds[col][0] for col in search_cols])
    hi_arr = np.array([bounds[col][1] for col in search_cols])
    range_arr = hi_arr - lo_arr

    # 初期集団
    cb(1, 3, f"初期集団生成: {pop_size}個体...")
    pop = lo_arr + rng.rand(pop_size, n_dim) * range_arr
    fitness = objective(pop)
    n_eval = pop_size
    all_candidates = pop.copy()

    # 進化ループ
    for gen in range(n_gen):
        if gen % 10 == 0:
            cb(2, 3, f"世代 {gen}/{n_gen} (最良: {fitness.max():.4f})")

        # トーナメント選択
        parents = np.empty_like(pop)
        for i in range(pop_size):
            a, b = rng.randint(0, pop_size, 2)
            parents[i] = pop[a] if fitness[a] > fitness[b] else pop[b]

        # SBX交叉
        offspring = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.rand() < cx_rate:
                offspring[i], offspring[i + 1] = _sbx_crossover(
                    parents[i], parents[i + 1], lo_arr, hi_arr, rng,
                )

        # 多項式突然変異
        for i in range(pop_size):
            for j in range(n_dim):
                if rng.rand() < mut_rate:
                    delta = range_arr[j] * 0.1
                    offspring[i, j] += rng.normal(0, delta)
                    offspring[i, j] = np.clip(offspring[i, j], lo_arr[j], hi_arr[j])

        # 評価
        off_fitness = objective(offspring)
        n_eval += pop_size

        # エリート保存: 親+子から上位pop_size個を選択
        combined = np.vstack([pop, offspring])
        combined_fit = np.concatenate([fitness, off_fitness])
        elite_idx = np.argsort(combined_fit)[::-1][:pop_size]
        pop = combined[elite_idx]
        fitness = combined_fit[elite_idx]

        all_candidates = np.vstack([all_candidates, offspring])

    cb(3, 3, "完了")
    return all_candidates, n_eval


def _sbx_crossover(
    p1: np.ndarray,
    p2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    rng: np.random.RandomState,
    eta: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)。

    参考: Deb & Agrawal (1995)
    """
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if abs(p1[i] - p2[i]) < 1e-14:
            continue
        u = rng.rand()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        c1[i] = np.clip(c1[i], lo[i], hi[i])
        c2[i] = np.clip(c2[i], lo[i], hi[i])
    return c1, c2


# ═══════════════════════════════════════════════════════════
# 手法5: ディリクレ分布最適化（組成系 — 合計=1 制約）
# ═══════════════════════════════════════════════════════════
def _optimize_dirichlet(
    objective: Callable,
    search_cols: list[str],
    bounds: dict[str, tuple[float, float]],
    params: dict,
    cb: Callable,
) -> tuple[np.ndarray, int]:
    """ディリクレ分布による組成系最適化。

    組成(合計=1)制約のあるデータに特化した手法。
    各イテレーションで上位候補の組成比からαパラメータを更新し、
    サンプリング分布を有望領域に収束させる。

    アルゴリズム:
        1. 初期α = 均一 (全要素 1.0) でディリクレサンプリング
        2. 候補を評価し、上位 top_k 件を選出
        3. 上位候補の平均組成比 × concentration でαを更新
        4. 更新αでディリクレサンプリング → 2 へ
        5. n_rounds 回繰り返し

    参考:
        - Ferguson (1973) "A Bayesian Analysis of Some Nonparametric Problems"
          原文: "The Dirichlet process is a measure on the space of
                 probability distributions"
          訳: ディリクレ過程は確率分布の空間上の測度である
        - Minka (2000) "Estimating a Dirichlet distribution"
          原文: "Given a set of observed count vectors, the distribution
                 of proportions is modeled by a Dirichlet"
          訳: 観測されたカウントベクトルの集合から、
              比率の分布をディリクレ分布でモデル化する

    method_params:
        n_samples_per_round: 各ラウンドのサンプル数 (default: 500)
        n_rounds: α更新回数 (default: 20)
        top_k: α更新に使う上位候補数 (default: 50)
        concentration: α集中度 (default: 10.0; 大きいほど収束が速い)
        alpha_init: 初期α値 (default: None → 均一)
        seed: 乱数シード (default: 42)
        total_sum: 合計値 (default: 1.0; 例えばwt%なら100.0)
        respect_bounds: boundsを尊重してリジェクション (default: True)
    """
    n_per_round = params.get("n_samples_per_round", 500)
    n_rounds = params.get("n_rounds", 20)
    top_k = params.get("top_k", 50)
    concentration = params.get("concentration", 10.0)
    seed = params.get("seed", 42)
    total_sum = params.get("total_sum", 1.0)
    respect_bounds = params.get("respect_bounds", True)
    n_dim = len(search_cols)

    rng = np.random.RandomState(seed)

    # 初期α: 均一、またはユーザー指定
    alpha_init = params.get("alpha_init", None)
    if alpha_init is not None:
        alpha = np.asarray(alpha_init, dtype=np.float64)
        if len(alpha) != n_dim:
            raise ValueError(
                f"alpha_init の長さ({len(alpha)})が探索変数数({n_dim})と不一致"
            )
    else:
        alpha = np.ones(n_dim, dtype=np.float64)

    # bounds配列
    lo_arr = np.array([bounds[col][0] for col in search_cols])
    hi_arr = np.array([bounds[col][1] for col in search_cols])

    all_candidates: list[np.ndarray] = []
    n_eval = 0

    cb(1, n_rounds + 1, f"ディリクレ最適化: α初期化 (dim={n_dim})")

    for rnd in range(n_rounds):
        cb(rnd + 1, n_rounds + 1,
           f"ラウンド {rnd+1}/{n_rounds} (α_max={alpha.max():.2f})")

        # ── ディリクレサンプリング ──
        # リジェクションサンプリング付き（bounds制約）
        valid_samples: list[np.ndarray] = []
        max_attempts = n_per_round * 10  # 最大試行数
        attempts = 0

        while len(valid_samples) < n_per_round and attempts < max_attempts:
            batch_size = min(n_per_round * 3, max_attempts - attempts)
            raw = rng.dirichlet(alpha, size=batch_size)
            raw *= total_sum  # 合計をtotal_sumに調整

            if respect_bounds:
                # bounds制約でフィルタ
                mask = np.ones(batch_size, dtype=bool)
                for j in range(n_dim):
                    mask &= (raw[:, j] >= lo_arr[j]) & (raw[:, j] <= hi_arr[j])
                valid = raw[mask]
            else:
                valid = raw

            for row in valid:
                if len(valid_samples) >= n_per_round:
                    break
                valid_samples.append(row)
            attempts += batch_size

        # bounds内に収まるサンプルが不足した場合、クリッピングで補充
        if len(valid_samples) < n_per_round:
            deficit = n_per_round - len(valid_samples)
            logger.warning(
                f"[Dirichlet] bounds内サンプル不足 ({len(valid_samples)}/{n_per_round})、"
                f"{deficit}件をクリッピングで補充"
            )
            raw_extra = rng.dirichlet(alpha, size=deficit) * total_sum
            for row in raw_extra:
                clipped = np.clip(row, lo_arr, hi_arr)
                # クリッピング後も合計をtotal_sumに正規化
                s = clipped.sum()
                if s > 0:
                    clipped = clipped / s * total_sum
                valid_samples.append(clipped)

        X_round = np.array(valid_samples)

        # ── 評価 ──
        scores_round = objective(X_round)
        n_eval += len(X_round)
        all_candidates.append(X_round)

        # ── α更新: 上位候補の平均組成比を反映 ──
        actual_top_k = min(top_k, len(X_round))
        top_indices = np.argsort(scores_round)[::-1][:actual_top_k]
        top_samples = X_round[top_indices]

        # 上位候補の正規化組成比を計算
        top_normalized = top_samples / (top_samples.sum(axis=1, keepdims=True) + 1e-15)
        mean_composition = top_normalized.mean(axis=0)

        # αを更新: 平均組成比 × concentration
        # 下限0.1を設定して完全に0にならないようにする（探索空間の縮小を防ぐ）
        alpha = np.maximum(mean_composition * concentration, 0.1)

        logger.debug(
            f"[Dirichlet] Round {rnd+1}: "
            f"best_score={scores_round.max():.6f}, "
            f"α_range=[{alpha.min():.3f}, {alpha.max():.3f}]"
        )

    cb(n_rounds + 1, n_rounds + 1, "完了")

    # 全ラウンドの候補を結合
    X_all = np.vstack(all_candidates)
    return X_all, n_eval
"""Complexity: 9, Description: 5手法(ランダム/グリッド/ベイズ/GA/ディリクレ)の逆解析エンジン。GP+EI/PI/UCBの本格的ベイズ最適化、SBX交叉付きGA、ディリクレ分布α更新型組成最適化、range/maximize/minimizeの3目標モードをサポート。"""
