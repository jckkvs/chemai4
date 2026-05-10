"""
backend/pipeline/pipeline_grid.py

パイプライングリッド生成モジュール。

各ステップ（imputer / scaler / encoder / feature_gen / feature_sel / estimator）に
複数の選択肢を指定すると、全組み合わせの sklearn Pipeline リストを返す。

Example:
    >>> from backend.pipeline import generate_pipeline_grid, PipelineGridConfig
    >>> grid_config = PipelineGridConfig(
    ...     task="regression",
    ...     numeric_imputers=["mean", "median"],
    ...     numeric_scalers=["standard", "robust"],
    ...     feature_gen_methods=["none", "polynomial"],
    ...     feature_sel_methods=["none", "rfr"],
    ...     estimator_keys=["ridge", "rf", "xgb"],
    ... )
    >>> combinations = generate_pipeline_grid(grid_config)
    >>> print(f"{len(combinations)} パイプライン生成")
    # 2×2×2×2×3 = 48 パイプライン
    >>> for name, config, pipe in combinations:
    ...     print(name)
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from sklearn.pipeline import Pipeline

from backend.pipeline.column_selector import ColumnMeta
from backend.pipeline.col_preprocessor import ColPreprocessConfig
from backend.pipeline.feature_generator import FeatureGenConfig
from backend.pipeline.feature_selector import FeatureSelectorConfig
from backend.pipeline.pipeline_builder import (
    PipelineConfig,
    build_pipeline,
)

logger = logging.getLogger(__name__)


# ============================================================
# 結果型
# ============================================================

class PipelineCombination(NamedTuple):
    """1つのパイプライン組み合わせを表す名前付きタプル。"""
    name: str           # 組み合わせラベル（例: "mean|standard|none|none|ridge"）
    config: PipelineConfig   # 対応するPipelineConfig
    pipeline: Pipeline       # 構築済みsklearn Pipeline（fit前）


# ============================================================
# グリッド設定クラス
# ============================================================

@dataclass
class PipelineGridConfig:
    """
    複数選択肢から全組み合わせを生成するグリッド設定。

    各リストの要素の全デカルト積が Pipeline 候補として展開される。
    リストが空またはNoneの場合はデフォルト1件として扱う。

    Attributes:
        task: "regression" | "classification"

        # Step 1（共通設定 — グリッドではなく全組み合わせで共有）
        col_select_mode: 入力列選択モード
        col_select_columns: 入力/除外列リスト
        col_select_range: 列インデックス範囲
        column_meta: ColumnMeta 辞書（単調性・グループ情報）
        apply_monotonic: monotonic_constraints 自動反映フラグ

        # Step 2: 前処理グリッド
        numeric_imputers: 数値欠損補間の候補リスト
            例: ["mean", "median", "knn"]
        numeric_scalers: 数値スケーラーの候補リスト
            例: ["standard", "robust", "power_yj"]
        cat_low_encoders: 低カーディナリティカテゴリエンコーダーの候補
            例: ["onehot", "ordinal"]
        cat_high_encoders: 高カーディナリティカテゴリエンコーダーの候補
            例: ["ordinal", "target"]

        # Step 3: 特徴量生成グリッド
        feature_gen_methods: 特徴量生成メソッドの候補リスト
            例: ["none", "polynomial", "interaction_only"]
        feature_gen_degrees: 多項式次数の候補リスト（method=polynomial時）
            例: [2, 3]

        # Step 4: 特徴量選択グリッド
        feature_sel_methods: 特徴量選択メソッドの候補リスト
            例: ["none", "lasso", "rfr", "select_kbest"]

        # Step 5: 推定器グリッド
        estimator_keys: モデルキーの候補リスト
            例: ["ridge", "rf", "xgb", "lgbm"]
        estimator_params_list: estimator_keysに対応するパラメータ辞書のリスト
            None の場合は全て {} を使用。長さは estimator_keys と一致させること。

        # 共通前処理設定（グリッド展開しない固定設定）
        base_preprocessor_config: 前処理固定設定（グリッド対象以外のパラメータ）
        base_feature_sel_config: 特徴量選択固定設定（グリッド対象以外）
    """
    task: str = "regression"

    # Step 1 共通
    col_select_mode: str = "all"
    col_select_columns: list[str] | None = None
    col_select_range: tuple[int, int] | None = None
    column_meta: dict[str, ColumnMeta] = field(default_factory=dict)
    apply_monotonic: bool = True

    # Step 2: 前処理グリッド
    numeric_imputers: list[str] = field(default_factory=lambda: ["mean"])
    numeric_scalers: list[str] = field(default_factory=lambda: ["standard"])
    cat_imputers: list[str] = field(default_factory=lambda: ["most_frequent"])
    cat_low_encoders: list[str] = field(default_factory=lambda: ["onehot"])
    cat_high_encoders: list[str] = field(default_factory=lambda: ["ordinal"])
    binary_imputers: list[str] = field(default_factory=lambda: ["most_frequent"])

    # 除外列 (未使用変数、excluderに相当)
    exclude_columns: list[str] = field(default_factory=list)

    # Step 3: 特徴量生成グリッド
    feature_gen_methods: list[str] = field(default_factory=lambda: ["none"])
    feature_gen_degrees: list[int] = field(default_factory=lambda: [2])

    # Step 4: 特徴量選択グリッド
    feature_sel_methods: list[str] = field(default_factory=lambda: ["none"])

    # Step 5: 推定器グリッド
    estimator_keys: list[str] = field(default_factory=lambda: ["rf"])
    estimator_params_list: list[dict[str, Any]] | None = None

    # 固定設定（グリッド展開しない追加パラメータ）
    base_preprocessor_config: ColPreprocessConfig | None = None
    base_feature_sel_config: FeatureSelectorConfig | None = None


# ============================================================
# グリッド生成関数
# ============================================================

def generate_pipeline_grid(
    grid_config: PipelineGridConfig,
    max_combinations: int | None = None,
) -> list[PipelineCombination]:
    """
    PipelineGridConfig の各選択肢の全デカルト積から
    Pipeline 候補リストを生成して返す。

    組み合わせ軸（全て × 全て）:
      numeric_imputer × numeric_scaler × cat_low_encoder × cat_high_encoder
      × feature_gen (method × degree) × feature_sel_method × estimator_key

    Args:
        grid_config: PipelineGridConfig（複数選択肢を持つ設定）
        max_combinations: 最大組み合わせ数（超える場合は先頭 N 件を返す）

    Returns:
        PipelineCombination(name, config, pipeline) のリスト

    Raises:
        ValueError: estimator_keys が空の場合
    """
    gc = grid_config

    if not gc.estimator_keys:
        raise ValueError("estimator_keys は最低1件必要です。")

    # estimator_params_list のデフォルト補完
    ep_list = gc.estimator_params_list
    if ep_list is None:
        ep_list = [{} for _ in gc.estimator_keys]
    elif len(ep_list) != len(gc.estimator_keys):
        logger.warning(
            f"estimator_params_list の長さ({len(ep_list)})が "
            f"estimator_keys({len(gc.estimator_keys)})と一致しません。"
            "不足分は {} で補完します。"
        )
        ep_list = ep_list + [{}] * (len(gc.estimator_keys) - len(ep_list))
        ep_list = ep_list[:len(gc.estimator_keys)]

    estimator_pairs = list(zip(gc.estimator_keys, ep_list))

    # 特徴量生成: method と degree の組み合わせ
    gen_combinations = _build_gen_combinations(gc)

    # 全デカルト積
    product_axes = [
        _ensure_nonempty(gc.numeric_imputers, "mean"),
        _ensure_nonempty(gc.numeric_scalers, "standard"),
        _ensure_nonempty(gc.cat_imputers, "most_frequent"),
        _ensure_nonempty(gc.cat_low_encoders, "onehot"),
        _ensure_nonempty(gc.cat_high_encoders, "ordinal"),
        _ensure_nonempty(gc.binary_imputers, "most_frequent"),
        gen_combinations,
        _ensure_nonempty(gc.feature_sel_methods, "none"),
        estimator_pairs,
    ]

    all_combos = list(itertools.product(*product_axes))
    total = len(all_combos)

    if max_combinations is not None and total > max_combinations:
        logger.warning(
            f"組み合わせ数 {total} が上限 {max_combinations} を超えます。"
            f"先頭 {max_combinations} 件を生成します。"
        )
        all_combos = all_combos[:max_combinations]

    results: list[PipelineCombination] = []

    for combo in all_combos:
        (
            imputer,
            scaler,
            cat_imp,
            cat_low_enc,
            cat_high_enc,
            bin_imp,
            (gen_method, gen_degree),
            sel_method,
            (est_key, est_params),
        ) = combo

        # ラベル生成
        name = _make_label(
            imputer, scaler, cat_imp, cat_low_enc, cat_high_enc,
            bin_imp, gen_method, gen_degree, sel_method, est_key,
        )

        # PipelineConfig 構築
        config = _build_pipeline_config(
            gc=gc,
            imputer=imputer,
            scaler=scaler,
            cat_imp=cat_imp,
            cat_low_enc=cat_low_enc,
            cat_high_enc=cat_high_enc,
            bin_imp=bin_imp,
            gen_method=gen_method,
            gen_degree=gen_degree,
            sel_method=sel_method,
            est_key=est_key,
            est_params=est_params,
        )

        # sklearn Pipeline 構築
        try:
            pipe = build_pipeline(config)
            results.append(PipelineCombination(name=name, config=config, pipeline=pipe))
        except Exception as e:
            logger.warning(f"Pipeline '{name}' の構築に失敗しました（スキップ）: {e}")

    logger.info(
        f"generate_pipeline_grid(): 総組み合わせ={total}, "
        f"生成成功={len(results)}"
    )
    return results


def count_combinations(grid_config: PipelineGridConfig) -> int:
    """
    PipelineGridConfig から生成される組み合わせ数を返す（Pipeline は構築しない）。

    Args:
        grid_config: PipelineGridConfig

    Returns:
        組み合わせ総数
    """
    gc = grid_config
    gen_combos = len(_build_gen_combinations(gc))

    return (
        len(_ensure_nonempty(gc.numeric_imputers, "mean"))
        * len(_ensure_nonempty(gc.numeric_scalers, "standard"))
        * len(_ensure_nonempty(gc.cat_imputers, "most_frequent"))
        * len(_ensure_nonempty(gc.cat_low_encoders, "onehot"))
        * len(_ensure_nonempty(gc.cat_high_encoders, "ordinal"))
        * len(_ensure_nonempty(gc.binary_imputers, "most_frequent"))
        * gen_combos
        * len(_ensure_nonempty(gc.feature_sel_methods, "none"))
        * len(_ensure_nonempty(gc.estimator_keys, "rf"))
    )


# ============================================================
# 内部ヘルパー
# ============================================================

def _build_gen_combinations(
    gc: PipelineGridConfig,
) -> list[tuple[str, int]]:
    """特徴量生成の (method, degree) ペアリストを返す。"""
    methods = _ensure_nonempty(gc.feature_gen_methods, "none")
    degrees = _ensure_nonempty([str(d) for d in gc.feature_gen_degrees], "2")

    combos: list[tuple[str, int]] = []
    for method in methods:
        if method == "none":
            # none ではdegreeは無意味 → 1件だけ
            if ("none", 2) not in combos:
                combos.append(("none", 2))
        else:
            for deg_str in degrees:
                combos.append((method, int(deg_str)))

    return combos if combos else [("none", 2)]


def _ensure_nonempty(lst: list, default: str) -> list:
    """リストが空の場合にデフォルト値1件を返す。"""
    return lst if lst else [default]


def _make_label(
    imputer: str,
    scaler: str,
    cat_imp: str,
    cat_low_enc: str,
    cat_high_enc: str,
    bin_imp: str,
    gen_method: str,
    gen_degree: int,
    sel_method: str,
    est_key: str,
) -> str:
    """短い組み合わせラベルを生成する。"""
    gen_str = gen_method if gen_method == "none" else f"{gen_method}{gen_degree}"
    return (
        f"imp={imputer}"
        f"|scl={scaler}"
        f"|ci={cat_imp}"
        f"|enc={cat_low_enc}"
        f"|bi={bin_imp}"
        f"|gen={gen_str}"
        f"|sel={sel_method}"
        f"|est={est_key}"
    )


def _build_pipeline_config(
    gc: PipelineGridConfig,
    imputer: str,
    scaler: str,
    cat_imp: str,
    cat_low_enc: str,
    cat_high_enc: str,
    bin_imp: str,
    gen_method: str,
    gen_degree: int,
    sel_method: str,
    est_key: str,
    est_params: dict[str, Any],
) -> PipelineConfig:
    """1つの組み合わせに対応する PipelineConfig を構築して返す。"""
    # 前処理設定（ベース設定 + グリッド上書き）
    base_pre = gc.base_preprocessor_config
    pre_config = ColPreprocessConfig(
        override_types=base_pre.override_types if base_pre else {},
        numeric_imputer=imputer,
        numeric_scaler=scaler,
        categorical_imputer=cat_imp,
        cat_low_encoder=cat_low_enc,
        cat_high_encoder=cat_high_enc,
        binary_imputer=bin_imp,
        binary_encoder=base_pre.binary_encoder if base_pre else "ordinal",
        add_missing_indicator=base_pre.add_missing_indicator if base_pre else False,
        cardinality_threshold=base_pre.cardinality_threshold if base_pre else 20,
        onehot_drop=base_pre.onehot_drop if base_pre else "first",
        onehot_handle_unknown=base_pre.onehot_handle_unknown if base_pre else "ignore",
        onehot_max_categories=base_pre.onehot_max_categories if base_pre else None,
        quantile_n_quantiles=base_pre.quantile_n_quantiles if base_pre else 1000,
    )

    # 特徴量生成設定
    gen_config = FeatureGenConfig(
        method=gen_method,
        degree=gen_degree,
    )

    # 特徴量選択設定（ベース設定 + メソッド上書き）
    base_sel = gc.base_feature_sel_config
    sel_config = FeatureSelectorConfig(
        method=sel_method,
        task=gc.task,
        threshold=base_sel.threshold if base_sel else "mean",
        max_features=base_sel.max_features if base_sel else None,
        percentile=base_sel.percentile if base_sel else 50,
        k=base_sel.k if base_sel else 10,
        score_func=base_sel.score_func if base_sel else "f_regression",
        relieff_n_features=base_sel.relieff_n_features if base_sel else 10,
        relieff_n_neighbors=base_sel.relieff_n_neighbors if base_sel else 100,
        boruta_n_estimators=base_sel.boruta_n_estimators if base_sel else 100,
        boruta_max_iter=base_sel.boruta_max_iter if base_sel else 100,
        estimator_key=base_sel.estimator_key if base_sel else None,
    )

    # exclude_columns: col_select_mode="exclude" で対応
    col_select_mode = gc.col_select_mode
    col_select_columns = gc.col_select_columns
    if gc.exclude_columns:
        col_select_mode = "exclude"
        col_select_columns = list(gc.exclude_columns)

    return PipelineConfig(
        task=gc.task,
        col_select_mode=col_select_mode,
        col_select_columns=col_select_columns,
        col_select_range=gc.col_select_range,
        column_meta=gc.column_meta,
        preprocessor_config=pre_config,
        feature_gen_config=gen_config,
        feature_sel_config=sel_config,
        estimator_key=est_key,
        estimator_params=est_params,
        apply_monotonic=gc.apply_monotonic,
    )
