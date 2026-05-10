"""
backend/pipeline/feature_selector.py

特徴量選択モジュール。
回帰/分類タスクに応じてモデルを自動切り替えし、複数の選択手法に対応する。

対応メソッド:
  none             : スキップ（パススルー）
  lasso            : SelectFromModel(LassoCV)
  ridge            : SelectFromModel(RidgeCV)
  rfr / rfc        : SelectFromModel(RandomForest)
  xgb              : SelectFromModel(XGBoost)
  select_from_model: SelectFromModel(任意モデル)
  select_percentile: SelectPercentile
  select_kbest     : SelectKBest
  relieff          : ReliefF (skrebate)  ← オプション
  boruta           : BorutaPy (boruta)   ← オプション
  genetic          : GeneticSelectionCV  ← オプション
  group_lasso      : GroupLasso (group_lasso) ← オプション
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectPercentile,
    SelectKBest,
    f_regression,
    f_classif,
    mutual_info_regression,
    mutual_info_classif,
)
from sklearn.linear_model import LassoCV, RidgeCV

from backend.utils.config import RANDOM_STATE
from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

# オプショナルライブラリ
_skrebate = safe_import("skrebate", "relieff")
_boruta = safe_import("boruta", "boruta")
_genetic = safe_import("sklearn_genetic", "sklearn-genetic-opt")
_group_lasso = safe_import("group_lasso", "group-lasso")
_xgb = safe_import("xgboost", "xgboost")


# ============================================================
# 設定クラス
# ============================================================

@dataclass
class FeatureSelectorConfig:
    """
    特徴量選択の設定。

    Attributes:
        method: 選択手法。
            "none" | "lasso" | "ridge" | "rfr" | "xgb" |
            "select_from_model" | "select_percentile" | "select_kbest" |
            "relieff" | "boruta" | "genetic" | "group_lasso"
        task: "regression" | "classification"
        threshold: SelectFromModel の閾値。"mean" | "median" | float
        max_features: SelectFromModel の最大選択特徴量数
        percentile: SelectPercentile のパーセンタイル（0-100）
        k: SelectKBest の選択数
        score_func: SelectPercentile/KBest のスコア関数名
            "f_regression" | "f_classif" |
            "mutual_info_regression" | "mutual_info_classif"
        relieff_n_features: ReliefF の選択特徴量数
        relieff_n_neighbors: ReliefF の近傍数
        boruta_n_estimators: Boruta の RF 推定器数
        boruta_max_iter: Boruta の最大イテレーション
        genetic_n_generations: GeneticSelection の世代数
        genetic_n_population: GeneticSelection の集団サイズ
        group_lasso_alpha: GroupLasso の正則化係数
        group_lasso_groups: グループ配列（None で ColumnMeta から自動生成）
        estimator_key: select_from_model 時のモデルキー（factory.py キー）
        estimator_params: estimator_key に渡すパラメータ
    """
    method: str = "none"
    task: str = "regression"

    # SelectFromModel 共通
    threshold: str | float = "mean"
    max_features: int | None = None

    # SelectPercentile / SelectKBest
    percentile: int = 50
    k: int | str = 10
    score_func: str = "f_regression"

    # ReliefF
    relieff_n_features: int = 10
    relieff_n_neighbors: int = 100

    # Boruta
    boruta_n_estimators: int = 100
    boruta_max_iter: int = 100

    # Genetic
    genetic_n_generations: int = 40
    genetic_n_population: int = 300
    genetic_cv: int = 5
    genetic_scoring: str | None = None  # None → タスクにより自動設定

    # GroupLasso
    group_lasso_alpha: float = 0.05
    group_lasso_groups: list[list[int]] | None = None

    # カスタムモデル（select_from_model 時）
    estimator_key: str | None = None
    estimator_params: dict[str, Any] = field(default_factory=dict)


# ============================================================
# メインクラス
# ============================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    特徴量選択 Transformer。

    FeatureSelectorConfig に従って特徴量を選択する。
    method="none" の場合はパススルー。

    分類/回帰でモデルを自動切り替えする。
    オプションライブラリ未インストール時は SelectFromModel(RF) にフォールバックする。

    Args:
        config: FeatureSelectorConfig（省略時は method="none"）
        column_meta: ColumnMeta辞書（GroupLasso のグループ配列自動生成に使用）
    """

    def __init__(
        self,
        config: FeatureSelectorConfig | None = None,
        column_meta: dict | None = None,
    ) -> None:
        self.config = config or FeatureSelectorConfig()
        self.column_meta = column_meta or {}  # dict[str, ColumnMeta]
        self._selector: Any = None
        self._feature_names_in: list[str] = []
        self._support_mask: np.ndarray | None = None
        self._fixed_indices: list[int] = []  # fixed=True の列インデックス


    # ----------------------------------------------------------
    # fit
    # ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
    ) -> "FeatureSelector":
        """
        特徴量選択手法を fit する。

        Args:
            X: 入力特徴量行列
            y: 目的変数（モデルベース手法で必要）

        Returns:
            self
        """
        cfg = self.config

        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            X_arr = X.values
        else:
            X_arr = np.asarray(X)
            self._feature_names_in = [f"x{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y) if y is not None else None

        # fixed 変数のインデックスを記録（選択手法に関わらず常に保持）
        self._fixed_indices: list[int] = []
        if self.column_meta:
            for idx, col in enumerate(self._feature_names_in):
                meta = self.column_meta.get(col)
                if meta is not None:
                    is_fixed = meta.fixed if hasattr(meta, "fixed") else meta.get("fixed", False)
                    if is_fixed:
                        self._fixed_indices.append(idx)

        if cfg.method == "none":
            logger.debug("FeatureSelector: method=none → パススルー")
            return self

        # 特徴量評価（特にLasso等の係数ベース手法）におけるスケール依存性を排除するため、
        # fit 用に入力行列全体を標準化する。これにより、値のスケールが異なる変数間でも
        # 公平に重要度が評価される。transform には影響を与えない。
        from sklearn.preprocessing import StandardScaler
        X_fit = StandardScaler().fit_transform(X_arr)

        selector = self._build_selector(cfg, X_fit, y_arr)

        if selector is None:
            # フォールバック: SelectFromModel(RF)
            logger.warning(
                f"FeatureSelector: method={cfg.method} が利用不可 → "
                "SelectFromModel(RandomForest) にフォールバック"
            )
            selector = self._build_sfm_rf(cfg)

        selector.fit(X_fit, y_arr)
        self._selector = selector

        # サポートマスクを取得
        try:
            self._support_mask = selector.get_support()
        except AttributeError:
            self._support_mask = np.ones(X_arr.shape[1], dtype=bool)

        n_selected = int(np.sum(self._support_mask))
        logger.info(
            f"FeatureSelector.fit(): method={cfg.method}, "
            f"{X_arr.shape[1]} → {n_selected} 特徴量選択"
        )
        return self

    # ----------------------------------------------------------
    # transform
    # ----------------------------------------------------------

    def transform(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any = None,
    ) -> np.ndarray:
        """
        選択された特徴量のみを返す。
        fixed=True の列は選択手法に関わらず常に含む。

        Args:
            X: 入力特徴量行列

        Returns:
            選択済み ndarray
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if self._selector is None or self.config.method == "none":
            return X_arr

        selected = self._selector.transform(X_arr)

        # fixed 列を追加（既に選択済みならどこに含まれるか高速path）
        if self._fixed_indices:
            selected_mask = self._support_mask if self._support_mask is not None else np.ones(X_arr.shape[1], dtype=bool)
            # fixed 列でまだ selected に含まれていないものを追加
            missing_idxs = [i for i in self._fixed_indices if not selected_mask[i]]
            if missing_idxs:
                extra_cols = X_arr[:, missing_idxs]
                selected = np.hstack([selected, extra_cols])
                logger.debug(f"fixed 列を強制追加: {[self._feature_names_in[i] for i in missing_idxs]}")

        return selected

    # ----------------------------------------------------------
    # ユーティリティ
    # ----------------------------------------------------------

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """選択された特徴量名を返す（fixed列は常に含む）。"""
        names = input_features or self._feature_names_in

        if self._support_mask is None or self.config.method == "none":
            return np.array(names)

        selected_names = [n for n, s in zip(names, self._support_mask) if s]

        # fixed 列でまだ選ばれていないものを末尾に追加
        if self._fixed_indices:
            selected_set = set(selected_names)
            for i in self._fixed_indices:
                if i < len(names):
                    n = names[i] if hasattr(names, "__getitem__") else list(names)[i]
                    if n not in selected_set:
                        selected_names.append(n)
                        selected_set.add(n)

        return np.array(selected_names)

    @property
    def support_mask(self) -> np.ndarray | None:
        """選択マスク（True=選択）を返す。"""
        return self._support_mask

    # ----------------------------------------------------------
    # セレクタ構築ディスパッチ
    # ----------------------------------------------------------

    def _build_selector(
        self,
        cfg: FeatureSelectorConfig,
        X: np.ndarray,
        y: np.ndarray | None,
    ) -> Any:
        """メソッド名に応じてセレクタを返す。利用不可の場合は None。"""
        method = cfg.method

        if method == "lasso":
            return self._build_sfm_lasso(cfg)

        elif method == "ridge":
            return self._build_sfm_ridge(cfg)

        elif method in ("rfr", "rfc"):
            return self._build_sfm_rf(cfg)

        elif method == "xgb":
            return self._build_sfm_xgb(cfg)

        elif method == "select_from_model":
            return self._build_sfm_custom(cfg)

        elif method == "select_percentile":
            return self._build_select_percentile(cfg)

        elif method == "select_kbest":
            return self._build_select_kbest(cfg)

        elif method == "relieff":
            return self._build_relieff(cfg)

        elif method == "boruta":
            return self._build_boruta(cfg)

        elif method == "genetic":
            return self._build_genetic(cfg, X, y)

        elif method == "group_lasso":
            return self._build_group_lasso(cfg)

        else:
            logger.warning(f"未知の特徴量選択メソッド '{method}' → none（パススルー）を使用")
            return None

    # ----------------------------------------------------------
    # SelectFromModel ファミリー
    # ----------------------------------------------------------

    def _build_sfm_lasso(self, cfg: FeatureSelectorConfig) -> SelectFromModel:
        estimator = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        return SelectFromModel(
            estimator, threshold=cfg.threshold, max_features=cfg.max_features
        )

    def _build_sfm_ridge(self, cfg: FeatureSelectorConfig) -> SelectFromModel:
        estimator = RidgeCV(cv=5)
        return SelectFromModel(
            estimator, threshold=cfg.threshold, max_features=cfg.max_features
        )

    def _build_sfm_rf(self, cfg: FeatureSelectorConfig) -> SelectFromModel:
        if cfg.task == "classification":
            estimator: Any = RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )
        return SelectFromModel(
            estimator, threshold=cfg.threshold, max_features=cfg.max_features
        )

    def _build_sfm_xgb(self, cfg: FeatureSelectorConfig) -> SelectFromModel | None:
        if not _xgb:
            logger.warning("xgboost 未インストール → RandomForest で代替")
            return self._build_sfm_rf(cfg)
        from xgboost import XGBRegressor, XGBClassifier  # type: ignore
        if cfg.task == "classification":
            estimator: Any = XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss"
            )
        else:
            estimator = XGBRegressor(
                n_estimators=100, random_state=RANDOM_STATE, eval_metric="rmse"
            )
        return SelectFromModel(
            estimator, threshold=cfg.threshold, max_features=cfg.max_features
        )

    def _build_sfm_custom(self, cfg: FeatureSelectorConfig) -> SelectFromModel:
        """estimator_key で factory.py からモデルを取得して SelectFromModel に wrap。"""
        if cfg.estimator_key:
            try:
                from backend.models.factory import get_model
                estimator = get_model(
                    cfg.estimator_key, task=cfg.task, **cfg.estimator_params
                )
            except Exception as e:
                logger.warning(f"estimator_key='{cfg.estimator_key}' の取得失敗: {e} → RF で代替")
                estimator = self._get_default_rf(cfg)
        else:
            estimator = self._get_default_rf(cfg)

        return SelectFromModel(
            estimator, threshold=cfg.threshold, max_features=cfg.max_features
        )

    # ----------------------------------------------------------
    # SelectPercentile / SelectKBest
    # ----------------------------------------------------------

    def _build_select_percentile(self, cfg: FeatureSelectorConfig) -> SelectPercentile:
        score_fn = self._resolve_score_func(cfg)
        return SelectPercentile(score_func=score_fn, percentile=cfg.percentile)

    def _build_select_kbest(self, cfg: FeatureSelectorConfig) -> SelectKBest:
        score_fn = self._resolve_score_func(cfg)
        return SelectKBest(score_func=score_fn, k=cfg.k)

    def _resolve_score_func(self, cfg: FeatureSelectorConfig) -> Any:
        """score_func 名から関数オブジェクトを返す。タスクに応じてデフォルトも補正する。"""
        func_map: dict[str, Any] = {
            "f_regression": f_regression,
            "f_classif": f_classif,
            "mutual_info_regression": mutual_info_regression,
            "mutual_info_classif": mutual_info_classif,
        }
        name = cfg.score_func

        # タスクに応じたデフォルト補正
        if name == "f_regression" and cfg.task == "classification":
            logger.info("task=classification なので score_func を f_classif に自動変更")
            name = "f_classif"
        elif name == "mutual_info_regression" and cfg.task == "classification":
            logger.info("task=classification なので score_func を mutual_info_classif に自動変更")
            name = "mutual_info_classif"

        fn = func_map.get(name)
        if fn is None:
            logger.warning(f"未知の score_func '{name}' → f_regression/f_classif を使用")
            fn = f_classif if cfg.task == "classification" else f_regression

        return fn

    # ----------------------------------------------------------
    # ReliefF
    # ----------------------------------------------------------

    def _build_relieff(self, cfg: FeatureSelectorConfig) -> Any:
        if not _skrebate:
            logger.warning("skrebate 未インストール → ReliefF は利用不可")
            return None
        from skrebate import ReliefF  # type: ignore
        return ReliefF(
            n_features_to_select=cfg.relieff_n_features,
            n_neighbors=cfg.relieff_n_neighbors,
        )

    # ----------------------------------------------------------
    # Boruta
    # ----------------------------------------------------------

    def _build_boruta(self, cfg: FeatureSelectorConfig) -> Any:
        if not _boruta:
            logger.warning("boruta 未インストール → Boruta は利用不可")
            return None
        from boruta import BorutaPy  # type: ignore
        if cfg.task == "classification":
            base_rf: Any = RandomForestClassifier(
                n_estimators=cfg.boruta_n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            base_rf = RandomForestRegressor(
                n_estimators=cfg.boruta_n_estimators,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        return BorutaPy(
            estimator=base_rf,
            n_estimators="auto",
            max_iter=cfg.boruta_max_iter,
            random_state=RANDOM_STATE,
            verbose=0,
        )

    # ----------------------------------------------------------
    # Genetic Selection
    # ----------------------------------------------------------

    def _build_genetic(
        self,
        cfg: FeatureSelectorConfig,
        X: np.ndarray,
        y: np.ndarray | None,
    ) -> Any:
        if not _genetic:
            logger.warning("sklearn_genetic 未インストール → GeneticSelection は利用不可")
            return None
        try:
            from sklearn_genetic import GAFeatureSelectionCV  # type: ignore
            scoring = cfg.genetic_scoring or (
                "accuracy" if cfg.task == "classification" else "r2"
            )
            estimator = self._get_default_rf(cfg)
            return GAFeatureSelectionCV(
                estimator=estimator,
                cv=cfg.genetic_cv,
                scoring=scoring,
                population_size=cfg.genetic_n_population,
                generations=cfg.genetic_n_generations,
                n_jobs=-1,
            )
        except Exception as e:
            logger.warning(f"GeneticSelection の構築失敗: {e} → 利用不可")
            return None

    # ----------------------------------------------------------
    # GroupLasso
    # ----------------------------------------------------------

    def _build_group_lasso(self, cfg: FeatureSelectorConfig) -> Any:
        """
        GroupLasso を構築する。
        group_lasso_groups が None の場合は ColumnMeta.group から自動生成する。
        """
        if not _group_lasso:
            logger.warning("group_lasso 未インストール → GroupLasso は利用不可")
            return None
        try:

            groups = cfg.group_lasso_groups
            if groups is None:
                groups = self._build_groups_from_meta()

            if not groups:
                logger.warning("GroupLasso: グループ情報が空 → RandomForest で代替")
                return None

            return _GroupLassoSelector(
                alpha=cfg.group_lasso_alpha,
                groups=groups,
            )
        except Exception as e:
            logger.warning(f"GroupLasso の構築失敗: {e} → 利用不可")
            return None

    def _build_groups_from_meta(self) -> list[list[int]] | None:
        """ColumnMeta.group からグループ配列（インデックスリストのリスト）を構築する。"""
        if not self.column_meta or not self._feature_names_in:
            return None

        group_map: dict[str, list[int]] = {}
        for idx, col in enumerate(self._feature_names_in):
            meta = self.column_meta.get(col)
            if meta is not None and meta.group is not None:
                group_map.setdefault(meta.group, []).append(idx)

        if not group_map:
            return None

        return list(group_map.values())

    # ----------------------------------------------------------
    # 共通ヘルパー
    # ----------------------------------------------------------

    def _get_default_rf(self, cfg: FeatureSelectorConfig) -> Any:
        """タスクに応じたデフォルト RF を返す。"""
        if cfg.task == "classification":
            return RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            )
        return RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )


# ============================================================
# GroupLasso ラッパー（sklearn インターフェース適合）
# ============================================================

class _GroupLassoSelector(BaseEstimator, TransformerMixin):
    """
    group_lasso.GroupLasso を sklearn TransformerMixin に適合させるラッパー。
    fit 後に係数がゼロでない特徴量のみを選択する。
    """

    def __init__(self, alpha: float = 0.05, groups: list[list[int]] | None = None) -> None:
        self.alpha = alpha
        self.groups = groups
        self._model: Any = None
        self._support: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_GroupLassoSelector":
        from group_lasso import GroupLasso  # type: ignore

        # group_lasso が期待する形式: 各特徴量のグループID 配列
        n_features = X.shape[1]
        group_ids = np.zeros(n_features, dtype=int)
        if self.groups:
            for gid, indices in enumerate(self.groups, start=1):
                for idx in indices:
                    if idx < n_features:
                        group_ids[idx] = gid

        self._model = GroupLasso(
            groups=group_ids,
            group_reg=self.alpha,
            l1_reg=0.0,
            scale_reg="inverse_group_size",
            supress_warning=True,
        )
        self._model.fit(X, y.reshape(-1, 1) if y is not None else y)

        coef = self._model.coef_.ravel()
        self._support = np.abs(coef) > 1e-10
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        if self._support is None:
            return X
        return X[:, self._support]

    def get_support(self) -> np.ndarray:
        if self._support is None:
            raise RuntimeError("fit() を先に呼んでください。")
        return self._support
