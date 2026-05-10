"""
backend/models/cv_manager.py

sklearn の全クロスバリデーション手法を統合管理するモジュール。
要件定義書 §3.5 に記載された全CV手法に対応する。
"""
from __future__ import annotations

import logging
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Iterator, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneOut,
    LeavePOut,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    TimeSeriesSplit,
    PredefinedSplit,
    cross_validate,
)

from backend.utils.config import RANDOM_STATE

logger = logging.getLogger(__name__)


# ============================================================
# WalkForward（時系列 Walk-Forward Validation）
# ============================================================

class WalkForwardSplit:
    """
    時系列データのウォークフォワード検証（拡張窓方式）。
    TimeSerieseSplit の gap パラメータ対応バージョン。

    Args:
        n_splits: 分割数
        min_train_size: 最小学習データのサンプル数
        gap: 学習データとテストデータの間のギャップ数
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int | None = None,
        gap: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        (train_indices, test_indices) のイテレータを返す。

        Args:
            X: 特徴量行列
            y: 目的変数（未使用、互換性のため）
            groups: グループラベル（未使用）

        Yields:
            (train_indices, test_indices) のタプル
        """
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        min_train = self.min_train_size or (n // (self.n_splits + 1))

        test_size = (n - min_train - self.gap) // self.n_splits
        if test_size <= 0:
            raise ValueError(
                f"データ数({n})が少なすぎてWalkForward分割できません。"
                f"n_splits={self.n_splits}を減らしてください。"
            )

        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            if test_end > n:
                break
            yield np.arange(train_end), np.arange(test_start, test_end)

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return self.n_splits


# ============================================================
# CV レジストリ
# ============================================================

# CV手法を文字列キーで取得するためのレジストリ
# (key, 日本語名, クラス, 必要引数フラグ)
_CV_REGISTRY: dict[str, dict[str, Any]] = {
    "groupfold": {
        "name": "Group K-Fold",
        "description": "グループを考慮したK-Fold",
        "class": GroupKFold,
        "default_params": {"n_splits": 5},
        "requires_groups": True,
        "requires_classification": False,
    },
    "stratifiedfold": {
        "name": "Stratified K-Fold",
        "description": "クラス比率を保持するK-Fold（分類向け）",
        "class": StratifiedKFold,
        "default_params": {"n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE},
        "requires_groups": False,
        "requires_classification": True,
    },
    "loo": {
        "name": "Leave-One-Out (LOO)",
        "description": "1サンプルずつ除外",
        "class": LeaveOneOut,
        "default_params": {},
        "requires_groups": False,
        "requires_classification": False,
    },
    "logo": {
        "name": "Leave-One-Group-Out",
        "description": "グループ単位でLeave-One-Out",
        "class": LeaveOneGroupOut,
        "default_params": {},
        "requires_groups": True,
        "requires_classification": False,
    },
    "walkthrough": {
        "name": "Walk-Forward Validation (Walkthrough)",
        "description": "時系列ウォークフォワード検証（拡張窓方式）",
        "class": WalkForwardSplit,
        "default_params": {"n_splits": 5, "gap": 0},
        "requires_groups": False,
        "requires_classification": False,
    },
    # 既存のレガシーキーも保持
    "kfold": {
        "name": "K-Fold",
        "description": "標準的なK分割交差検証",
        "class": KFold,
        "default_params": {"n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE},
        "requires_groups": False,
        "requires_classification": False,
    },
    "stratified_kfold": {
        "name": "Stratified K-Fold",
        "description": "クラス比率を保持するK-Fold（分類向け）",
        "class": StratifiedKFold,
        "default_params": {"n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE},
        "requires_groups": False,
        "requires_classification": True,
    },
    "group_kfold": {
        "name": "Group K-Fold",
        "description": "グループを考慮したK-Fold",
        "class": GroupKFold,
        "default_params": {"n_splits": 5},
        "requires_groups": True,
        "requires_classification": False,
    },
    "stratified_group_kfold": {
        "name": "Stratified Group K-Fold",
        "description": "グループとクラス比率を考慮したK-Fold",
        "class": StratifiedGroupKFold,
        "default_params": {"n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE},
        "requires_groups": True,
        "requires_classification": True,
    },
    "lpo": {
        "name": "Leave-P-Out",
        "description": "Pサンプルずつ除外",
        "class": LeavePOut,
        "default_params": {"p": 2},
        "requires_groups": False,
        "requires_classification": False,
    },
    "lpgo": {
        "name": "Leave-P-Groups-Out",
        "description": "Pグループずつ除外",
        "class": LeavePGroupsOut,
        "default_params": {"n_groups": 2},
        "requires_groups": True,
        "requires_classification": False,
    },
    "repeated_kfold": {
        "name": "Repeated K-Fold",
        "description": "複数回繰り返すK-Fold",
        "class": RepeatedKFold,
        "default_params": {"n_splits": 5, "n_repeats": 3, "random_state": RANDOM_STATE},
        "requires_groups": False,
        "requires_classification": False,
    },
    "repeated_stratified_kfold": {
        "name": "Repeated Stratified K-Fold",
        "description": "複数回繰り返すStratified K-Fold",
        "class": RepeatedStratifiedKFold,
        "default_params": {"n_splits": 5, "n_repeats": 3, "random_state": RANDOM_STATE},
        "requires_groups": False,
        "requires_classification": True,
    },
    "shuffle_split": {
        "name": "Shuffle Split",
        "description": "ランダム分割（サンプル数指定）",
        "class": ShuffleSplit,
        "default_params": {
            "n_splits": 5, "test_size": 0.2, "random_state": RANDOM_STATE
        },
        "requires_groups": False,
        "requires_classification": False,
    },
    "stratified_shuffle_split": {
        "name": "Stratified Shuffle Split",
        "description": "クラス比率を保持したシャッフル分割",
        "class": StratifiedShuffleSplit,
        "default_params": {
            "n_splits": 5, "test_size": 0.2, "random_state": RANDOM_STATE
        },
        "requires_groups": False,
        "requires_classification": True,
    },
    "group_shuffle_split": {
        "name": "Group Shuffle Split",
        "description": "グループを考慮したシャッフル分割",
        "class": GroupShuffleSplit,
        "default_params": {
            "n_splits": 5, "test_size": 0.2, "random_state": RANDOM_STATE
        },
        "requires_groups": True,
        "requires_classification": False,
    },
    "timeseries": {
        "name": "Time Series Split",
        "description": "時系列向け（未来にテスト、過去に学習）",
        "class": TimeSeriesSplit,
        "default_params": {"n_splits": 5},
        "requires_groups": False,
        "requires_classification": False,
    },
    "predefined": {
        "name": "Predefined Split",
        "description": "事前定義された分割（test_fold配列が必要）",
        "class": PredefinedSplit,
        "default_params": {},
        "requires_groups": False,
        "requires_classification": False,
    },
    "walk_forward": {
        "name": "Walk-Forward Validation",
        "description": "時系列ウォークフォワード検証（拡張窓）",
        "class": WalkForwardSplit,
        "default_params": {"n_splits": 5},
        "requires_groups": False,
        "requires_classification": False,
    },
    "leave_one_group_out": {
        "name": "Leave-One-Group-Out",
        "description": "グループ単位でLeave-One-Out（logoのエイリアス）",
        "class": LeaveOneGroupOut,
        "default_params": {},
        "requires_groups": True,
        "requires_classification": False,
    },
    "scaffold": {
        "name": "Scaffold Split",
        "description": "分子骨格に基づく分割",
        "class": GroupKFold,
        "default_params": {"n_splits": 5},
        "requires_groups": True, # Technically requires_smiles but handled differently, relies on group input for Scaffold
        "requires_classification": False,
    },
}

def _get_cv_class(class_name: str) -> Type[Any]:
    """sklearn.model_selection から CV クラスを動的に取得する。"""
    # 既知のエイリアス解決
    alias_map = {
        "groupfold": "GroupKFold",
        "stratifiedfold": "StratifiedKFold",
        "walkthrough": "WalkForwardSplit",
        "kfold": "KFold",
        "stratified_kfold": "StratifiedKFold",
        "group_kfold": "GroupKFold",
        "stratified_group_kfold": "StratifiedGroupKFold",
        "loo": "LeaveOneOut",
        "lpo": "LeavePOut",
        "logo": "LeaveOneGroupOut",
        "lpgo": "LeavePGroupsOut",
        "repeated_kfold": "RepeatedKFold",
        "repeated_stratified_kfold": "RepeatedStratifiedKFold",
        "shuffle_split": "ShuffleSplit",
        "stratified_shuffle_split": "StratifiedShuffleSplit",
        "group_shuffle_split": "GroupShuffleSplit",
        "timeseries": "TimeSeriesSplit",
        "predefined": "PredefinedSplit",
    }
    
    target_name = alias_map.get(class_name.lower(), class_name)
    
    # 内部クラス（WalkForward等）をチェック
    if target_name == "WalkForwardSplit":
        return WalkForwardSplit

    # sklearn から取得
    try:
        mod = importlib.import_module("sklearn.model_selection")
        cls = getattr(mod, target_name, None)
        if cls is None:
            raise AttributeError(f"sklearn.model_selection にクラス '{target_name}' が見つかりません。")
        return cls
    except (ImportError, AttributeError) as e:
        raise ValueError(f"CV手法 '{class_name}' (クラス名: {target_name}) をロードできませんでした: {e}")


@dataclass
class CVConfig:
    """クロスバリデーションの設定。"""
    cv_key: str = "stratified_kfold"
    n_splits: int = 5
    groups: np.ndarray | pd.Series | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


def get_cv(config: CVConfig) -> Any:
    """
    CVConfig に基づいて CV スプリッタを返す。
    全引数が extra_params 経由で設定可能。

    Args:
        config: CVConfig インスタンス

    Returns:
        sklearn互換のCV スプリッタ
    """
    key = config.cv_key
    
    # クラスの取得
    if key in _CV_REGISTRY:
        entry = _CV_REGISTRY[key]
        cv_class = entry["class"]
        params = {**entry.get("default_params", {})}
    else:
        cv_class = _get_cv_class(key)
        params = {}

    # 基本パラメータの上書き
    if "n_splits" in inspect.signature(cv_class.__init__).parameters:
        params["n_splits"] = config.n_splits
    
    # ユーザ指定の追加パラメータ（全引数対応）
    params.update(config.extra_params)

    # 特殊ケース: PredefinedSplit
    if cv_class.__name__ == "PredefinedSplit" and "test_fold" not in params:
        raise ValueError("PredefinedSplit には 'test_fold' 配列が必要です。")

    # 引数バリデーション & 型変換
    sig = inspect.signature(cv_class.__init__)
    valid_params = {}
    invalid_params = []
    
    for k, v in params.items():
        if k not in sig.parameters and not sig.parameters.get("kwargs"):
            invalid_params.append(k)
            continue
            
        # 型推論 & 変換
        param_info = sig.parameters.get(k)
        anno = param_info.annotation if param_info else None
        
        if v is not None:
            try:
                # 1. アノテーションが明確な場合
                if anno is int:
                    valid_params[k] = int(v)
                elif anno is float:
                    valid_params[k] = float(v)
                elif anno is bool:
                    if isinstance(v, str):
                        valid_params[k] = v.lower() in ("true", "1", "yes")
                    else:
                        valid_params[k] = bool(v)
                
                # 2. アノテーションがないか Any / Parameter.empty の場合、値から推論
                elif isinstance(v, str):
                    low_v = v.lower()
                    if low_v in ("true", "false", "yes", "no"):
                        valid_params[k] = low_v in ("true", "yes")
                    elif v.isdigit():
                        valid_params[k] = int(v)
                    else:
                        try:
                            valid_params[k] = float(v)
                        except ValueError:
                            valid_params[k] = v
                else:
                    valid_params[k] = v
            except (ValueError, TypeError):
                valid_params[k] = v
        else:
            valid_params[k] = v
            
    # 特殊な依存関係の調整 (KFold等: random_state を指定したら shuffle=True が必須)
    if valid_params.get("random_state") is not None and "shuffle" in sig.parameters:
        if not valid_params.get("shuffle"):
            logger.info(f"CVクラス {cv_class.__name__}: random_state が指定されたため shuffle=True に自動設定します。")
            valid_params["shuffle"] = True

    if invalid_params:
        logger.warning(f"CVクラス {cv_class.__name__} に無効な引数が指定されました（無視します）: {invalid_params}")

    try:
        return cv_class(**valid_params)
    except Exception as e:
        raise ValueError(f"CVスプリッタ {cv_class.__name__} の初期化に失敗しました: {e}")


def list_cv_methods(
    task: str = "regression",
    requires_groups: bool | None = None,
) -> list[dict[str, Any]]:
    """
    利用可能なCV手法の一覧を返す。

    Args:
        task: "regression" | "classification"
        requires_groups: True の場合はグループが必要なもののみ返す

    Returns:
        [{key, name, description, requires_groups}] のリスト
    """
    results = []
    for key, entry in _CV_REGISTRY.items():
        if task == "regression" and entry.get("requires_classification"):
            continue
        if requires_groups is not None and entry["requires_groups"] != requires_groups:
            continue
        results.append({
            "key": key,
            "name": entry["name"],
            "description": entry["description"],
            "requires_groups": entry["requires_groups"],
        })
    return results


def run_cross_validation(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv_config: CVConfig,
    scoring: str | list[str] | dict[str, Any],
    groups: np.ndarray | pd.Series | None = None,
    n_jobs: int = -1,
    return_train_score: bool = True,
    fit_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    クロスバリデーションを実行して結果を返す。

    Args:
        model: sklearn互換の推定器
        X: 特徴量行列
        y: 目的変数
        cv_config: CVConfig インスタンス
        scoring: sklearn scoring 文字列またはリスト
        groups: グループラベル（Group系CVで必要）
        n_jobs: 並列数
        return_train_score: 訓練スコアも返す場合 True
        fit_params: fit()に渡す追加パラメータ

    Returns:
        {
            "test_score": np.ndarray,
            "train_score": np.ndarray (return_train_score=Trueのとき),
            "fit_time": np.ndarray,
            "score_time": np.ndarray,
            "mean_test_score": float,
            "std_test_score": float,
        }
    """
    cv = get_cv(cv_config)

    # groupsはCV・cross_validate両方に渡す
    effective_groups = groups if groups is not None else cv_config.groups

    # sklearn 1.6+ では fit_params が廃止 → 空でない場合のみ params で渡す
    import sklearn
    sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

    cv_kwargs: dict[str, Any] = dict(
        cv=cv,
        scoring=scoring,
        groups=effective_groups,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
    )
    if fit_params:
        if sklearn_version >= (1, 6):
            cv_kwargs["params"] = fit_params
        else:
            cv_kwargs["fit_params"] = fit_params

    results = cross_validate(model, X, y, **cv_kwargs)

    # sklearn 1.6+では単一scoringでも "test_score" / "train_score" キーで返る。
    # 後方互换性のため "test_{scoring}" キーも追加する。
    if isinstance(scoring, str):
        expected_key = f"test_{scoring}"
        if expected_key not in results and "test_score" in results:
            results[expected_key] = results["test_score"]
        if return_train_score:
            train_key = f"train_{scoring}"
            if train_key not in results and "train_score" in results:
                results[train_key] = results["train_score"]

    # 集計スコアを追加
    if isinstance(scoring, str):
        test_key = f"test_{scoring}"
        if test_key in results:
            results["mean_test_score"] = float(np.mean(results[test_key]))
            results["std_test_score"] = float(np.std(results[test_key]))

    mean_v = results.get('mean_test_score', None)
    std_v = results.get('std_test_score', 0)
    mean_str = f"{mean_v:.4f}" if isinstance(mean_v, float) else str(mean_v)
    logger.info(f"CV完了: {cv_config.cv_key} / mean={mean_str} ±{std_v:.4f}")
    return results
