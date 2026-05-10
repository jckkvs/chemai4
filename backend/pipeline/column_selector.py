"""
backend/pipeline/column_selector.py

入力列制御モジュール。
mlxtend.preprocessing.ColumnSelector をラップし、
include / exclude / all モードと ColumnMeta（単調性・グループ情報）を管理する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from backend.utils.optional_import import safe_import

logger = logging.getLogger(__name__)

_mlxtend = safe_import("mlxtend.preprocessing", "mlxtend")


# ============================================================
# ColumnMeta — 変数メタ情報
# ============================================================

@dataclass
class ColumnMeta:
    """
    1列のメタ情報（ユーザーが設定）。

    Attributes:
        monotonic: 単調性制約。
            0  = なし
            1  = 単調増加
            -1 = 単調減少
            2  = 単調（方向自動検出: fit時にSpearman相関で判定）
        constraint_strength: 単調性制約の強度。
            None    = デフォルト
            "weak"  = ソフト制約（できるだけ単調にする）
            "strong" = ハード制約（±3σ 全範囲で厳密単調）
        linearity: 線形性のヒント。"linear" | "nonlinear" | "unknown"
        group: グループラベル文字列（GroupLasso等で使用）。None=グループなし
        scale_hint: スケーラー優先ヒント。None=自動 |
            "standard" | "minmax" | "robust" | "maxabs" | "power_yj" | "log" | "none"
        description: ユーザーが記録する列の説明テキスト
        fixed: True のとき特徴量選択で常に保持（選択手法に関わらず残す）
    """
    monotonic: int = 0                          # 0 | 1 | -1 | 2
    constraint_strength: str | None = None      # None | "weak" | "strong"
    linearity: str = "unknown"                  # "linear" | "nonlinear" | "unknown"
    group: str | None = None
    scale_hint: str | None = None
    description: str = ""
    fixed: bool = False

    def to_dict(self) -> dict:
        """シリアライズ（state保存用）。"""
        return {
            "monotonic": self.monotonic,
            "constraint_strength": self.constraint_strength,
            "linearity": self.linearity,
            "group": self.group,
            "scale_hint": self.scale_hint,
            "description": self.description,
            "fixed": self.fixed,
        }

    @staticmethod
    def from_dict(d: dict) -> "ColumnMeta":
        """デシリアライズ（state復元用）。"""
        return ColumnMeta(
            monotonic=int(d.get("monotonic", 0)),
            constraint_strength=d.get("constraint_strength") or None,
            linearity=str(d.get("linearity", "unknown")),
            group=d.get("group") or None,
            scale_hint=d.get("scale_hint") or None,
            description=str(d.get("description", "")),
            fixed=bool(d.get("fixed", False)),
        )

# ============================================================
# ColumnSelectorWrapper — mlxtend ラッパー
# ============================================================

class ColumnSelectorWrapper(BaseEstimator, TransformerMixin):
    """
    mlxtend.preprocessing.ColumnSelector のラッパー。

    3つのモードに対応:
      - "all"     : 全列をパススルー（mlxtend不使用）
      - "include" : 指定列名リスト or インデックス範囲のみを通過
      - "exclude" : 指定列名リストを除いた残り全てを通過

    mlxtend 未インストール時は "include" モードでも
    pandas の列選択ロジックで代替動作する。

    Args:
        mode: "all" | "include" | "exclude"
        columns: 対象列名リスト（include/exclude 時）
        col_range: (start_idx, end_idx) 列インデックス範囲（include 時）。
                   columns より優先度が低い（columnsが指定されていれば無視）。
        column_meta: 列名 → ColumnMeta の辞書。後段のモジュールが参照する。
    """

    def __init__(
        self,
        mode: str = "all",
        columns: list[str] | None = None,
        col_range: tuple[int, int] | None = None,
        column_meta: dict[str, ColumnMeta] | None = None,
    ) -> None:
        self.mode = mode
        self.columns = columns           # ← Noneのまま保持（sklearn clone()互換性のため）
        self.col_range = col_range
        self.column_meta = column_meta   # ← Noneのまま保持

        self._selected_columns: list[str] = []
        self._input_columns: list[str] = []

    # ----------------------------------------------------------
    # fit
    # ----------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "ColumnSelectorWrapper":
        """
        選択列を確定する。

        Args:
            X: 入力 DataFrame
            y: 未使用（sklearn 互換のため）

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ColumnSelectorWrapper は pandas DataFrame のみ受け付けます。"
                f"受け取った型: {type(X).__name__}"
            )

        self._input_columns = X.columns.tolist()
        _cols = self.columns or []           # fit時にNone対e応
        _meta = self.column_meta or {}       # fit時にNone対e応

        if self.mode == "all":
            self._selected_columns = self._input_columns.copy()

        elif self.mode == "include":
            self._selected_columns = self._resolve_include_columns(X, _cols)

        elif self.mode == "exclude":
            exclude_set = set(_cols)
            self._selected_columns = [
                c for c in self._input_columns if c not in exclude_set
            ]

        else:
            raise ValueError(
                f"未知の mode '{self.mode}'。'all' / 'include' / 'exclude' を指定してください。"
            )

        if not self._selected_columns:
            raise ValueError(
                f"選択された列が0件です（mode={self.mode}, columns={_cols}）。"
            )

        logger.info(
            f"ColumnSelectorWrapper.fit(): mode={self.mode}, "
            f"選択列数={len(self._selected_columns)}/{len(self._input_columns)}"
        )
        return self

    # ----------------------------------------------------------
    # transform
    # ----------------------------------------------------------

    def transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> pd.DataFrame:
        """
        選択列のみを含む DataFrame を返す。

        Args:
            X: 入力 DataFrame

        Returns:
            列選択済み DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"ColumnSelectorWrapper は pandas DataFrame のみ受け付けます。"
                f"受け取った型: {type(X).__name__}"
            )

        # fit 時に確定した列のうち、実際に存在する列だけ選択
        available = [c for c in self._selected_columns if c in X.columns]
        missing = set(self._selected_columns) - set(available)
        if missing:
            logger.warning(f"以下の列が transform 時に存在しません（スキップ）: {missing}")

        return X[available].copy()

    # ----------------------------------------------------------
    # ユーティリティ
    # ----------------------------------------------------------

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """選択された列名の配列を返す。"""
        return np.array(self._selected_columns)

    @property
    def selected_columns(self) -> list[str]:
        """fit 後に選択された列名リストを返す。"""
        return self._selected_columns.copy()

    def get_column_meta(self, col: str) -> ColumnMeta:
        """指定列の ColumnMeta を返す。未登録の場合はデフォルトを返す。"""
        _meta = self.column_meta or {}
        return _meta.get(col, ColumnMeta())

    def get_monotonic_constraints(
        self,
        feature_names: list[str] | None = None,
    ) -> tuple[int, ...]:
        """
        feature_names の順序に合わせた単調性制約タプルを返す。
        estimator の monotonic_constraints 引数に直接渡せる形式。

        Args:
            feature_names: 特徴量名のリスト（省略時は選択列を使用）

        Returns:
            (0 | 1 | -1, ...) のタプル
        """
        _meta = self.column_meta or {}
        cols = feature_names or self._selected_columns
        return tuple(_meta.get(c, ColumnMeta()).monotonic for c in cols)

    def get_groups_array(
        self,
        feature_names: list[str] | None = None,
    ) -> list[str | None]:
        """
        feature_names 順のグループラベルリストを返す。
        GroupLasso 等のグループ配列生成に使用する。

        Args:
            feature_names: 特徴量名のリスト（省略時は選択列を使用）

        Returns:
            [group_label | None, ...] のリスト
        """
        _meta = self.column_meta or {}
        cols = feature_names or self._selected_columns
        return [_meta.get(c, ColumnMeta()).group for c in cols]

    # ----------------------------------------------------------
    # 内部ヘルパー
    # ----------------------------------------------------------

    def _resolve_include_columns(self, X: pd.DataFrame, cols: list[str]) -> list[str]:
        """include モードで選択列を解決する。"""
        all_cols = X.columns.tolist()

        # 列名リストが指定されている場合
        if cols:
            valid = []
            for c in cols:
                if c in X.columns:
                    valid.append(c)
                else:
                    logger.warning(f"指定列 '{c}' がデータに存在しません（スキップ）。")

            # mlxtend が使える場合は ColumnSelector 経由で検証（互換テスト）
            if _mlxtend and valid:
                try:
                    from mlxtend.preprocessing import ColumnSelector  # type: ignore
                    cs = ColumnSelector(cols=tuple(valid))
                    cs.fit(X.to_numpy())
                    logger.debug("mlxtend.ColumnSelector による選択列検証OK")
                except Exception as e:
                    logger.debug(f"mlxtend.ColumnSelector 検証スキップ: {e}")

            return valid

        # 列インデックス範囲が指定されている場合
        if self.col_range is not None:
            start, end = self.col_range
            selected = all_cols[start:end]
            logger.info(f"列範囲 [{start}:{end}] → {len(selected)} 列を選択")
            return selected

        # どちらも指定なし → 全列
        logger.warning(
            "include モードですが columns も col_range も未指定です。全列を使用します。"
        )
        return all_cols
