"""
backend/data/data_cleaner.py

インタラクティブなデータクリーニング操作を提供するモジュール。
EDAタブから直接呼び出される純粋関数群で構成。

Implements: F-CLEAN-001〜006
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# データクラス: 操作ログ
# ============================================================

@dataclass
class CleaningAction:
    """1回のクリーニング操作のログレコード。

    Attributes:
        action_type: 操作種別 (例: "drop_columns", "drop_missing_rows")
        description: 人間向けの操作説明
        rows_before: 操作前の行数
        rows_after: 操作後の行数
        cols_before: 操作前の列数
        cols_after: 操作後の列数
        details: 追加情報 (除外した列名リスト等)
        timestamp: 操作実行時刻 (ISO 8601)
    """
    action_type: str
    description: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def rows_removed(self) -> int:
        """除去された行数。"""
        return self.rows_before - self.rows_after

    @property
    def cols_removed(self) -> int:
        """除去された列数。"""
        return self.cols_before - self.cols_after


# ============================================================
# クリーニング関数群
# ============================================================

def drop_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, CleaningAction]:
    """指定列をDataFrameから除外する。

    Args:
        df: 入力DataFrame
        columns: 除外する列名のリスト

    Returns:
        (クリーニング後のDataFrame, CleaningAction)

    Raises:
        ValueError: columns が空の場合
    """
    if not columns:
        raise ValueError("除外する列が指定されていません。")

    existing = [c for c in columns if c in df.columns]
    if not existing:
        raise ValueError(
            f"指定された列がDataFrameに存在しません: {columns}"
        )

    rows_before, cols_before = df.shape
    result = df.drop(columns=existing)

    action = CleaningAction(
        action_type="drop_columns",
        description=f"{len(existing)}列を除外: {', '.join(existing[:5])}"
                    + (f" 他{len(existing)-5}列" if len(existing) > 5 else ""),
        rows_before=rows_before,
        rows_after=len(result),
        cols_before=cols_before,
        cols_after=result.shape[1],
        details={"dropped_columns": existing},
    )
    logger.info("drop_columns: %d列を除外", len(existing))
    return result, action


def drop_rows_with_missing(
    df: pd.DataFrame,
    threshold: float = 0.5,
    subset: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, CleaningAction]:
    """欠損率が閾値を超える行を削除する。

    各行の欠損率（= 欠損セル数 / 全列数）が threshold 以上の行を除去。

    Args:
        df: 入力DataFrame
        threshold: 欠損率の閾値 (0.0〜1.0)。0.0で欠損が1つでもあれば削除。
        subset: チェック対象列。Noneの場合は全列。

    Returns:
        (クリーニング後のDataFrame, CleaningAction)

    Raises:
        ValueError: threshold が 0〜1 の範囲外の場合
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold は 0.0〜1.0 の範囲で指定してください: {threshold}")

    rows_before, cols_before = df.shape

    if subset is not None:
        check_cols = [c for c in subset if c in df.columns]
    else:
        check_cols = list(df.columns)

    if not check_cols:
        # チェック対象列がない場合は変更なし
        action = CleaningAction(
            action_type="drop_missing_rows",
            description="チェック対象列がないため操作なし",
            rows_before=rows_before,
            rows_after=rows_before,
            cols_before=cols_before,
            cols_after=cols_before,
        )
        return df.copy(), action

    n_check = len(check_cols)
    missing_per_row = df[check_cols].isna().sum(axis=1)

    if threshold == 0.0:
        # 欠損が1つでもあれば削除
        mask = missing_per_row == 0
    else:
        missing_rate_per_row = missing_per_row / n_check
        mask = missing_rate_per_row < threshold

    result = df.loc[mask].reset_index(drop=True)
    n_removed = rows_before - len(result)

    action = CleaningAction(
        action_type="drop_missing_rows",
        description=f"欠損率≥{threshold:.0%}の行を{n_removed}行削除",
        rows_before=rows_before,
        rows_after=len(result),
        cols_before=cols_before,
        cols_after=result.shape[1],
        details={"threshold": threshold, "rows_removed": n_removed, "subset": check_cols},
    )
    logger.info("drop_rows_with_missing: %d行を削除 (閾値=%.2f)", n_removed, threshold)
    return result, action


def remove_constant_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, CleaningAction]:
    """ユニーク値が1以下の定数列を除去する。

    Args:
        df: 入力DataFrame

    Returns:
        (クリーニング後のDataFrame, CleaningAction)
    """
    rows_before, cols_before = df.shape

    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    if not const_cols:
        action = CleaningAction(
            action_type="remove_constant_columns",
            description="定数列は見つかりませんでした",
            rows_before=rows_before,
            rows_after=rows_before,
            cols_before=cols_before,
            cols_after=cols_before,
            details={"constant_columns": []},
        )
        return df.copy(), action

    result = df.drop(columns=const_cols)

    action = CleaningAction(
        action_type="remove_constant_columns",
        description=f"定数列を{len(const_cols)}列除去: {', '.join(const_cols[:5])}"
                    + (f" 他{len(const_cols)-5}列" if len(const_cols) > 5 else ""),
        rows_before=rows_before,
        rows_after=len(result),
        cols_before=cols_before,
        cols_after=result.shape[1],
        details={"constant_columns": const_cols},
    )
    logger.info("remove_constant_columns: %d列を除去", len(const_cols))
    return result, action


def clip_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    columns: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, CleaningAction]:
    """IQR法で外れ値をクリッピングする。

    Q1 - iqr_multiplier*IQR 〜 Q3 + iqr_multiplier*IQR の範囲にclipする。

    Args:
        df: 入力DataFrame
        iqr_multiplier: IQRの倍率 (デフォルト: 1.5)
        columns: 対象列。Noneの場合は全数値列。

    Returns:
        (クリーニング後のDataFrame, CleaningAction)

    Raises:
        ValueError: iqr_multiplier が正でない場合
    """
    if iqr_multiplier <= 0:
        raise ValueError(f"iqr_multiplier は正の数を指定してください: {iqr_multiplier}")

    rows_before, cols_before = df.shape
    result = df.copy()

    if columns is not None:
        target_cols = [c for c in columns if c in result.columns]
    else:
        target_cols = list(result.select_dtypes(include="number").columns)

    total_clipped = 0
    clipped_details: dict[str, int] = {}

    for col in target_cols:
        series = result[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        n_before = ((series < lower) | (series > upper)).sum()
        if n_before > 0:
            result[col] = series.clip(lower=lower, upper=upper)
            clipped_details[col] = int(n_before)
            total_clipped += int(n_before)

    action = CleaningAction(
        action_type="clip_outliers",
        description=f"IQR×{iqr_multiplier}で{total_clipped}値をクリッピング"
                    f"（{len(clipped_details)}列が対象）",
        rows_before=rows_before,
        rows_after=len(result),
        cols_before=cols_before,
        cols_after=result.shape[1],
        details={
            "iqr_multiplier": iqr_multiplier,
            "total_clipped": total_clipped,
            "clipped_per_column": clipped_details,
        },
    )
    logger.info("clip_outliers: %d値をクリッピング (IQR×%.1f)", total_clipped, iqr_multiplier)
    return result, action


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, CleaningAction]:
    """重複行を除去する。

    Args:
        df: 入力DataFrame
        subset: 重複判定に使用する列。Noneの場合は全列。
        keep: 保持する重複行 ("first", "last", False)

    Returns:
        (クリーニング後のDataFrame, CleaningAction)
    """
    rows_before, cols_before = df.shape

    result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    n_removed = rows_before - len(result)

    action = CleaningAction(
        action_type="remove_duplicates",
        description=f"重複行を{n_removed}行除去",
        rows_before=rows_before,
        rows_after=len(result),
        cols_before=cols_before,
        cols_after=result.shape[1],
        details={"rows_removed": n_removed, "keep": keep},
    )
    logger.info("remove_duplicates: %d行を除去", n_removed)
    return result, action


# ============================================================
# プレビュー・診断ヘルパー
# ============================================================

def preview_missing_impact(
    df: pd.DataFrame,
    threshold: float = 0.5,
    subset: Optional[Sequence[str]] = None,
) -> int:
    """欠損行削除の影響行数をプレビューする（実際には削除しない）。

    Args:
        df: 入力DataFrame
        threshold: 欠損率の閾値 (0.0〜1.0)
        subset: チェック対象列

    Returns:
        削除される行数
    """
    if subset is not None:
        check_cols = [c for c in subset if c in df.columns]
    else:
        check_cols = list(df.columns)

    if not check_cols:
        return 0

    n_check = len(check_cols)
    missing_per_row = df[check_cols].isna().sum(axis=1)

    if threshold == 0.0:
        return int((missing_per_row > 0).sum())

    missing_rate = missing_per_row / n_check
    return int((missing_rate >= threshold).sum())


def preview_outlier_impact(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    columns: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """外れ値クリッピングの影響値数をプレビューする。

    Returns:
        列名 -> クリップされる値の数 の辞書
    """
    if columns is not None:
        target_cols = [c for c in columns if c in df.columns]
    else:
        target_cols = list(df.select_dtypes(include="number").columns)

    result: dict[str, int] = {}
    for col in target_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers > 0:
            result[col] = n_outliers

    return result


def get_cleaning_summary(df: pd.DataFrame) -> dict:
    """現在のDataFrameのクリーニング候補をサマリーで返す。

    Returns:
        dict with keys: n_const_cols, const_cols, n_dup_rows,
                        n_all_missing_cols, all_missing_cols,
                        total_missing_rate
    """
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    all_missing_cols = [c for c in df.columns if df[c].isna().all()]
    n_dup = int(df.duplicated().sum())
    total_missing = float(df.isna().mean().mean())

    return {
        "n_const_cols": len(const_cols),
        "const_cols": const_cols,
        "n_dup_rows": n_dup,
        "n_all_missing_cols": len(all_missing_cols),
        "all_missing_cols": all_missing_cols,
        "total_missing_rate": total_missing,
    }
