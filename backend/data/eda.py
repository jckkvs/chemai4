"""
backend/data/eda.py

探索的データ分析（EDA）ユーティリティモジュール。
分布統計、相関分析、外れ値検出、目的変数分析などを提供する。
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 基本統計
# ============================================================

@dataclass
class ColumnStats:
    """1列の統計情報。"""
    name: str
    dtype: str
    n_total: int
    n_null: int
    null_rate: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    top_values: list[tuple[Any, int]] = field(default_factory=list)


def compute_column_stats(df: pd.DataFrame) -> list[ColumnStats]:
    """
    DataFrame の全列の統計情報を計算して返す。

    Args:
        df: 入力DataFrame

    Returns:
        ColumnStats のリスト
    """
    stats: list[ColumnStats] = []
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_null = int(series.isna().sum())
        null_rate = n_null / n_total if n_total > 0 else 0.0
        n_unique = int(series.nunique())

        cs = ColumnStats(
            name=col,
            dtype=str(series.dtype),
            n_total=n_total,
            n_null=n_null,
            null_rate=null_rate,
            n_unique=n_unique,
        )

        numeric = series.dropna()
        if pd.api.types.is_numeric_dtype(series):
            numeric_f = numeric.astype(float)
            cs.mean = float(numeric_f.mean())
            cs.std = float(numeric_f.std()) if len(numeric_f) > 1 else 0.0
            cs.min = float(numeric_f.min())
            cs.p25 = float(numeric_f.quantile(0.25))
            cs.p50 = float(numeric_f.quantile(0.50))
            cs.p75 = float(numeric_f.quantile(0.75))
            cs.max = float(numeric_f.max())
            cs.skewness = float(numeric_f.skew()) if len(numeric_f) > 2 else None
            cs.kurtosis = float(numeric_f.kurtosis()) if len(numeric_f) > 3 else None
        else:
            vc = series.value_counts().head(5)
            cs.top_values = [(str(v), int(c)) for v, c in zip(vc.index, vc.values)]

        stats.append(cs)
    return stats


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    DataFrame 全体のサマリーを返す。

    Returns:
        {
            "n_rows": int, "n_cols": int,
            "n_numeric": int, "n_categorical": int, "n_datetime": int,
            "total_null_rate": float,
            "n_duplicates": int,
            "memory_mb": float,
        }
    """
    n_rows, n_cols = df.shape
    n_numeric = int(df.select_dtypes(include="number").shape[1])
    n_categorical = int(df.select_dtypes(include=["object", "category"]).shape[1])
    n_datetime = int(df.select_dtypes(include="datetime").shape[1])
    total_null = int(df.isna().sum().sum())
    total_cells = n_rows * n_cols
    total_null_rate = total_null / total_cells if total_cells > 0 else 0.0
    n_duplicates = int(df.duplicated().sum())
    memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_datetime": n_datetime,
        "total_null_rate": round(total_null_rate, 4),
        "n_duplicates": n_duplicates,
        "memory_mb": round(memory_mb, 3),
    }


# ============================================================
# 相関分析
# ============================================================

def compute_correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    数値列の相関行列を計算して返す。

    Args:
        df: 入力DataFrame
        method: "pearson" | "spearman" | "kendall"
        target_col: 指定時は目的変数との相関のみを1列のDataFrameで返す

    Returns:
        相関行列（DataFrame）、または目的変数との相関（Series）
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("相関計算には最低2列の数値列が必要です。")

    # 定数列を除外してから相関計算（ConstantInputWarningを回避）
    non_constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
    if len(non_constant_cols) < 2:
        # 全て定数列の場合はNaN行列を返す
        return pd.DataFrame(
            np.nan, index=numeric_df.columns, columns=numeric_df.columns
        )
    filtered_df = numeric_df[non_constant_cols]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        corr_matrix = filtered_df.corr(method=method)
        # 除外した列をNaNで埋めて元の形に戻す
        for col in numeric_df.columns:
            if col not in non_constant_cols:
                corr_matrix[col] = np.nan
                corr_matrix.loc[col, :] = np.nan
        corr_matrix = corr_matrix.reindex(
            index=numeric_df.columns, columns=numeric_df.columns
        )

    if target_col and target_col in corr_matrix.columns:
        return corr_matrix[[target_col]].drop(index=target_col, errors="ignore")

    return corr_matrix


# ============================================================
# 外れ値検出
# ============================================================

@dataclass
class OutlierResult:
    """外れ値検出結果。"""
    col: str
    method: str
    n_outliers: int
    outlier_rate: float
    lower_bound: float | None
    upper_bound: float | None
    outlier_indices: list[int] = field(default_factory=list)


def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    cols: list[str] | None = None,
) -> list[OutlierResult]:
    """
    外れ値を検出して結果を返す。

    Args:
        df: 入力DataFrame
        method: "iqr" | "zscore" | "modified_zscore"
        k: IQR法の係数（デフォルト1.5）
        z_threshold: Zスコア法の閾値（デフォルト3.0）
        cols: 検出対象列（None=全数値列）

    Returns:
        OutlierResult のリスト
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = cols if cols else numeric_cols

    results: list[OutlierResult] = []
    for col in target_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        lower: float | None = None
        upper: float | None = None

        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask = (df[col] < lower) | (df[col] > upper)

        elif method == "zscore":
            mu = float(series.mean())
            sigma = float(series.std())
            if sigma == 0:
                continue
            z_scores = (df[col] - mu) / sigma
            mask = z_scores.abs() > z_threshold
            lower = mu - z_threshold * sigma
            upper = mu + z_threshold * sigma

        elif method == "modified_zscore":
            median = float(series.median())
            mad = float((series - median).abs().median())
            if mad == 0:
                continue
            m_scores = 0.6745 * (df[col] - median) / mad
            mask = m_scores.abs() > z_threshold
            lower = median - z_threshold * mad / 0.6745
            upper = median + z_threshold * mad / 0.6745

        else:
            raise ValueError(f"未知の外れ値検出手法: {method}")

        outlier_indices = df.index[mask.fillna(False)].tolist()
        n_outliers = len(outlier_indices)
        n_valid = int(df[col].notna().sum())

        results.append(OutlierResult(
            col=col,
            method=method,
            n_outliers=n_outliers,
            outlier_rate=n_outliers / n_valid if n_valid > 0 else 0.0,
            lower_bound=lower,
            upper_bound=upper,
            outlier_indices=list(map(int, outlier_indices[:100])),  # 最大100件
        ))
    return results


# ============================================================
# 分布分析
# ============================================================

def compute_distribution(
    series: pd.Series,
    bins: int = 30,
) -> dict[str, Any]:
    """
    1列のヒストグラムデータとカーネル密度推定用データを返す。

    Returns:
        {
            "counts": list[int],
            "bin_edges": list[float],
            "bin_centers": list[float],
        }
    """
    s = series.dropna()
    if not pd.api.types.is_numeric_dtype(s):
        vc = s.value_counts().head(30)
        return {"categories": vc.index.tolist(), "counts": vc.values.tolist()}

    counts, bin_edges = np.histogram(s, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
    }


# ============================================================
# 目的変数分析
# ============================================================

def analyze_target(
    df: pd.DataFrame,
    target_col: str,
    task: str = "auto",
) -> dict[str, Any]:
    """
    目的変数の統計・分布情報を返す。

    Args:
        df: 入力DataFrame
        target_col: 目的変数列名
        task: "regression" | "classification" | "auto"

    Returns:
        統計情報の辞書
    """
    if target_col not in df.columns:
        raise ValueError(f"目的変数列 '{target_col}' がデータに存在しません。")

    series = df[target_col].dropna()
    n = len(series)

    # タスク自動判定
    if task == "auto":
        if pd.api.types.is_float_dtype(series):
            task = "regression"
        elif series.nunique() <= 20:
            task = "classification"
        else:
            task = "regression"

    result: dict[str, Any] = {
        "col": target_col,
        "task": task,
        "n": n,
        "n_null": int(df[target_col].isna().sum()),
        "null_rate": float(df[target_col].isna().mean()),
        "n_unique": int(series.nunique()),
    }

    if task == "regression":
        result.update({
            "mean": float(series.mean()),
            "std": float(series.std()) if n > 1 else 0.0,
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()) if n > 2 else None,
            "kurtosis": float(series.kurtosis()) if n > 3 else None,
        })
    else:
        vc = series.value_counts()
        result["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        result["class_balance"] = {str(k): round(v / len(series), 4) for k, v in vc.items()}
        result["is_balanced"] = float(vc.min() / vc.max()) >= 0.5 if len(vc) >= 2 else True

    return result


# ============================================================
# 新規追加EDA機能 (QQプロット、VIF)
# ============================================================

def compute_vif(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    threshold: float = 10.0,
) -> dict[str, Any]:
    """
    分散膨張因子（VIF）を計算し、多重共線性を診断する。
    
    Args:
        df: 数値のみを含むDataFrame
        cols: 対象列（None=全数値列）
        threshold: VIF閾値（デフォルト10.0）
    
    Returns:
        {
            "vif_values": {"col_name": vif_value, ...},
            "high_vif_cols": ["col1", "col2"],  # threshold 超え
            "recommendation": str,
        }
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return {"error": "statsmodels is required for VIF calculation"}
    
    numeric_df = df.select_dtypes(include="number")
    target_cols = cols if cols else numeric_df.columns.tolist()
    
    # 定数列・欠損列を除外
    valid_cols = []
    for c in target_cols:
        if c in numeric_df.columns and numeric_df[c].nunique() > 1 and numeric_df[c].notna().all():
            valid_cols.append(c)
            
    if len(valid_cols) < 2:
        return {"error": "VIF計算には最低2つの有効な数値列（欠損なし、定数でない）が必要です"}
        
    X = numeric_df[valid_cols].copy()
    
    # statsmodelsのVIF計算のため定数項を追加するケースもあるが、
    # スケーリングして計算する方が安定するため、ここでは簡単にそのまま計算（実用上は標準化推奨）
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    vif_dict = {}
    high_vif = []
    
    for i, col in enumerate(valid_cols):
        try:
            vif = float(variance_inflation_factor(X_scaled, i))
            vif_dict[col] = vif
            if vif > threshold:
                high_vif.append(col)
        except Exception:
            # 完璧な多重共線性がある場合などはエラーになることがある
            vif_dict[col] = float('inf')
            high_vif.append(col)
            
    # 推奨事項の生成
    rec = "多重共線性の問題は検出されませんでした。"
    if high_vif:
        rec = f"高い多重共線性が検出されました。以下の列のいずれかを除外することを検討してください: {', '.join(high_vif)}"
        
    return {
        "vif_values": vif_dict,
        "high_vif_cols": high_vif,
        "recommendation": rec,
    }


def compute_qq_data(series: pd.Series) -> dict[str, Any]:
    """
    QQプロット用の理論分位点・実測分位点を計算する。
    
    Args:
        series: 対象の数値Series
        
    Returns:
        {
            "theoretical_quantiles": list[float],
            "sample_quantiles": list[float],
            "r_squared": float (適合度)
        }
    """
    try:
        from scipy import stats
    except ImportError:
        return {"error": "scipy is required for QQ plot calculation"}
        
    s = series.dropna()
    if not pd.api.types.is_numeric_dtype(s) or len(s) < 2:
        return {"error": "QQプロットには有効な数値データが必要です"}
        
    # sample数が大きすぎる場合は間引く (描画負荷軽減)
    if len(s) > 1000:
        s = s.sample(1000, random_state=42)
        
    # stats.probplotを利用
    (theoretical, sample), (slope, intercept, r) = stats.probplot(s, dist="norm")
    
    return {
        "theoretical_quantiles": theoretical.tolist(),
        "sample_quantiles": sample.tolist(),
        "r_squared": float(r**2),
        "line": {
            "slope": float(slope),
            "intercept": float(intercept)
        }
    }


def convert_numpy_to_list(obj: Any) -> Any:
    """numpy配列や数値をPythonネイティブ型に変換する（再帰的）"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {k: convert_numpy_to_list(v) for k, v in obj.__dict__.items()}
    return obj
