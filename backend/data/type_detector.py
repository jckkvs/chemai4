"""
backend/data/type_detector.py

DataFrameの各列の変数型を自動判定するモジュール。
判定結果は ColumnTransformer のパイプライン構築に使われる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
# from scipy import stats (環境エラー回避のため遅延インポートへ移動)

from backend.utils.config import (
    TYPE_DETECTOR_CARDINALITY_THRESHOLD,
    TYPE_DETECTOR_SKEWNESS_THRESHOLD,
    TYPE_DETECTOR_OUTLIER_IQR_FACTOR,
)

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """変数の種別を表す列挙型。"""
    BINARY = auto()           # 2値変数
    CATEGORY_LOW = auto()     # カテゴリ（低カーディナリティ）
    CATEGORY_HIGH = auto()    # カテゴリ（高カーディナリティ）
    NUMERIC_NORMAL = auto()   # 数値（正規分布に近い）
    NUMERIC_LOG = auto()      # 数値（右裾重い → 対数変換候補）
    NUMERIC_POWER = auto()    # 数値（冪乗分布 → PowerTransformer候補）
    NUMERIC_OUTLIER = auto()  # 数値（外れ値多い → RobustScaler候補）
    PERIODIC = auto()         # 周期変数（角度・時刻等）
    DATETIME = auto()         # 日時変数
    TEXT = auto()             # テキスト変数
    CONSTANT = auto()         # 定数列（分散=0）
    SMILES = auto()           # SMILES化合物


@dataclass
class ColumnInfo:
    """1列の判定結果を保持するデータクラス。"""
    name: str
    col_type: ColumnType
    n_unique: int
    null_rate: float
    skewness: float | None = None
    has_outliers: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        return self.col_type in {
            ColumnType.NUMERIC_NORMAL,
            ColumnType.NUMERIC_LOG,
            ColumnType.NUMERIC_POWER,
            ColumnType.NUMERIC_OUTLIER,
        }

    @property
    def is_categorical(self) -> bool:
        return self.col_type in {
            ColumnType.BINARY,
            ColumnType.CATEGORY_LOW,
            ColumnType.CATEGORY_HIGH,
        }


@dataclass
class DetectionResult:
    """データセット全体の型判定結果。"""
    column_info: dict[str, ColumnInfo]
    smiles_columns: list[str]
    datetime_columns: list[str]
    constant_columns: list[str]
    text_columns: list[str]
    
    @property
    def numeric_columns(self) -> list[str]:
        return self.get_numeric_columns()

    @property
    def categorical_columns(self) -> list[str]:
        return self.get_categorical_columns()

    @property
    def binary_columns(self) -> list[str]:
        return self.get_columns_by_type(ColumnType.BINARY)

    @property
    def ignored_columns(self) -> list[str]:
        # 定数列と一意すぎるID列などを「除外対象」とする
        return self.constant_columns + self.get_columns_by_type(ColumnType.CATEGORY_HIGH)

    def get_columns_by_type(self, col_type: ColumnType) -> list[str]:
        """指定した型の列名リストを返す。"""
        return [
            name for name, info in self.column_info.items()
            if info.col_type == col_type
        ]

    def get_numeric_columns(self) -> list[str]:
        """全数値列の名前リストを返す。"""
        numeric_types = {
            ColumnType.NUMERIC_NORMAL,
            ColumnType.NUMERIC_LOG,
            ColumnType.NUMERIC_POWER,
            ColumnType.NUMERIC_OUTLIER,
        }
        return [
            name for name, info in self.column_info.items()
            if info.col_type in numeric_types
        ]

    def get_categorical_columns(self) -> list[str]:
        """全カテゴリ列の名前リストを返す。"""
        cat_types = {
            ColumnType.BINARY,
            ColumnType.CATEGORY_LOW,
            ColumnType.CATEGORY_HIGH,
        }
        return [
            name for name, info in self.column_info.items()
            if info.col_type in cat_types
        ]

    def summary_table(self) -> pd.DataFrame:
        """判定結果のサマリーDataFrameを返す（GUI表示用）。"""
        rows = []
        for name, info in self.column_info.items():
            rows.append({
                "列名": name,
                "判定種別": info.col_type.name,
                "ユニーク数": info.n_unique,
                "欠損率": f"{info.null_rate:.1%}",
                "歪度": f"{info.skewness:.2f}" if info.skewness is not None else "-",
                "外れ値": "あり" if info.has_outliers else "なし",
                "備考": "; ".join(info.notes),
            })
        return pd.DataFrame(rows)


class TypeDetector:
    """
    DataFrameの各列の変数型を自動判定するクラス。

    Implements: 要件定義書 §3.2 変数型自動判定

    Args:
        cardinality_threshold: カテゴリ(少)/カテゴリ(多)の境界ユニーク数
        skewness_threshold: 歪度がこれ以上の場合に右裾重いと判定
        outlier_iqr_factor: IQR外れ値判定の係数
        smiles_col_hints: SMILES列名のヒント（リスト、部分一致）
        periodic_cols: 周期変数として扱う列名のリスト
    """

    _SMILES_HINTS = ["smiles", "smi", "structure", "mol", "compound"]

    def __init__(
        self,
        cardinality_threshold: int = TYPE_DETECTOR_CARDINALITY_THRESHOLD,
        skewness_threshold: float = TYPE_DETECTOR_SKEWNESS_THRESHOLD,
        outlier_iqr_factor: float = TYPE_DETECTOR_OUTLIER_IQR_FACTOR,
        smiles_col_hints: list[str] | None = None,
        periodic_cols: list[str] | None = None,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.skewness_threshold = skewness_threshold
        self.outlier_iqr_factor = outlier_iqr_factor
        self.smiles_col_hints = smiles_col_hints or self._SMILES_HINTS
        self.periodic_cols = set(periodic_cols or [])

    def detect(self, df: pd.DataFrame) -> DetectionResult:
        """
        DataFrameの全列を解析して判定結果を返す。

        Args:
            df: 解析対象のDataFrame

        Returns:
            DetectionResult インスタンス
        """
        column_info: dict[str, ColumnInfo] = {}

        for col in df.columns:
            info = self._detect_column(df[col], col)
            column_info[col] = info
            logger.debug(f"  {col}: {info.col_type.name} (null={info.null_rate:.1%})")

        result = DetectionResult(
            column_info=column_info,
            smiles_columns=[
                n for n, i in column_info.items() if i.col_type == ColumnType.SMILES
            ],
            datetime_columns=[
                n for n, i in column_info.items() if i.col_type == ColumnType.DATETIME
            ],
            constant_columns=[
                n for n, i in column_info.items() if i.col_type == ColumnType.CONSTANT
            ],
            text_columns=[
                n for n, i in column_info.items() if i.col_type == ColumnType.TEXT
            ],
        )
        logger.info(f"型判定完了: {len(column_info)}列")
        return result

    def _detect_column(self, series: pd.Series, name: str) -> ColumnInfo:
        """1列の型を判定して ColumnInfo を返す。"""
        n_total = len(series)
        n_null = series.isna().sum()
        null_rate = n_null / n_total if n_total > 0 else 0.0
        series_clean = series.dropna()
        n_unique = series_clean.nunique()
        notes: list[str] = []

        # 定数列（分散=0）
        if n_unique <= 1:
            return ColumnInfo(
                name=name, col_type=ColumnType.CONSTANT,
                n_unique=n_unique, null_rate=null_rate,
                notes=["分散=0の定数列"]
            )

        # 周期変数（ユーザー指定）
        if name in self.periodic_cols:
            return ColumnInfo(
                name=name, col_type=ColumnType.PERIODIC,
                n_unique=n_unique, null_rate=null_rate,
                notes=["ユーザー指定の周期変数"]
            )

        # datetime型
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnInfo(
                name=name, col_type=ColumnType.DATETIME,
                n_unique=n_unique, null_rate=null_rate,
            )

        # 数値型
        if pd.api.types.is_numeric_dtype(series):
            return self._classify_numeric(series_clean, name, n_unique, null_rate)

        # 文字列型
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # SMILES判定
            if self._looks_like_smiles(series_clean, name):
                return ColumnInfo(
                    name=name, col_type=ColumnType.SMILES,
                    n_unique=n_unique, null_rate=null_rate,
                    notes=["SMILES列として判定"]
                )

            # datetime文字列 → 変換を試みる
            if self._looks_like_datetime(series_clean):
                notes.append("日時文字列として判定")
                return ColumnInfo(
                    name=name, col_type=ColumnType.DATETIME,
                    n_unique=n_unique, null_rate=null_rate, notes=notes
                )

            # 長テキスト（平均長が閾値超）
            avg_len = series_clean.astype(str).str.len().mean()
            if avg_len > 50:
                return ColumnInfo(
                    name=name, col_type=ColumnType.TEXT,
                    n_unique=n_unique, null_rate=null_rate,
                    notes=[f"平均文字列長={avg_len:.0f}"]
                )

            # カテゴリ判定
            return self._classify_categorical(series_clean, name, n_unique, null_rate)

        # カテゴリdtype
        if pd.api.types.is_categorical_dtype(series):
            return self._classify_categorical(series_clean, name, n_unique, null_rate)

        # 不明
        notes.append("dtype不明、カテゴリとして扱う")
        return ColumnInfo(
            name=name, col_type=ColumnType.CATEGORY_LOW,
            n_unique=n_unique, null_rate=null_rate, notes=notes
        )

    def _classify_numeric(
        self,
        series: pd.Series,
        name: str,
        n_unique: int,
        null_rate: float,
    ) -> ColumnInfo:
        """数値列の詳細分類。"""
        notes: list[str] = []

        # バイナリ（整数で0/1のみ）
        if n_unique == 2:
            unique_vals = set(series.unique())
            if unique_vals <= {0, 1, True, False}:
                return ColumnInfo(
                    name=name, col_type=ColumnType.BINARY,
                    n_unique=n_unique, null_rate=null_rate,
                    notes=["数値バイナリ (0/1)"]
                )

        # 外れ値チェック（IQR法）
        q25, q75 = np.percentile(series, [25, 75])
        iqr = q75 - q25
        if iqr > 0:
            lower = q25 - self.outlier_iqr_factor * iqr
            upper = q75 + self.outlier_iqr_factor * iqr
            outlier_ratio = ((series < lower) | (series > upper)).mean()
            has_outliers = outlier_ratio > 0.05
        else:
            has_outliers = False
            outlier_ratio = 0.0

        # 歪度
        from scipy import stats
        skewness = float(stats.skew(series))

        # 右裾重い & 全正 → 対数変換候補
        if skewness >= self.skewness_threshold and (series > 0).all():
            col_type = ColumnType.NUMERIC_LOG
            notes.append(f"右裾重い（歪度={skewness:.2f}）、全正値 → LogTransformer推奨")
        elif abs(skewness) >= self.skewness_threshold:
            # 冪乗変換（Yeo-Johnson）候補
            col_type = ColumnType.NUMERIC_POWER
            notes.append(f"歪度={skewness:.2f} → PowerTransformer(Yeo-Johnson)推奨")
        elif has_outliers:
            col_type = ColumnType.NUMERIC_OUTLIER
            notes.append(f"外れ値率={outlier_ratio:.1%} → RobustScaler推奨")
        else:
            col_type = ColumnType.NUMERIC_NORMAL
            notes.append(f"歪度={skewness:.2f}（正規分布に近い）")

        return ColumnInfo(
            name=name, col_type=col_type,
            n_unique=n_unique, null_rate=null_rate,
            skewness=skewness, has_outliers=has_outliers, notes=notes,
        )

    def _classify_categorical(
        self,
        series: pd.Series,
        name: str,
        n_unique: int,
        null_rate: float,
    ) -> ColumnInfo:
        """文字列/カテゴリ列の分類。"""
        notes: list[str] = []

        if n_unique == 2:
            return ColumnInfo(
                name=name, col_type=ColumnType.BINARY,
                n_unique=n_unique, null_rate=null_rate,
                notes=["文字列バイナリ"]
            )
        if n_unique <= self.cardinality_threshold:
            notes.append(f"ユニーク数={n_unique} ≤ {self.cardinality_threshold} → OneHotEncoder推奨")
            return ColumnInfo(
                name=name, col_type=ColumnType.CATEGORY_LOW,
                n_unique=n_unique, null_rate=null_rate, notes=notes,
            )
        notes.append(
            f"ユニーク数={n_unique} > {self.cardinality_threshold} → TargetEncoder推奨"
        )
        return ColumnInfo(
            name=name, col_type=ColumnType.CATEGORY_HIGH,
            n_unique=n_unique, null_rate=null_rate, notes=notes,
        )

    def _looks_like_smiles(self, series: pd.Series, col_name: str) -> bool:
        """
        列名のヒントと値のパターンでSMILESかどうかを判定する。
        RDKitが利用可能な場合は、SMILESパーサーで直接判定を試みる。
        利用不可の場合はヒューリスティックな判定（SMILES特有の文字）で判定する。
        """
        # 1. 列名ヒントによる判定
        col_lower = col_name.lower()
        name_match = any(hint in col_lower for hint in self.smiles_col_hints)
        if name_match:
            return True

        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False

        # 2. RDKitを利用した厳密なSMILES解釈
        from backend.chem.rdkit_adapter import _rdkit
        if _rdkit:
            from rdkit import Chem
            def _is_valid_smiles_rdkit(s: str) -> bool:
                # 極端に短い、または単一文字構成のもの（カテゴリ列「A, B, C」など）を除外
                if len(s) < 3 and s.isalpha():
                    return False
                # ログ出力を抑制してパースを試みる
                _rdkit.RDLogger.DisableLog('rdApp.*')
                try:
                    mol = Chem.MolFromSmiles(s)
                    return mol is not None
                finally:
                    _rdkit.RDLogger.EnableLog('rdApp.*')
            
            frac_smiles_rdkit = sample.apply(_is_valid_smiles_rdkit).mean()
            return frac_smiles_rdkit > 0.6

        # 3. フォールバック判定 (RDKitがない場合)
        # 化学元素記号の存在確認 (SMILESを構成する主要な原子)
        atom_symbols = set("BNOFPSClBrI") | set("bcnosp")
        # 特殊記号や結合、リング、枝分かれ、アイソトープ等のSMILES記号
        smiles_chars = set("()=#@+-[]\\/%1234567890") | atom_symbols
        
        def _is_smiles_like_heuristic(s: str) -> bool:
            # 極端に短いまたは長すぎるものは除外 (ID列などの誤爆防止)
            if not (1 <= len(s) <= 2000):
                return False
            # SMILES構成文字以外が含まれているかチェック
            if any(c not in smiles_chars for c in s):
                return False
            # 少なくとも1つは主要な化学元素記号を含む
            has_atom = any(c in atom_symbols for c in s)
            return has_atom

        frac_smiles_heuristic = sample.apply(_is_smiles_like_heuristic).mean()
        return frac_smiles_heuristic > 0.8


    @staticmethod
    def _looks_like_datetime(series: pd.Series) -> bool:
        """文字列がdatetimeに変換できるか判定する。"""
        sample = series.dropna().astype(str).head(10)
        if sample.empty:
            return False
        try:
            pd.to_datetime(sample)
            return True
        except Exception:
            return False
