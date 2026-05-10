"""
backend/chem/mixture_csv_template.py

混合物入力用CSVテンプレートの生成・パースモジュール。

テンプレートダウンロード → ユーザー記入 → アップロード → パース
のワークフローを支援する。

形式: Wide形式（1行 = 1混合物）
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Wide形式テンプレートのカラム定義
# 1行 = 1混合物（最大5成分まで）
WIDE_TEMPLATE_COLUMNS = [
    "Sample_ID",
    "Target_Property",
]
# 動的に Compound_N_* カラムを追加
MAX_COMPOUNDS_DEFAULT = 5
def _build_wide_columns(max_compounds: int = MAX_COMPOUNDS_DEFAULT) -> list[str]:
    cols = list(WIDE_TEMPLATE_COLUMNS)
    for i in range(1, max_compounds + 1):
        cols.extend([
            f"Compound_{i}_SMILES",
            f"Compound_{i}_Name",
            f"Compound_{i}_WT%",
        ])
    cols.append("Notes")
    return cols


# 後方互換性のためのlong形式（廃止予定）
TEMPLATE_COLUMNS = [
    "session_id",
    "component_order",
    "smiles",
    "compound_name",
    "ratio_value",
    "ratio_unit",
    "other_ratio_unit",
]

# サンプルデータ（wide形式）
_SAMPLE_WIDE_ROWS = [
    {
        "Sample_ID": "MIX_001",
        "Target_Property": 3.982,
        "Compound_1_SMILES": "CC(C)O",
        "Compound_1_Name": "Isopropanol",
        "Compound_1_WT%": 31.64,
        "Compound_2_SMILES": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Compound_2_Name": "Ibuprofen",
        "Compound_2_WT%": 42.56,
        "Compound_3_SMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Compound_3_Name": "Caffeine",
        "Compound_3_WT%": 25.8,
        "Notes": "Sample mixture 1",
    },
    {
        "Sample_ID": "MIX_002",
        "Target_Property": -1.698,
        "Compound_1_SMILES": "CCO",
        "Compound_1_Name": "Ethanol",
        "Compound_1_WT%": 41.07,
        "Compound_2_SMILES": "CCO",
        "Compound_2_Name": "Ethanol",
        "Compound_2_WT%": 43.58,
        "Compound_3_SMILES": "c1ccccc1",
        "Compound_3_Name": "Benzene",
        "Compound_3_WT%": 15.36,
        "Notes": "Sample mixture 2",
    },
]

def _write_wide_samples(writer: csv.DictWriter, max_compounds: int = MAX_COMPOUNDS_DEFAULT) -> None:
    """wide形式でサンプルデータを書き込む。"""
    for row in _SAMPLE_WIDE_ROWS:
        writer.writerow(row)


@dataclass
class ParsedMixture:
    """パース済みの1混合物。"""
    session_id: str
    components: list[dict[str, Any]]
    ratio_unit: str
    target_property: float | None = None
    other_ratio_unit: str = ""
    warnings: list[str] = field(default_factory=list)


def generate_template_csv(
    include_samples: bool = True,
    max_compounds: int = MAX_COMPOUNDS_DEFAULT,
    n_empty_rows: int = 4,
) -> bytes:
    """
    混合物入力用CSVテンプレートを生成する（wide形式）。

    Args:
        include_samples: True のときサンプルデータを含める。
        max_compounds: 1混合物あたりの最大成分数。
        n_empty_rows: 空行の追加数。

    Returns:
        BOM付きUTF-8のCSVバイト列（Excel互換）。
    """
    headers = _build_wide_columns(max_compounds)
    buf = io.StringIO()

    # ヘッダー行前のコメント
    buf.write("# ChemAI2 混合物入力テンプレート v2.0 (Wide形式)\n")
    buf.write("# 1行 = 1混合物\n")
    buf.write("# Compound_N_SMILES: 成分NのSMILES\n")
    buf.write("# Compound_N_Name: 成分Nの名称（任意）\n")
    buf.write("# Compound_N_WT%: 成分Nの重量パーセンテージ\n")
    buf.write("# Target_Property: 目的変数値（ML学習用・任意）\n")

    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    if include_samples:
        _write_wide_samples(writer, max_compounds)

    # 空行
    for _ in range(n_empty_rows):
        writer.writerow({c: "" for c in headers})

    csv_str = buf.getvalue()
    # BOM付きUTF-8 (Excel互換)
    return ("﻿" + csv_str).encode("utf-8")


def generate_template_dataframe(include_samples: bool = True) -> pd.DataFrame:
    """テンプレートをDataFrameとして返す（wide形式）。"""
    if include_samples:
        return pd.DataFrame(_SAMPLE_WIDE_ROWS)
    return pd.DataFrame(columns=_build_wide_columns())


def parse_mixture_csv(
    csv_content: str | bytes | io.IOBase | Path,
) -> list[ParsedMixture]:
    """
    混合物CSVファイルをパースする。
    自動判定: wide形式（Compound_*）またはlong形式（session_id）のいずれか。

    Args:
        csv_content: CSV内容（文字列、バイト列、ファイルオブジェクト、パス）。

    Returns:
        ParsedMixture のリスト。

    Raises:
        ValueError: 必須カラムが欠落している場合。
    """
    # 入力形式の統一
    if isinstance(csv_content, (str, bytes)):
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8-sig")
        # コメント行を除外
        lines = [
            line for line in csv_content.splitlines()
            if not line.strip().startswith("#") and line.strip()
        ]
        csv_str = "\n".join(lines)
        df = pd.read_csv(io.StringIO(csv_str))
    elif isinstance(csv_content, Path):
        text = csv_content.read_text(encoding="utf-8-sig")
        return parse_mixture_csv(text)
    else:
        df = pd.read_csv(csv_content, comment="#")

    # 自動判定: wide形式かlong形式か
    if any("Compound_" in c for c in df.columns):
        return _parse_wide_format(df)
    elif "session_id" in df.columns:
        return _parse_long_format(df)
    else:
        raise ValueError("不明なCSV形式: 'session_id' または 'Compound_' カラムが必要です")


def _parse_wide_format(df: pd.DataFrame) -> list[ParsedMixture]:
    """wide形式（1行=1混合物）をパースする。"""
    mixtures: list[ParsedMixture] = []

    # Compound_N_* カラムから最大成分数を判定
    max_n = 0
    for col in df.columns:
        import re
        m = re.match(r"Compound_(\d+)_SMILES$", col)
        if m:
            n = int(m.group(1))
            max_n = max(max_n, n)

    for idx, row in df.iterrows():
        warnings: list[str] = []
        components: list[dict[str, Any]] = []
        sample_id = str(row.get("Sample_ID", f"MIX_{idx+1}")).strip()

        # Target_Property
        target_property = None
        if "Target_Property" in df.columns and pd.notna(row["Target_Property"]):
            try:
                target_property = float(row["Target_Property"])
            except (ValueError, TypeError):
                pass

        # 成分を抽出
        for i in range(1, max_n + 1):
            smiles_col = f"Compound_{i}_SMILES"
            if smiles_col not in df.columns:
                break
            smiles = str(row.get(smiles_col, "")).strip()
            if not smiles or smiles == "nan":
                continue
            name_col = f"Compound_{i}_Name"
            wt_col = f"Compound_{i}_WT%"

            comp = {
                "smiles": smiles,
                "compound_name": str(row.get(name_col, "")).strip() or None,
                "ratio_value": float(row.get(wt_col, 0)),
                "ratio_unit": "weight",
            }
            components.append(comp)

        if len(components) < 2:
            warnings.append(f"混合物 '{sample_id}' の成分数が2未満: {len(components)}")
            continue

        # WT% の合計チェック
        total_wt = sum(c["ratio_value"] for c in components)
        if total_wt <= 0:
            warnings.append(f"混合物 '{sample_id}' の重量合計が不正: {total_wt}")

        mixtures.append(ParsedMixture(
            session_id=sample_id,
            components=components,
            ratio_unit="weight",
            target_property=target_property,
            warnings=warnings,
        ))

    logger.info(
        "Wide形式CSV パース完了: %d混合物, %d成分",
        len(mixtures),
        sum(len(m.components) for m in mixtures),
    )
    return mixtures


def _parse_long_format(df: pd.DataFrame) -> list[ParsedMixture]:
    """long形式（session_id単位）をパースする。"""
    # 空行を除去
    df = df.dropna(subset=["session_id", "smiles"]).copy()
    df["session_id"] = df["session_id"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].str.len() > 0]

    # session_id でグループ化
    mixtures: list[ParsedMixture] = []

    for sid, group in df.groupby("session_id", sort=False):
        warnings: list[str] = []
        components: list[dict[str, Any]] = []

        ratio_units = group["ratio_unit"].dropna().unique()
        if len(ratio_units) > 1:
            warnings.append(
                f"session '{sid}' 内で比率タイプが混在: {list(ratio_units)}"
            )
        ratio_unit = str(ratio_units[0]) if len(ratio_units) > 0 else "weight"

        for _, row in group.iterrows():
            comp = {
                "smiles": str(row["smiles"]).strip(),
                "ratio_value": float(row.get("ratio_value", 1.0)),
                "ratio_unit": ratio_unit,
                "compound_name": str(row.get("compound_name", "")).strip() or None,
                "component_order": int(row.get("component_order", 0)),
            }
            components.append(comp)

        if len(components) < 2:
            warnings.append(f"session '{sid}' の成分数が2未満: {len(components)}")

        # component_order でソート
        components.sort(key=lambda c: c.get("component_order", 0))

        other_unit = ""
        if ratio_unit == "other":
            ou = group["other_ratio_unit"].dropna()
            if len(ou) > 0:
                other_unit = str(ou.iloc[0]).strip()

        mixtures.append(ParsedMixture(
            session_id=str(sid),
            components=components,
            ratio_unit=ratio_unit,
            other_ratio_unit=other_unit,
            warnings=warnings,
        ))

    logger.info(
        "Long形式CSV パース完了: %d混合物, %d成分",
        len(mixtures),
        sum(len(m.components) for m in mixtures),
    )
    return mixtures
