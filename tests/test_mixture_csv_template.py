"""
tests/test_mixture_csv_template.py

mixture_csv_template のユニットテスト。
"""
import io

import pandas as pd
import pytest

from backend.chem.mixture_csv_template import (
    TEMPLATE_COLUMNS,
    ParsedMixture,
    generate_template_csv,
    generate_template_dataframe,
    parse_mixture_csv,
)


# ── テンプレート生成 ──

def test_generate_csv_bytes():
    csv_bytes = generate_template_csv()
    assert isinstance(csv_bytes, bytes)
    text = csv_bytes.decode("utf-8")
    assert "session_id" in text
    assert "MIX_001" in text


def test_generate_csv_no_samples():
    csv_bytes = generate_template_csv(include_samples=False)
    text = csv_bytes.decode("utf-8")
    assert "session_id" in text
    assert "MIX_001" not in text


def test_generate_dataframe():
    df = generate_template_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6  # 6 sample rows
    assert "smiles" in df.columns
    assert "ratio_unit" in df.columns


def test_template_columns():
    assert "session_id" in TEMPLATE_COLUMNS
    assert "smiles" in TEMPLATE_COLUMNS
    assert "ratio_value" in TEMPLATE_COLUMNS


# ── CSVパース ──

def test_parse_sample_csv():
    csv_bytes = generate_template_csv()
    mixtures = parse_mixture_csv(csv_bytes)
    assert len(mixtures) == 3  # MIX_001, MIX_002, MIX_003
    assert all(isinstance(m, ParsedMixture) for m in mixtures)


def test_parse_mixture_components():
    csv_bytes = generate_template_csv()
    mixtures = parse_mixture_csv(csv_bytes)
    mix1 = next(m for m in mixtures if m.session_id == "MIX_001")
    assert len(mix1.components) == 2
    assert mix1.ratio_unit == "weight"
    assert mix1.components[0]["smiles"] == "CCO"


def test_parse_mole_ratio():
    csv_bytes = generate_template_csv()
    mixtures = parse_mixture_csv(csv_bytes)
    mix2 = next(m for m in mixtures if m.session_id == "MIX_002")
    assert mix2.ratio_unit == "mole"


def test_parse_other_ratio():
    csv_bytes = generate_template_csv()
    mixtures = parse_mixture_csv(csv_bytes)
    mix3 = next(m for m in mixtures if m.session_id == "MIX_003")
    assert mix3.ratio_unit == "other"
    assert "volume" in mix3.other_ratio_unit.lower()


def test_parse_string_input():
    csv_str = """session_id,component_order,smiles,compound_name,ratio_value,ratio_unit,other_ratio_unit
MIX_T,1,CCO,ethanol,50,weight,
MIX_T,2,O,water,50,weight,"""
    mixtures = parse_mixture_csv(csv_str)
    assert len(mixtures) == 1
    assert len(mixtures[0].components) == 2


def test_parse_missing_column_error():
    csv_str = "session_id,smiles\nMIX,CCO"
    with pytest.raises(ValueError, match="必須カラムが不足"):
        parse_mixture_csv(csv_str)


def test_parse_empty_rows_filtered():
    csv_str = """session_id,component_order,smiles,compound_name,ratio_value,ratio_unit,other_ratio_unit
MIX_T,1,CCO,ethanol,50,weight,
,,,,,,
MIX_T,2,O,water,50,weight,
,,,,,,"""
    mixtures = parse_mixture_csv(csv_str)
    assert len(mixtures) == 1
    assert len(mixtures[0].components) == 2


def test_parse_comments_ignored():
    csv_str = """# comment line
# another comment
session_id,component_order,smiles,compound_name,ratio_value,ratio_unit,other_ratio_unit
MIX_T,1,CCO,eth,50,weight,
MIX_T,2,O,water,50,weight,"""
    mixtures = parse_mixture_csv(csv_str)
    assert len(mixtures) == 1


def test_parse_component_order_sorting():
    csv_str = """session_id,component_order,smiles,compound_name,ratio_value,ratio_unit,other_ratio_unit
MIX_T,2,O,water,50,weight,
MIX_T,1,CCO,ethanol,50,weight,"""
    mixtures = parse_mixture_csv(csv_str)
    # component_order でソートされる
    assert mixtures[0].components[0]["smiles"] == "CCO"
    assert mixtures[0].components[1]["smiles"] == "O"


def test_parse_mixed_ratio_warning():
    csv_str = """session_id,component_order,smiles,compound_name,ratio_value,ratio_unit,other_ratio_unit
MIX_T,1,CCO,ethanol,50,weight,
MIX_T,2,O,water,50,mole,"""
    mixtures = parse_mixture_csv(csv_str)
    assert len(mixtures[0].warnings) > 0
    assert "混在" in mixtures[0].warnings[0]


def test_roundtrip_generate_parse():
    """生成→パースのラウンドトリップ。"""
    csv_bytes = generate_template_csv()
    mixtures = parse_mixture_csv(csv_bytes)
    assert len(mixtures) == 3

    # 再度生成してパースしても同じ結果
    csv_bytes2 = generate_template_csv()
    mixtures2 = parse_mixture_csv(csv_bytes2)
    assert len(mixtures2) == 3
    for m1, m2 in zip(mixtures, mixtures2):
        assert m1.session_id == m2.session_id
        assert len(m1.components) == len(m2.components)
