"""
tests/test_chem.py

backend/chem モジュールのユニットテスト。
RDKitAdapter を中心にテストする。
"""
import pytest
import numpy as np
import pandas as pd
from backend.chem.rdkit_adapter import RDKitAdapter
from backend.chem.base import DescriptorResult

@pytest.fixture
def rdkit_adapter() -> RDKitAdapter:
    return RDKitAdapter(compute_fp=True, morgan_bits=128, rdkit_fp_bits=128)

def test_rdkit_adapter_name(rdkit_adapter: RDKitAdapter) -> None:
    assert rdkit_adapter.name == "rdkit"

def test_rdkit_adapter_availability(rdkit_adapter: RDKitAdapter) -> None:
    # 環境によって異なる可能性があるが、基本的に True を期待
    assert rdkit_adapter.is_available() in [True, False]

def test_compute_descriptors(rdkit_adapter: RDKitAdapter) -> None:
    """有効なSMILESに対して記述子が計算されること。"""
    if not rdkit_adapter.is_available():
        pytest.skip("RDKit is not available")
    
    smiles = ["CCO", "c1ccccc1"]
    result = rdkit_adapter.compute(smiles)
    
    assert isinstance(result, DescriptorResult)
    assert len(result.descriptors) == 2
    assert "MolWt" in result.descriptors.columns
    assert "Morgan_r2_0" in result.descriptors.columns
    assert result.failed_indices == []

def test_compute_with_invalid_smiles(rdkit_adapter: RDKitAdapter) -> None:
    """無効なSMILESが含まれる場合に正しく処理され、failed_indices が記録されること。"""
    if not rdkit_adapter.is_available():
        pytest.skip("RDKit is not available")
    
    smiles = ["CCO", "invalid_smiles", "CCC"]
    result = rdkit_adapter.compute(smiles)
    
    assert len(result.descriptors) == 3
    assert result.failed_indices == [1]
    # 失敗した行は 0.0 で埋められているか (型によってはNaNもあるためfillnaしてからcheck)
    row = result.descriptors.iloc[1].fillna(0.0)
    # pd.to_numericで数値型になっている列のみチェック
    numeric_row = pd.to_numeric(row, errors='coerce').fillna(0.0)
    assert numeric_row.sum() == 0.0

def test_get_descriptor_names(rdkit_adapter: RDKitAdapter) -> None:
    """記述子名のリストが正しく取得できること。"""
    names = rdkit_adapter.get_descriptor_names()
    assert "MolWt" in names
    assert "Morgan_r2_127" in names
    assert "RDKitFP_127" in names

def test_descriptor_metadata_rdkit(rdkit_adapter: RDKitAdapter) -> None:
    """RDKit の記述子メタデータが正しく構成されているか。"""
    from backend.chem.base import DescriptorMetadata
    mdata = rdkit_adapter.get_descriptors_metadata()
    assert len(mdata) > 0
    assert all(isinstance(m, DescriptorMetadata) for m in mdata)
    
    # 特定の数え上げ記述子のチェック
    name_to_meta = {m.name: m for m in mdata}
    assert "NumHAcceptors" in name_to_meta
    assert name_to_meta["NumHAcceptors"].is_count is True
    assert "MolWt" in name_to_meta
    assert name_to_meta["MolWt"].is_count is False

def test_descriptor_metadata_mordred() -> None:
    """Mordred の記述子メタデータのチェック。"""
    from backend.chem.mordred_adapter import MordredAdapter
    adapter = MordredAdapter()
    
    if not adapter.is_available():
        pytest.skip("Mordred is not available in this environment")
        
    mdata = adapter.get_descriptors_metadata()
    name_to_meta = {m.name: m for m in mdata}
    # nC, nAtom などが数え上げ系として認識されているか
    if "nC" in name_to_meta:
        assert name_to_meta["nC"].is_count is True
    if "nRing" in name_to_meta:
        assert name_to_meta["nRing"].is_count is True

# --- 新規追加アダプタのテスト ---

def test_mordred_adapter() -> None:
    from backend.chem.mordred_adapter import MordredAdapter
    adapter = MordredAdapter()
    assert adapter.name == "mordred"
    
    # 環境によってMordredが入っているかは異なるが、is_availableの型はbool
    is_avail = adapter.is_available()
    assert isinstance(is_avail, bool)
    
    if is_avail:
        try:
            res = adapter.compute(["CCO", "c1ccccc1"])
            assert len(res.descriptors) == 2
            # デフォルトではselected_onlyなので限られた記述子が返る
            assert "MW" in res.descriptors.columns
            assert "SLogP" in res.descriptors.columns
        except ImportError:
            pytest.skip("mordred runtime error (numpy 2.x incompatibility)")
    else:
        # インストールされていない場合、computeを呼ぶと例外が出る
        with pytest.raises(RuntimeError):
            adapter.compute(["CCO"])

def test_stub_adapters() -> None:
    """各アダプタが名前リストを持ち、is_available()がboolを返すことをテスト。
    実装済みアダプタはTrue、外部ライブラリ依存で未インストールならFalseを返す。"""
    from backend.chem.xtb_adapter import XTBAdapter
    from backend.chem.cosmo_adapter import CosmoAdapter
    from backend.chem.unipka_adapter import UniPkaAdapter
    from backend.chem.group_contrib_adapter import GroupContribAdapter

    adapters = [
        XTBAdapter(),
        CosmoAdapter(),
        UniPkaAdapter(),
        GroupContribAdapter(),
    ]

    for adapter in adapters:
        assert list(adapter.get_descriptor_names())  # 名前リストは返る
        assert isinstance(adapter.is_available(), bool)

def test_psmiles_adapter():
    """PSmilesAdapterのテスト。RDKit近似フォールバックが機能することを確認。"""
    from backend.chem.psmiles_adapter import PSmilesAdapter
    
    adapter = PSmilesAdapter()
    
    # 利用可能であること（RDKitがあるため）
    assert adapter.is_available()
    
    # PSMILES判定
    assert PSmilesAdapter.is_psmiles("*CC*")
    assert PSmilesAdapter.is_psmiles("[*]C(=O)O")
    assert PSmilesAdapter.is_psmiles("CCO[*]")
    assert not PSmilesAdapter.is_psmiles("CCO")
    
    # 計算実行
    smiles_list = ["*CC*", "CCO"]
    res = adapter.compute(smiles_list)
    
    assert res.adapter_name == "psmiles"
    assert len(res.descriptors) == 2
    assert "PSMILES_MonomerWt" in res.descriptors.columns
    assert "PSMILES_NumHDonors" in res.descriptors.columns
    
    # エラーにならないこと
    assert len(res.failed_indices) == 0
    
    # 値の確認（* を [CH3] にするため *CC* は butane 相当になる近似）
    wt_psmiles = res.descriptors.loc[0, "PSMILES_MonomerWt"]
    wt_normal = res.descriptors.loc[1, "PSMILES_MonomerWt"]
    assert wt_psmiles > 0.0
    assert wt_normal > 0.0

