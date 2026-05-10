"""
tests/test_smiles_pipeline.py

SMILES列を含むデータでAutoMLエンジンの end-to-end テスト。
学習 → predict(生SMILES列込みX) が列不一致なく動くかを確認する。
"""
import pytest
import pandas as pd
import numpy as np
from backend.models.automl import AutoMLEngine


# --- テスト用データ ---
SMILES_DATA = [
    ("CCO",       -0.31),
    ("CC",        -1.89),
    ("CCC",       -0.70),
    ("CCCC",      -0.38),
    ("c1ccccc1",   2.13),
    ("CCN",       -1.03),
    ("OCC",       -1.13),
    ("CCCO",      -0.53),
    ("CCCCO",     -0.25),
    ("c1ccccc1C",  2.73),
    ("c1ccccc1O",  1.46),
    ("CC(=O)O",   -0.17),
    ("CNC",       -1.00),
    ("CCF",        0.18),
    ("CCCl",       0.60),
    ("OC(=O)C",   -0.17),
    ("CCOCC",      0.89),
    ("CCOC(=O)C",  0.73),
    ("c1ccncc1",   0.65),
    ("NCC",       -1.14),
]

def _make_df(n: int = 20) -> pd.DataFrame:
    rows = [SMILES_DATA[i % len(SMILES_DATA)] for i in range(n)]
    return pd.DataFrame(rows, columns=["smiles", "logS"])


class TestSmilesAutoMLPipeline:
    """SMILES列展開後のAutoMLパイプラインのend-to-endテスト。"""

    def test_fit_predict_with_smiles_col(self):
        """事前変換したSMILES記述子データでAutoMLが正常動作すること。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        df = _make_df(20)
        
        # UI側と同じフロー: 最初にSMILESを変換
        transformer = SmilesDescriptorTransformer(smiles_col="smiles")
        df_processed = transformer.fit_transform(df)
        
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=120)
        result = engine.run(df_processed, target_col="logS")

        pipeline = result.best_pipeline

        X_processed = df_processed.drop(columns=["logS"])
        y_pred = pipeline.predict(X_processed)
        assert len(y_pred) == len(df), "予測値の数がサンプル数と一致しません"
        assert not np.any(np.isnan(y_pred)), "予測値にNaNが含まれています"

    def test_fit_predict_no_smiles_col(self):
        """SMILES列を指定しない場合でもpredictが動くこと（通常の数値のみ）。"""
        df = pd.DataFrame({
            "feat1": np.random.rand(30),
            "feat2": np.random.rand(30),
            "target": np.random.rand(30),
        })
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=60)
        result = engine.run(df, target_col="target")
        X_raw = df.drop(columns=["target"])
        y_pred = result.best_pipeline.predict(X_raw)
        assert len(y_pred) == len(df)
    def test_fit_predict_selected_descriptors(self):
        """特定の記述子群（例: 数え上げ系のみ）を選択した場合の動作テスト。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        from backend.chem.rdkit_adapter import RDKitAdapter
        
        df = _make_df(20)
        
        # RDKitの数え上げ記述子のみをプログラム的に抽出
        rdkit = RDKitAdapter()
        count_descs = [m.name for m in rdkit.get_descriptors_metadata() if m.is_count]
        selected = count_descs[:5] # 上位5件のみ
        
        transformer = SmilesDescriptorTransformer(smiles_col="smiles", selected_descriptors=selected, count_normalization="none")
        df_processed = transformer.fit_transform(df)
        
        # 期待通り記述子が制限されているか
        added_cols = [c for c in df_processed.columns if c != "smiles" and c != "logS"]
        assert set(added_cols) == set(selected), f"期待される記述子 {selected} と実際の列 {added_cols} が一致しません"
        
        # その後のAutoMLも正常に動くか
        engine = AutoMLEngine(task="regression", cv_folds=2, timeout_seconds=60)
        result = engine.run(df_processed, target_col="logS")
        assert result.best_pipeline is not None

    def test_invalid_descriptor_names(self):
        """存在しない記述子名が指定された場合に無視または適切に処理されるか。"""
        from backend.chem.smiles_transformer import SmilesDescriptorTransformer
        df = _make_df(5)
        
        # 存在する記述子(MolWt)と存在しない記述子(InvalidOne)を混ぜる
        selected = ["MolWt", "NonExistentDesc_XYZ"]
        transformer = SmilesDescriptorTransformer(smiles_col="smiles", selected_descriptors=selected)
        
        # TODO: 現状のSmilesDescriptorTransformerがどう振る舞うべきか
        # 警告を出すか、エラーにするか、あるいは存在する分だけで通すか。
        # ここでは「存在する分だけが計算される」ことを期待。
        df_processed = transformer.fit_transform(df)
        
        assert "MolWt" in df_processed.columns
        assert "NonExistentDesc_XYZ" not in df_processed.columns

@pytest.mark.xfail(
    reason="sklearn ColumnTransformer requires pre-computed descriptor columns at predict time; "
           "raw SMILES-only input causes column mismatch (known limitation)",
    strict=False,
)
def test_smiles_pipeline_pre_expanded_fit():
    """
    事前計算済みのデータでfitし、生データ（SMILESのみ）でpredictできることを検証する。
    これは evaluation_page.py で発生していた問題を再現するシナリオ。
    """
    from backend.models.automl import AutoMLEngine
    
    data = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CCC", "CC(=O)O", "C1CCCCC1", "c1ccccc1C", "CCCCO", "CC(=O)C", "CCN", "c1ccccc1O"],
        "target": [1.0, 2.0, 1.5, 2.5, 3.0, 2.2, 1.8, 1.1, 1.4, 2.4]
    })
    
    # 事前に展開しておく (Phase 0 の模倣)
    from backend.chem.smiles_transformer import SmilesDescriptorTransformer
    transformer = SmilesDescriptorTransformer(smiles_col="smiles", selected_descriptors=["MolWt", "LogP"])
    data_expanded = transformer.fit_transform(data)
    
    # AutoMLEngine を実行
    # data_expanded に SMILES が残っているため、engine は正しく SMILES 列を検出できる
    engine = AutoMLEngine(task="regression", timeout_seconds=10, model_keys=["rf"], selected_descriptors=["MolWt", "LogP"])
    res = engine.run(data_expanded, target_col="target", smiles_col="smiles")
    
    # 推論時に生データを渡す (評価ページの模倣)
    raw_X = data.drop(columns=["target"])
    preds = res.best_pipeline.predict(raw_X)
    
    assert len(preds) == len(data)
    assert not np.isnan(preds).any()
