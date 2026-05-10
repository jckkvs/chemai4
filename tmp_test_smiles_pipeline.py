"""
SMILESパイプラインのエンドツーエンドテストスクリプト
"""
import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from backend.models.automl import AutoMLEngine
from backend.data.preprocessor import PreprocessConfig

# テストデータ作成（SMILESを含む）
np.random.seed(42)
_DUMMY_SMILES = [
    'C', 'CC', 'CCC', 'CCO', 'CCN', 'c1ccccc1', 'c1ccccc1O', 'CC(=O)O', 'CC(C)C', 'C1CCCCC1',
    'c1ccncc1', 'c1ncncn1', 'C1COCCO1', 'CO', 'CN', 'c1ccc(O)cc1', 'c1ccc(N)cc1', 'CCC(=O)O',
    'CCCO', 'CCCN', 'c1ccccc1Cl', 'c1ccccc1Br', 'c1ccc(F)cc1', 'CC(=O)N', 'CC(=O)O'
]
n = 25
df_test = pd.DataFrame({
    'SMILES': np.random.choice(_DUMMY_SMILES, n),
    'solubility': np.random.randn(n) * 2 - 2
})

print(f'テストDF shape: {df_test.shape}')
print(f'SMILES列サンプル: {df_test["SMILES"].tolist()[:5]}')

engine = AutoMLEngine(
    task='regression',
    cv_folds=3,  # 少量データなのでfolds=3
    model_keys=['lr', 'rf'],  # 高速モデルのみ
    timeout_seconds=60,
    selected_descriptors=None,
)

cfg = PreprocessConfig(numeric_scaler='standard')
print('AutoMLEngine.run()を開始...')
result = engine.run(df_test, target_col='solubility', smiles_col='SMILES', preprocess_config=cfg)
print(f'automl完了！最良モデル={result.best_model_key}, スコア={result.best_score:.4f}')
print(f'smiles_transformer: {result.smiles_transformer is not None}')
print(f'smiles_correlations件数: {len(result.smiles_correlations)}')
print(f'processed_X shape: {result.processed_X.shape if result.processed_X is not None else None}')
print()
print('=== 全テスト合格 ===')
