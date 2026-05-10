import os
import sys
import logging
import pandas as pd
import traceback
from backend.models.automl import AutoMLEngine
import warnings

warnings.filterwarnings("ignore")

# 詳細なデバッグログの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_pipeline")

def run_debug():
    logger.info("Starting pipeline debug...")
    
    logger.debug("Generating mock data")
    smiles_list = [
        "CC", "CCO", "c1ccccc1", "C(=O)O",
        "CC(=O)C", "C1=CC=CC=C1O", "C1=CN=CC=N1", "CC1=CC=CC=C1",
        "CC(C)O", "CCN", "C1CCCCC1"
    ]
    df = pd.DataFrame({
        "SMILES": smiles_list,
        "solubility_logS": [-1.0, 0.5, -2.1, 1.2, 0.8, -0.5, 0.3, -1.5, 0.1, -0.8, 1.5]
    })
    logger.debug(f"Loaded {len(df)} rows.")

    # プリセット（基本物性）の模倣
    selected_descriptors = {
        "rdkit": ["ExactMolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds"],
        "hsp": ["hsp_d", "hsp_p", "hsp_h"],
        "mordred": ["FilterItLogS", "McGowan_Volume", "TopoPSA", "Lipinski"],
        "dl": []
    }

    try:
        # 2. パイプライン初期化
        logger.info("Initializing AutoMLPipeline...")
        automl = AutoMLEngine(
            task="regression",
            cv_folds=2,
            selected_descriptors=selected_descriptors,
            active_engines=["RDKitAdapter", "MordredAdapter", "GroupContribAdapter"]
        )
        automl.count_normalization = "density"
        
        # 3. 実行
        logger.info("Running automl.run()...")
        results = automl.run(df, target_col="solubility_logS", smiles_col="SMILES")
        
        logger.info("Pipeline finished successfully!")
        print("Performance:", results.get("performance"))
    except Exception as e:
        logger.error("Pipeline crashed!", exc_info=True)

if __name__ == "__main__":
    run_debug()
