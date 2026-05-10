
def generate_reproduction_script(
    model_path: str = "model.pkl",
    data_path: str = "dataset.csv",
    target_col: str = "target",
    task: str = "regression",
    smiles_col: str = None
) -> str:
    """
    Generates a standalone Python script that users can run to reproduce predictions
    using their exported ChemAI2 model pipeline.
    """
    script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChemAI2: Reproduction Script
This script loads the exported model pipeline and performs predictions on new data.
"""

import sys
import pandas as pd
import joblib

MODEL_PATH = "{model_path}"
DATA_PATH = "{data_path}"
TARGET_COL = "{target_col}"
TASK_TYPE = "{task}"
SMILES_COL = {"'" + smiles_col + "'" if smiles_col else "None"}

def main():
    print(f"Loading data from {{DATA_PATH}}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file {{DATA_PATH}} not found.")
        sys.exit(1)

    print(f"Loading model pipeline from {{MODEL_PATH}}...")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file {{MODEL_PATH}} not found.")
        sys.exit(1)

    # In production, features are everything except the target
    # If there are specific features dropped, the pipeline preprocessing step will handle them.
    if TARGET_COL in df.columns:
        X = df.drop(columns=[TARGET_COL])
        y_true = df[TARGET_COL]
    else:
        X = df.copy()
        y_true = None

    print(f"Making predictions...")
    if TASK_TYPE == "classification" and hasattr(pipeline, "predict_proba"):
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)
        df["Prediction"] = preds
        for i in range(probs.shape[1]):
            df[f"Probability_Class_{{i}}"] = probs[:, i]
    else:
        preds = pipeline.predict(X)
        df["Prediction"] = preds

    out_file = "predictions_output.csv"
    df.to_csv(out_file, index=False)
    print(f"Predictions saved to {{out_file}}")

if __name__ == "__main__":
    main()
'''
    return script
