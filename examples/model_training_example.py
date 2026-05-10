"""
examples/model_training_example.py

モデルファクトリー (backend.models.factory) を使用した機械学習モデルの訓練と評価の例。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from backend.models.factory import get_model, list_models

def run_example():
    # 1. 回帰タスクのテストデータの生成
    print("--- テストデータの生成 ---")
    np.random.seed(42)
    X = np.random.rand(200, 5)
    # y = 3*x1 + 2*x2 - 5*x3 + noise
    y = 3 * X[:, 0] + 2 * X[:, 1] - 5 * X[:, 2] + np.random.randn(200) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. 利用可能な回帰モデルの表示
    print("\n--- 利用可能な回帰モデル ---")
    models = list_models(task="regression")
    for m in models[:5]: # 最初の5つのみ表示
        print(f"  Key: {m['key']}, Name: {m['name']}, Tags: {m['tags']}")

    # 3. Random Forest モデルの生成と訓練
    print("\n--- Random Forest モデルの訓練 ---")
    rf_model = get_model("rf", task="regression", n_estimators=50)
    rf_model.fit(X_train, y_train)
    
    # 4. 予測と評価
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 5. Gradient Boosting モデルの生成と訓練 (パラメータ上書きの例)
    print("\n--- Gradient Boosting モデルの訓練 (learning_rate=0.05) ---")
    gbm_model = get_model("gbm", task="regression", learning_rate=0.05, n_estimators=100)
    gbm_model.fit(X_train, y_train)
    
    y_pred_gbm = gbm_model.predict(X_test)
    print(f"R2 Score (GBM): {r2_score(y_test, y_pred_gbm):.4f}")

if __name__ == "__main__":
    run_example()
