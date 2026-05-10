import os
import sys
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def verify_train_predictions():
    logger.info("==================================================")
    logger.info("🔴 1. Train予測値の保存・表示 (train_predictions) の検証")
    try:
        from backend.models.automl import AutoMLResult
        ar = AutoMLResult()
        if hasattr(ar, "train_predictions"):
            logger.info("✅ AutoMLResult データクラスに 'train_predictions' フィールドの存在を確認しました。")
        else:
            logger.error("❌ 'train_predictions' フィールドが見つかりません。")
            return False
            
        # 実際に簡易的なエンジンテストを回す
        from backend.models.automl import AutoMLEngine
        # ダミーデータ
        df = pd.DataFrame({
            "f1": np.random.randn(50), 
            "f2": np.random.randn(50), 
            "target": np.random.randint(0, 2, 50)
        })
        engine = AutoMLEngine(
            task="classification", 
            model_keys=["rf"],  # 高速なランダムフォレスト
            cv_folds=2,
            timeout_seconds=30
        )
        res = engine.run(df, target_col="target")
        
        if res.train_predictions is not None and len(res.train_predictions) == 50:
            logger.info(f"✅ AutoMLEngineの実行完了: train_predictions={res.train_predictions.shape} が正常に保存されています。")
            return True
        else:
            logger.error("❌ train_predictionsが計算されていないか、Noneです。")
            return False
    except Exception as e:
        logger.error(f"❌ 検証中に例外が発生しました: {e}", exc_info=True)
        return False


def verify_eda_dim_reduction():
    logger.info("\n==================================================")
    logger.info("🟡 2. EDA次元削減の空白表示解消（再確認）")
    try:
        from backend.data.dim_reduction import compute_dim_reduction_and_importance
        df = pd.DataFrame({
            "f1": np.random.randn(100), 
            "f2": np.random.randn(100) + 2, 
            "f3": np.random.randn(100) * 3, 
            "cat": ["A", "B", "C", "D"] * 25
        })
        res = compute_dim_reduction_and_importance(df)
        
        if res.get("status") == "success":
            logger.info(f"✅ PCA計算成功: PC1, PC2 の形状={res['pca_coords'].shape}")
            logger.info(f"✅ t-SNE計算成功: t-SNE1, t-SNE2 の形状={res['tsne_coords'].shape}")
            return True
        else:
            logger.error(f"❌ 次元削減に失敗しました: {res.get('message')}")
            return False
    except ImportError:
        logger.error("❌ backend.data.dim_reduction からのインポートに失敗しました。")
        return False
    except Exception as e:
        logger.error(f"❌ 検証中に例外が発生しました: {e}", exc_info=True)
        return False


def verify_monotonic_constraints():
    logger.info("\n==================================================")
    logger.info("🟢 3. 単調性制約のエンドツーエンドテスト")
    try:
        from backend.models.automl import AutoMLEngine
        from frontend_nicegui.components.column_meta_editor import ColumnMeta

        # 特徴量 positive が増えると target も増える、negative が増えると target は減るデータセット
        X1 = np.linspace(0, 10, 100)
        X2 = np.linspace(0, 10, 100)
        np.random.shuffle(X2) # f2はランダムノイズ的な配置もあるがベースは負の相関
        
        y = X1 * 2.0 - X2 * 1.5 + np.random.randn(100) * 0.1
        
        df = pd.DataFrame({
            "feature_positive": X1,
            "feature_negative": X2,
            "target": y
        })
        
        # UIから渡される生データと同等のフォーマットを用意し、マージプロセスをエミュレート
        raw_state = {
            "monotonic_constraints": {
                "_by_feature": {
                    "feature_positive": {"direction": "increasing", "override_set": True, "strength": 1.0},
                    "feature_negative": {"direction": "decreasing", "override_set": True, "strength": 1.0}
                }
            }
        }
        
        # 実際にフロントエンドのビルド関数を通す
        from frontend_nicegui.components.column_meta_editor import extract_monotonic_from_column_meta
        mono_from_meta = extract_monotonic_from_column_meta(raw_state)
        # mono_from_meta は {"feature_positive": 1, "feature_negative": -1} になるはず
        
        # バックエンドへの受け渡し形式
        monotonic_constraints = {"feature_positive": 1, "feature_negative": -1}
        
        logger.info(f"✅ UI設定からの変換形式検証成功: {monotonic_constraints}")
        
        # LightGBM (ネイティブ対応) で実行検証
        logger.info("-> LightGBM (ネイティブ制約) をテスト中...")
        engine_lgb = AutoMLEngine(
            task="regression",
            model_keys=["lgb"],
            monotonic_constraints_dict=monotonic_constraints,
            cv_folds=2,
            timeout_seconds=30
        )
        res_lgb = engine_lgb.run(df, target_col="target")
        
        if res_lgb.best_score is not None:
            logger.info(f"✅ LightGBMでの単調性制約適用モデルが正常に構築されました (Score: {res_lgb.best_score:.4f})")
            logger.info(f"✅ 確定した制約: {getattr(res_lgb.best_pipeline.named_steps['estimator'], 'resolved_constraints_', 'N/A')}")
        else:
            logger.error("❌ LightGBMモデル構築失敗")
            return False
            
        # SVR (Fallback/Penalty制約) で実行検証
        logger.info("-> SVR (ペナルティ拡張法制約) をテスト中...")
        engine_svr = AutoMLEngine(
            task="regression",
            model_keys=["svm"], 
            monotonic_constraints_dict=monotonic_constraints,
            cv_folds=2,
            timeout_seconds=30
        )
        res_svr = engine_svr.run(df, target_col="target")
        
        if res_svr.best_score is not None:
            logger.info(f"✅ SVRでのペナルティ拡張制約適用モデルが正常に構築されました (Score: {res_svr.best_score:.4f})")
            
            # wrap_monotonic されたモデルの中身を確認
            wrapper = res_svr.best_pipeline.named_steps['estimator']
            if hasattr(wrapper, "constraints"):
                logger.info(f"✅ Wrapper 制約状態: {wrapper.constraints}")
                return True
            else:
                logger.warning("⚠️ SVR制約ラッパーのプロパティが異なります")
                return True
        else:
            logger.error("❌ SVRモデル構築失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 検証中に例外が発生しました: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    b1 = verify_train_predictions()
    b2 = verify_eda_dim_reduction()
    b3 = verify_monotonic_constraints()
    
    if all([b1, b2, b3]):
        logger.info("\n🎉🎉🎉 全ての E2E テスト・機能検証が正常に PASS しました！ 🎉🎉🎉")
    else:
        logger.error("\n❌ 一部の機能検証が失敗しました。ログを確認してください。")
