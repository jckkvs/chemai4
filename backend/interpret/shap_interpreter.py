"""
backend/interpret/shap_interpreter.py

Pipeline分割によるSHAP高速化と二重スケール（標準化 vs 生データ）での解釈モデル。
"""
import shap
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Dict
import logging

from backend.mlops.data_routing import MultiViewResult

logger = logging.getLogger(__name__)

def calculate_shap_values(
    pipeline: Pipeline,
    X_train_raw: pd.DataFrame,
    background_samples: int = 100,
) -> MultiViewResult:
    """
    SHAP値を標準化空間で正確に計算し、表示用に生データを保持した MultiViewResult を返す。
    
    Returns:
        MultiViewResult
            core_output: np.ndarray (SHAP値)
            views: 
                - compare: 標準化データ(X_scaled) と 共に出力
                - interpret: 生データ(X_raw) と 共に出力
            metadata: base_value, feature_names, X_scaled, X_raw, feature_types などを格納
    """
    
    # ── 1. Pipeline分解 ──
    # 最終ステップが推定機（モデル）とし、それ以前が前処理（スケーラー等）とみなす
    try:
        model = pipeline.steps[-1][1]
        preprocessor = pipeline[:-1]
    except Exception as e:
        logger.error(f"Pipelineの分解に失敗しました: {e}")
        # リストでない、または単一のモデルしかない場合
        model = pipeline
        preprocessor = None
    
    # ── 2. 特徴量名の取得 ──
    try:
        if preprocessor is not None:
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = list(pipeline.get_feature_names_out())
    except AttributeError:
        feature_names = list(X_train_raw.columns) if hasattr(X_train_raw, 'columns') else [f"f{i}" for i in range(np.asarray(X_train_raw).shape[1])]
    except Exception:
        feature_names = [f"f{i}" for i in range(np.asarray(X_train_raw).shape[1])]
        
    X_raw_df = X_train_raw.copy() if isinstance(X_train_raw, pd.DataFrame) else pd.DataFrame(X_train_raw, columns=feature_names)

    # ── 3. 標準化・前処理（SHAP計算用モデル入力空間） ──
    if preprocessor is not None:
        try:
            X_scaled = preprocessor.transform(X_raw_df)
        except Exception:
            # 既に変換済みなどが渡された場合のフォールバック
            X_scaled = X_raw_df.values
    else:
        X_scaled = X_raw_df.values
        
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # ── 4. 特徴量タイプの分類（UI表示切り替え用メタデータ） ──
    feature_types = _classify_features(X_raw_df, feature_names)
    
    # ── 5. Explainerの最適選択 ──
    model_name = model.__class__.__name__.lower()
    
    if any(name in model_name for name in [
        'randomforest', 'gradientboosting', 'histgradient', 
        'extratree', 'lgbm', 'xgb', 'catboost'
    ]):
        # dataを渡すことで、Interventional SHAP値計算（期待値なども正確になる）
        explainer = shap.TreeExplainer(model, data=X_scaled_df)
    elif any(name in model_name for name in ['linear', 'ridge', 'lasso', 'elasticnet']):
        explainer = shap.LinearExplainer(model, data=X_scaled_df)
    else:
        # KernelExplainer: 計算コスト高のため背景サンプルを制限
        logger.info(f"フォールバック: KernelExplainerを使用します (サンプル={background_samples})")
        X_bg = shap.sample(X_scaled_df, min(background_samples, len(X_scaled_df)))
        def _predict_fn(x):
            return model.predict(x)
        explainer = shap.KernelExplainer(_predict_fn, data=X_bg)
    
    # ── 6. SHAP値計算（常に標準化・モデル入力空間で実施） ──
    shap_values = explainer.shap_values(X_scaled_df)
    
    # 多クラス分類等の場合は1つ目のクラスを暫定採用する（リストで返る場合がある）
    is_multiclass = False
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        is_multiclass = True
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
        is_multiclass = True

    # MultiViewResult用のView定義
    # SHAPプロットで「比較（標準化）」と「解釈（生）」を取得するための関数群
    def _view_compare(core_shaps: np.ndarray, **kwargs):
        """標準化空間のデータ(X_scaled)とSHAP値を返す"""
        return {"data": X_scaled_df, "shaps": core_shaps}

    def _view_interpret(core_shaps: np.ndarray, **kwargs):
        """生データのデータ(X_raw)とSHAP値を返す。One-Hot不一致時は補正データを使用"""
        aligned_raw = _align_raw_data(X_raw_df, feature_names, X_scaled_df)
        return {"data": aligned_raw, "shaps": core_shaps}
        
    def _view_global_mean(core_shaps: np.ndarray, **kwargs):
        """全体の絶対値平均（重要度）"""
        return np.abs(core_shaps).mean(axis=0)

    res = MultiViewResult(
        core_output=shap_values,
        views={
            "compare": _view_compare,
            "interpret": _view_interpret,
            "global_mean": _view_global_mean
        },
        metadata={
            "base_value": explainer.expected_value,
            "feature_names": feature_names,
            "feature_types": feature_types,
            "model_type": model_name,
            "is_multiclass": is_multiclass,
            "X_scaled": X_scaled_df,  # 生保存用
            "X_raw": X_raw_df         # 生保存用
        }
    )
    
    return res

def _classify_features(X_raw: pd.DataFrame, feature_names: list) -> Dict[str, str]:
    """
    特徴量を「continuous」「categorical」「onehot」に分類。
    表示ロジックの切り替えに使用。
    """
    types = {}
    for name in feature_names:
        if name in X_raw.columns:
            col = X_raw[name]
            if col.dtype == 'object' or col.nunique() < 10:
                types[name] = 'categorical'
            else:
                types[name] = 'continuous'
        else:
            # One-Hotエンコーディング等で展開された特徴量
            # 元の列名から推定（例: "Solvent_A" -> "Solvent"）
            base_name = name.rsplit('_', 1)[0] if '_' in name else name
            if base_name in X_raw.columns and X_raw[base_name].dtype == 'object':
                types[name] = 'onehot'
            else:
                types[name] = 'continuous'
    return types

def _align_raw_data(X_raw: pd.DataFrame, target_columns: list, X_scaled_df: pd.DataFrame) -> pd.DataFrame:
    """
    生データのDataFrameを、SHAP計算後の特徴量名に合わせる。
    One-Hot展開や対数変換などで列数・名前が異なる場合、生データと標準化データを適切にマージ・フォールバックする。
    """
    available = [c for c in target_columns if c in X_raw.columns]
    
    if len(available) == len(target_columns):
        # 完全一致
        return X_raw[target_columns].copy()
    else:
        # 部分一致: 一致する列は生データを使い、不一致（例えば One-Hot 展開済み列）はX_scaledを使用する
        fallback = pd.DataFrame(np.nan, index=X_scaled_df.index, columns=target_columns)
        for col in target_columns:
            if col in X_raw.columns:
                fallback[col] = X_raw[col].values
            else:
                # 存在しない列（計算で新たに作られたダミー変数等）は、0/1の数値などがそのまま利用できるようにする想定
                fallback[col] = X_scaled_df[col].values
        return fallback
