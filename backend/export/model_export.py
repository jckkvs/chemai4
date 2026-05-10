"""
backend/export/model_export.py

訓練済みの機械学習モデルを複数の形式（joblib, ONNX, PMML）でエクスポートするモジュール。
外部依存パッケージ (skl2onnx, nyoka) がないシステムでは gracefully-fallback して joblib のみを許可します。
"""
import os
import joblib
import logging
from typing import Any

logger = logging.getLogger(__name__)

def export_model(model: Any, filepath: str, format: str = "joblib", **kwargs: Any) -> str:
    """
    モデルをファイルにエクスポートする。

    Args:
        model: 学習済みのscikit-learn準拠モデルまたはパイプライン
        filepath: 出力先のベースファイルパスまたは拡張子付きパス
        format: "joblib" | "onnx" | "pmml"
        **kwargs: 変換用の追加引数（ONNXの initial_types など）

    Returns:
        保存されたファイルのパス
    """
    format = format.lower()
    
    # 拡張子の補完
    if not filepath.endswith(f".{format}"):
        filepath = f"{filepath}.{format}"

    # 保存ディレクトリの作成
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if format == "joblib":
        joblib.dump(model, filepath)
        logger.info(f"モデルを joblib 形式で保存しました: {filepath}")
        return filepath

    elif format == "onnx":
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "ONNXエクスポートには 'skl2onnx' が必要です。"
                "pip install skl2onnx でインストールするか、format='joblib' を使用してください。"
            )

        # デフォルトは (None, n_features) のFloat行列を想定。kwargsで上書き可能。
        initial_types = kwargs.get("initial_types")
        if initial_types is None:
            # 推定: もしモデルが hasattrでn_features_in_を持つならそれを使うが、
            # わからない場合はとりあえず Unknown としてエラーを出さないようにする
            n_features = getattr(model, "n_features_in_", 10)
            initial_types = [('float_input', FloatTensorType([None, n_features]))]

        onnx_model = convert_sklearn(model, initial_types=initial_types)
        with open(filepath, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"モデルを ONNX 形式で保存しました: {filepath}")
        return filepath

    elif format == "pmml":
        try:
            from nyoka import sklearn_to_pmml
        except ImportError:
            raise ImportError(
                "PMMLエクスポートには 'nyoka' が必要です。"
                "pip install nyoka でインストールするか、format='joblib' を使用してください。"
            )

        features = kwargs.get("features")
        target_name = kwargs.get("target_name", "target")
        sklearn_to_pmml(
            pipeline=model,
            col_names=features,
            target_name=target_name,
            pmml_f_name=filepath
        )
        logger.info(f"モデルを PMML 形式で保存しました: {filepath}")
        return filepath

    else:
        raise ValueError(f"未対応のフォーマットです: {format}。 'joblib', 'onnx', 'pmml' を指定してください。")
