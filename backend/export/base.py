"""
backend/export/base.py
すべてのExporterが継承する抽象基底クラス
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any


class BaseExporter(abc.ABC):
    """すべてのエクスポータが実装しなければならない共通インターフェース。

    Parameters
    ----------
    output_dir : str | Path
        生成ファイルを保存するディレクトリ。存在しない場合は自動作成する。
    """

    def __init__(self, output_dir: str | Path = "exports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def export(self, result: dict[str, Any], filename: str) -> Path:
        """解析結果を特定フォーマットに変換してディスクへ書き出す。

        Parameters
        ----------
        result : dict
            automl_result 等のアプリケーション共通辞書。
            最低限以下のキーを含むこと:
            - "best_model_name": str
            - "metrics": dict  {"R2": float, "RMSE": float, "MAE": float}
            - "feature_importances": dict {feature: importance}  (任意)
            - "dataframe_head": pd.DataFrame  (任意)
        filename : str
            拡張子を含まないファイル名。

        Returns
        -------
        Path
            書き出したファイルの絶対パス。
        """
        raise NotImplementedError
