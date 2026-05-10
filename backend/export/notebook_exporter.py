"""
backend/export/notebook_exporter.py
nbformat を用いた Jupyter Notebook (.ipynb) 自動生成エンジン。

依存: nbformat>=5.9
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
)

from .base import BaseExporter


class NotebookExporter(BaseExporter):
    """解析結果を Jupyter Notebook に変換するエクスポータ。

    生成するセル構成:
    1. Markdown: タイトル + セクションヘッダー
    2. Code: 必要パッケージの import
    3. Code: データ読み込み（pandas.read_csv サンプル）
    4. Code: 前処理パイプライン設定の再現コード
    5. Code: モデル学習 + 指標の計算・表示
    6. Code: SHAP による特徴量重要度の可視化
    7. Markdown: 解析サマリー（指標値・最良モデル名）
    """

    def _imports_cell(self) -> nbformat.NotebookNode:
        code = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
print("✅ Libraries loaded successfully")
"""
        return new_code_cell(code)

    def _data_load_cell(self, result: dict[str, Any]) -> nbformat.NotebookNode:
        csv_path = result.get("csv_path", "data.csv")
        result.get("target_col", "target")
        result.get("smiles_col", "")

        lines = [
            '# データの読み込み',
            f'df = pd.read_csv("{csv_path}")',
            'print(f"読み込み完了: {{df.shape[0]}} 行 × {{df.shape[1]}} 列")',
            'df.head()',
        ]
        return new_code_cell("\n".join(lines))

    def _preprocessing_cell(self, result: dict[str, Any]) -> nbformat.NotebookNode:
        target = result.get("target_col", "target")
        feature_cols = result.get("feature_cols", [])
        scaler_name = result.get("scaler", "StandardScaler")
        imputer_name = result.get("imputer", "SimpleImputer(strategy='median')")

        feature_list_str = repr(feature_cols) if feature_cols else "df.columns.drop(target).tolist()"
        code = f"""\
from sklearn.preprocessing import {scaler_name}
from sklearn.impute import {imputer_name.split('(')[0]}
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

target_col = "{target}"
feature_cols = {feature_list_str}

X = df[feature_cols]
y = df[target_col]

# 前処理パイプライン
preprocessor = Pipeline([
    ("imputer", {imputer_name}),
    ("scaler", {scaler_name}()),
])

X_processed = preprocessor.fit_transform(X)
print(f"前処理後の特徴量行列: {{X_processed.shape}}")
"""
        return new_code_cell(code)

    def _model_cell(self, result: dict[str, Any]) -> nbformat.NotebookNode:
        model_name = result.get("best_model_name", "RandomForestRegressor")
        best_params = result.get("best_params", {})
        cv_folds = result.get("cv_folds", 5)
        result.get("metrics", {})

        params_str = ", ".join(f"{k}={repr(v)}" for k, v in best_params.items()) if best_params else ""
        code = f"""\
# ベストモデルの再学習
# ChemAI で採択されたモデル: {model_name}
# ハイパーパラメータ: {best_params}

try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    model = {model_name}({params_str})
except Exception as e:
    print(f"モデルのインポートに失敗しました: {{e}}")
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits={cv_folds}, shuffle=True, random_state=42)
cv_results = cross_validate(
    model, X_processed, y,
    cv=cv,
    scoring={{"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}},
    return_train_score=False,
)

print(f"R²   : {{cv_results['test_r2'].mean():.4f}} ± {{cv_results['test_r2'].std():.4f}}")
print(f"RMSE : {{-cv_results['test_neg_rmse'].mean():.4f}}")
print(f"MAE  : {{-cv_results['test_neg_mae'].mean():.4f}}")

# 最終モデルを全データで再学習
model.fit(X_processed, y)
print("✅ モデル学習完了")
"""
        return new_code_cell(code)

    def _shap_cell(self) -> nbformat.NotebookNode:
        code = """\
# SHAP による特徴量重要度の可視化
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_processed)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_processed, feature_names=feature_cols, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
"""
        return new_code_cell(code)

    def _summary_markdown(self, result: dict[str, Any]) -> nbformat.NotebookNode:
        model_name = result.get("best_model_name", "N/A")
        metrics = result.get("metrics", {})
        lines = [
            "## 📊 解析サマリー",
            "",
            f"- **採択モデル**: {model_name}",
        ]
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- **{k}**: {val_str}")

        ai_comment = result.get("ai_commentary", "")
        if ai_comment:
            lines.extend(["", "### 🤖 AIコメント", "", ai_comment])

        return new_markdown_cell("\n".join(lines))

    def export(self, result: dict[str, Any], filename: str) -> Path:
        """解析結果を Jupyter Notebook ファイルとして output_dir へ書き出す。

        Parameters
        ----------
        result : dict
            必須キー: "best_model_name", "metrics"
            任意キー: "best_params", "csv_path", "target_col", "feature_cols",
                      "scaler", "imputer", "cv_folds", "ai_commentary"
        filename : str
            拡張子なしのファイル名（例: "analysis_notebook"）。

        Returns
        -------
        Path
            書き出した .ipynb ファイルの絶対パス。
        """
        out_path = self.output_dir / f"{filename}.ipynb"

        nb = new_notebook()
        nb.cells = [
            new_markdown_cell(
                f"# ChemAI ML Studio: 解析ノートブック\n\n"
                f"> 自動生成日時: {Path(filename).stem}\n\n"
                "このノートブックは ChemAI ML Studio によって自動生成されました。"
            ),
            self._imports_cell(),
            new_markdown_cell("## 1. データの読み込み"),
            self._data_load_cell(result),
            new_markdown_cell("## 2. 前処理パイプライン"),
            self._preprocessing_cell(result),
            new_markdown_cell("## 3. モデル学習と評価"),
            self._model_cell(result),
            new_markdown_cell("## 4. 特徴量重要度（SHAP）"),
            self._shap_cell(),
            self._summary_markdown(result),
        ]

        with open(out_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return out_path.resolve()
