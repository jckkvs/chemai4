"""
backend/llm/analysis_advisor.py

LLMを使用して解析方針とアドバイスを生成するモジュール。
Bonsai 8B (GGUF) を使用して、データに基づいた解析アドバイスを提供する。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 解析アドバイス用システムプロンプト
_ANALYSIS_ADVICE_SYSTEM_PROMPT = """\
あなたは化学データ解析の専門家です。
与えられたデータ情報に基づいて、最適な解析方針とアドバイスを提供してください。

## 出力形式
以下の構造でMarkdown形式で出力してください：

### 推奨モデル
データの特性に基づいて、使用すべきモデル（XGBoost, LightGBM, RandomForest, Linear等）を推奨し、その理由を説明してください。

### 推奨特徴量・記述子
どのような特徴量や記述子が有効かをアドバイスしてください。
SMILESデータがある場合は、どの記述子エンジン（RDKit, Mordred等）が適切かも含めてください。

### 注意点・潜在的な問題
データに潜在する問題（欠損値、外れ値、多重共線性など）や、解析時の注意点があれば挙げてください。

### 解析戦略
このデータに対する推奨される解析の進め方をステップで示してください。

## 制約
- 簡潔に、箇条書きを活用して読みやすく書く
- 具体的な数値（サンプル数、特徴量数など）を参照してアドバイスする
- 専門用語を使うが、必要に応じて簡単な説明を添える
"""


class AnalysisAdvisor:
    """
    LLMを使用して解析方針とアドバイスを生成するクラス。
    """

    def __init__(self, provider=None):
        """
        AnalysisAdvisorを初期化する。

        Args:
            provider: LLMProviderインスタンス。Noneの場合はGGUFProviderを使用。
        """
        if provider is None:
            from backend.llm.providers.gguf_provider import GGUFProvider
            provider = GGUFProvider()
        self._provider = provider

    @property
    def is_available(self) -> bool:
        """LLMが利用可能かどうかを返す。"""
        return self._provider.is_available

    def generate_advice(self, state: dict[str, Any]) -> str:
        """
        データ状態から解析アドバイスを生成する。

        Args:
            state: アプリケーションの状態辞書（df, target_col, smiles_col等を含む）

        Returns:
            LLMが生成した解析アドバイス文字列
        """
        if not self.is_available:
            return self._fallback_advice(state)

        try:
            from backend.llm.provider import LLMRequest

            # データ情報を収集
            data_summary = self._build_data_summary(state)

            # ユーザープロンプトを構築
            user_prompt = f"""\
以下のデータに対する解析アドバイスをしてください：

{data_summary}
"""

            request = LLMRequest(
                user_prompt=user_prompt,
                system_prompt=_ANALYSIS_ADVICE_SYSTEM_PROMPT,
                max_tokens=1024,
                temperature=0.3,
            )

            logger.info("[AnalysisAdvisor] LLMにアドバイス生成を依頼...")
            response = self._provider.generate(request)
            logger.info("[AnalysisAdvisor] アドバイス生成完了")

            return response.content

        except Exception as e:
            logger.warning("[AnalysisAdvisor] LLMアドバイス生成失敗: %s", e)
            return self._fallback_advice(state)

    def _build_data_summary(self, state: dict[str, Any]) -> str:
        """データ状態から要約文字列を構築する。"""
        lines = []

        # データフレーム情報
        df = state.get("df")
        if df is not None:
            lines.append(f"- サンプル数: {len(df)}")
            lines.append(f"- 列数: {len(df.columns)}")
            lines.append(f"- 列名: {', '.join(df.columns.tolist())}")

            # データ型情報
            dtype_counts = df.dtypes.value_counts()
            lines.append(f"- データ型: {dict(dtype_counts)}")

            # 欠損値情報
            na_counts = df.isna().sum()
            na_info = {col: int(cnt) for col, cnt in na_counts.items() if cnt > 0}
            if na_info:
                lines.append(f"- 欠損値あり: {na_info}")
            else:
                lines.append("- 欠損値: なし")

            # 数値列の基本統計量
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                lines.append(f"- 数値列: {len(numeric_cols)}個")
                # ターゲット列の情報
                target_col = state.get("target_col")
                if target_col and target_col in df.columns:
                    target_data = df[target_col].dropna()
                    if len(target_data) > 0:
                        lines.append(f"- 目的変数「{target_col}」: 範囲 [{target_data.min():.4f}, {target_data.max():.4f}], 平均 {target_data.mean():.4f}")

        else:
            lines.append("- データ: 未読込")

        # SMILES列
        smiles_col = state.get("smiles_col")
        if smiles_col:
            lines.append(f"- SMILES列: {smiles_col}")
            # SMILESデータがある場合の追加情報
            if df is not None and smiles_col in df.columns:
                valid_smiles = df[smiles_col].dropna()
                lines.append(f"- 有効SMILES数: {len(valid_smiles)}")
        else:
            lines.append("- SMILES列: なし")

        # タスクタイプ
        task_type = state.get("task_type", "auto")
        lines.append(f"- タスクタイプ: {task_type}")

        # 選択されたモデル
        selected_models = state.get("selected_models")
        if selected_models:
            lines.append(f"- 選択済みモデル: {', '.join(selected_models)}")
        else:
            lines.append("- 選択済みモデル: デフォルト")

        # 選択された記述子エンジン
        engine_flags = {
            "use_rdkit": "RDKit",
            "use_mordred": "Mordred",
            "use_xtb": "XTB",
            "use_skfp": "scikit-FP",
            "use_molai": "MolAI",
            "use_mol2vec": "Mol2Vec",
            "use_groupcontrib": "GroupContrib",
            "use_uma": "UMA",
            "use_padel": "PaDEL",
            "use_descriptastorus": "DescriptaStorus",
            "use_molfeat": "Molfeat",
            "use_chemprop": "Chemprop",
            "use_cosmo": "COSMO-RS",
            "use_unipka": "UniPka",
        }
        active_engines = [name for key, name in engine_flags.items() if state.get(key)]
        if active_engines:
            lines.append(f"- 有効記述子エンジン: {', '.join(active_engines)}")

        return "\n".join(lines)

    def _fallback_advice(self, state: dict[str, Any]) -> str:
        """LLMが利用できない場合のフォールバック・アドバイス。"""
        lines = [
            "### LLMアドバイス（簡易版）",
            "",
            "LLMが利用できないため、基本アドバイスを表示します：",
            "",
            "### 推奨モデル",
            "- 回帰タスクの場合: XGBoost, LightGBM, RandomForest を推奨",
            "- 分類タスクの場合: XGBoost, LightGBM, LogisticRegression を推奨",
            "",
            "### 推奨特徴量・記述子",
        ]

        if state.get("smiles_col"):
            lines.append("- SMILESデータあり: RDKit, Mordred記述子を推奨")
            lines.append("- 物理化学的性質を捉えるために複数の記述子エンジンを組み合わせることを推奨")
        else:
            lines.append("- 数値特徴量を標準化（StandardScaler）して使用することを推奨")

        lines.extend([
            "",
            "### 注意点",
            "- 欠損値は事前に確認し、適切に処理してください",
            "- 外れ値がないか確認してください",
            "- 特徴量が多い場合は次元削減や特徴量選択を検討してください",
            "",
            "### 解析戦略",
            "1. データの前処理（欠損値、外れ値処理）",
            "2. 記述子生成（SMILESがある場合）",
            "3. 特徴量選択・次元削減",
            "4. モデル学習と交差検証",
            "5. 結果の解釈（SHAP等）",
        ])

        return "\n".join(lines)
