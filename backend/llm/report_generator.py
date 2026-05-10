"""
backend/llm/report_generator.py

AutoML解析結果からLLMを使って解析レポートを生成する。
既存のLLMCodeReviewerと同じプロバイダーパターンに従う。
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from backend.llm.provider import LLMProvider
    from backend.models.automl import AutoMLResult

logger = logging.getLogger(__name__)

# ── システムプロンプト ──────────────────────────────────────────────────────────

_REPORT_SYSTEM_PROMPT = """\
あなたはChemAI ML Studioの解析レポートを生成する専門家です。
化学情報学・機械学習・データ分析に精通しています。

## 出力形式
必ず以下のJSON形式で回答してください（コードブロックなし）:
{
  "title": "レポートタイトル",
  "summary": "エグゼクティブサマリー（3-5文、日本語）",
  "sections": [
    {
      "title": "セクションタイトル",
      "content": "Markdown形式の詳細説明",
      "section_type": "summary|performance|features|data_quality|recommendations"
    }
  ]
}

## レポートに含めるべきセクション

### 1. エグゼクティブサマリー (section_type: "summary")
- 解析の全体像（タスク種別、データ規模）
- 最良モデルとその性能
- 主要な発見

### 2. モデル性能評価 (section_type: "performance")
- 最良モデル名とスコア
- 全モデル比較（model_scoresを参照）
- CVスコアの安定性（標準偏差）
- 過学習の有無（Train vs CVの差）

### 3. 特徴量重要度分析 (section_type: "features")
- 上位の重要特徴量の解釈
- 化学的意味（SMILES記述子の場合）
- 特徴量選択の妥当性

### 4. データ品質と前処理 (section_type: "data_quality")
- データ規模と欠損状況
- 前処理の内容（preprocess_report）
- 適用領域（AD）の評価
- SMILES相関（smiles_correlations）

### 5. 推奨事項 (section_type: "recommendations")
- モデル改善案
- 追加すべき記述子や特徴量
- データ収集の提案
- 次のステップ

## 重要ルール
- 全て日本語で出力する
- 化学ドメイン知識を活用する
- 数値は正確に引用する（小数点以下4桁まで）
- 過度に技術的になりすぎない（経営層も理解できること）
- JSON以外は出力しない
- sections配列には最低3つ以上のセクションを含めること
"""


# ── レポート生成クラス ──────────────────────────────────────────────────────────

class LLMReportGenerator:
    """
    AutoML解析結果からLLMを使って解析レポートを生成する。

    使用例:
        from backend.llm import get_llm_provider
        from backend.llm.report_generator import LLMReportGenerator
        provider = get_llm_provider("huggingface")
        gen = LLMReportGenerator(provider)
        result = gen.generate_report(automl_result)
        print(result.to_markdown())
    """

    def __init__(self, provider: "LLMProvider") -> None:
        self.provider = provider

    def generate_report(
        self,
        automl_result: "AutoMLResult",
        state: Optional[dict] = None,
        max_tokens: int = 4096,
    ) -> "ReportResult":
        """
        AutoML解析結果からLLMレポートを生成する。

        Args:
            automl_result: AutoML解析結果
            state: 追加コンテキスト（データフレーム情報等）
            max_tokens: 最大生成トークン数

        Returns:
            ReportResult
        """
        from backend.llm.provider import LLMRequest
        from backend.llm.report_schemas import ReportResult

        # 1. プロンプト構築
        user_prompt = self._build_prompt(automl_result, state)

        # 2. LLMリクエスト作成
        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=_REPORT_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.3,  # レポート生成は創造性より正確性を優先
        )

        # 3. LLM呼び出し
        try:
            from datetime import datetime, timezone
            response = self.provider.generate(request)
            tokens_used = response.tokens_used if response.tokens_used else 0
            result = _parse_report_response(response.content, tokens_used)
            # メタデータを設定
            result.generated_at = datetime.now(timezone.utc).isoformat()
            result.model_info = f"{self.provider.name}"
            return result
        except Exception as e:
            logger.warning(f"[ReportGen] LLMレポート生成失敗: {e}")
            result = _static_fallback_report(automl_result)
            from datetime import datetime, timezone
            result.generated_at = datetime.now(timezone.utc).isoformat()
            result.model_info = f"{self.provider.name} (fallback)"
            return result

    def _build_prompt(
        self, ar: "AutoMLResult", state: Optional[dict]
    ) -> str:
        """AutoMLResultからプロンプトを構築する。"""
        lines: list[str] = []

        # 基本情報
        lines.append("## 解析情報")
        lines.append(f"- タスク: {ar.task}")
        lines.append(f"- スコアリング: {ar.scoring}")
        lines.append(f"- 最良モデル: {ar.best_model_key}")
        lines.append(f"- 最良スコア: {ar.best_score:.4f}")
        lines.append(f"- 解析時間: {ar.elapsed_seconds:.1f}秒")

        # モデル比較
        lines.append("\n## モデルスコア（全モデル）")
        sorted_scores = sorted(ar.model_scores.items(), key=lambda x: -x[1])
        for key, score in sorted_scores:
            detail = ar.model_details.get(key, {})
            std = detail.get("std", 0.0)
            fit_time = detail.get("fit_time", 0.0)
            lines.append(f"- {key}: {score:.4f} ± {std:.4f} ( fit: {fit_time:.1f}s )")

        # データ情報
        if ar.X_train is not None:
            lines.append("\n## データ情報")
            lines.append(f"- サンプル数: {ar.X_train.shape[0]}")
            lines.append(f"- 特徴量数: {ar.X_train.shape[1]}")
            if ar.y_train is not None:
                import numpy as np
                if ar.task == "regression":
                    lines.append(f"- 目標変数: 平均={float(np.mean(ar.y_train)):.4f}, 標準偏差={float(np.std(ar.y_train)):.4f}")
                else:
                    import collections
                    counts = collections.Counter(ar.y_train)
                    for cls, cnt in counts.most_common():
                        lines.append(f"- クラス {cls}: {cnt}件")

        # 前処理情報
        if ar.preprocess_report:
            lines.append("\n## 前処理レポート")
            lines.append(ar.preprocess_report.generate_summary())

        # SMILES相関
        if ar.smiles_correlations:
            lines.append("\n## SMILES記述子と目標変数の相関（上位10）")
            sorted_corr = sorted(
                ar.smiles_correlations.items(), key=lambda x: -abs(x[1])
            )[:10]
            for name, corr in sorted_corr:
                lines.append(f"- {name}: {corr:.4f}")

        # 適用領域（Applicability Domain）
        if ar.in_domain_cv is not None:
            in_domain_ratio = float(ar.in_domain_cv.mean()) * 100
            lines.append("\n## 適用領域（Applicability Domain）")
            lines.append(f"- 適用領域内サンプル: {in_domain_ratio:.1f}%")

        # 警告
        if ar.warnings:
            lines.append("\n## 警告・注意事項")
            for w in ar.warnings:
                lines.append(f"- {w}")

        # 単調制約
        if ar.resolved_constraints:
            lines.append("\n## 単調制約（Monotonic Constraints）")
            for feat, constraint in ar.resolved_constraints.items():
                label = "増加傾向" if constraint == 1 else ("減少傾向" if constraint == -1 else "なし")
                lines.append(f"- {feat}: {label}")

        # 特徴量重要度（パイプラインから抽出試行）
        lines.append("\n## 特徴量重要度（上位10）")
        _append_feature_importance(lines, ar)

        return "\n".join(lines)


# ── レスポンスパース ────────────────────────────────────────────────────────────

def _parse_report_response(raw: str, tokens_used: int) -> "ReportResult":
    """LLM出力をReportResultに変換する。"""
    from backend.llm.report_schemas import ReportResult, ReportSection

    try:
        # JSON抽出
        clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("```").strip()
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)

        data = json.loads(clean)

        sections = []
        for item in data.get("sections", []):
            sections.append(ReportSection(
                title=item.get("title", ""),
                content=item.get("content", ""),
                section_type=item.get("section_type", "summary"),
            ))

        return ReportResult(
            success=True,
            title=data.get("title", "解析レポート"),
            summary=data.get("summary", ""),
            sections=sections,
            raw_response=raw,
            tokens_used=tokens_used,
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"[ReportGen] JSONパース失敗: {e}\nraw: {raw[:300]}")
        # フォールバック: 生テキストを行で分割してセクション化
        from backend.llm.report_schemas import ReportResult, ReportSection
        return ReportResult(
            success=True,
            title="解析レポート",
            summary="LLM生成レポート（フォーマット解析失敗のため全文表示）",
            sections=[ReportSection(
                title="LLM生成結果",
                content=raw,
                section_type="summary",
            )],
            raw_response=raw,
            tokens_used=tokens_used,
        )


def _static_fallback_report(ar: "AutoMLResult") -> "ReportResult":
    """LLMが使えない場合の静的フォールバックレポート。"""
    from backend.llm.report_schemas import ReportResult, ReportSection

    lines: list[str] = []
    lines.append(f"## 解析概要")
    lines.append(f"- タスク: {ar.task}")
    lines.append(f"- 最良モデル: {ar.best_model_key}")
    lines.append(f"- スコア: {ar.best_score:.4f}")
    lines.append(f"- 解析時間: {ar.elapsed_seconds:.1f}秒")

    lines.append(f"\n## モデル比較")
    for key, score in sorted(ar.model_scores.items(), key=lambda x: -x[1]):
        detail = ar.model_details.get(key, {})
        std = detail.get("std", 0.0)
        lines.append(f"- {key}: {score:.4f} ± {std:.4f}")

    if ar.X_train is not None:
        lines.append(f"\n## データ情報")
        lines.append(f"- サンプル数: {ar.X_train.shape[0]}")
        lines.append(f"- 特徴量数: {ar.X_train.shape[1]}")

    content = "\n".join(lines)

    return ReportResult(
        success=True,
        title=f"{ar.task}解析レポート（{ar.best_model_key}）",
        summary=f"{ar.task}タスクにおいて{ar.best_model_key}が最良モデルとして選択されました（スコア: {ar.best_score:.4f}）。",
        sections=[
            ReportSection(
                title="解析結果概要",
                content=content,
                section_type="summary",
            ),
        ],
        error_message="LLMを使用できなかったため、静的レポートを生成しました。",
    )


def _append_feature_importance(lines: list[str], ar: "AutoMLResult") -> None:
    """パイプラインから特徴量重要度を抽出してプロンプトに追加する。"""
    try:
        pipeline = ar.best_pipeline
        if pipeline is None:
            lines.append("- 情報なし（パイプラインがNone）")
            return

        # Pipelineの最終ステップ（推定器）から重要度を取得
        estimator = pipeline.steps[-1][1] if pipeline.steps else None
        if estimator is None:
            lines.append("- 情報なし（推定器が見つかりません）")
            return

        import numpy as np

        # 特徴量名を取得（processed_Xから）
        feature_names = None
        if ar.processed_X is not None:
            feature_names = list(ar.processed_X.columns)
        elif ar.X_train is not None:
            feature_names = list(ar.X_train.columns)

        # 重要度の取得を試行
        importance = None
        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)

        if importance is not None and feature_names is not None:
            # 上位10を表示
            indices = np.argsort(importance)[::-1][:10]
            for i in indices:
                if i < len(feature_names):
                    lines.append(f"- {feature_names[i]}: {importance[i]:.4f}")
        elif importance is not None:
            indices = np.argsort(importance)[::-1][:10]
            for i in indices:
                lines.append(f"- 特徴量{i}: {importance[i]:.4f}")
        else:
            lines.append("- このモデルは特徴量重要度を提供しません")
    except Exception as e:
        logger.warning(f"[ReportGen] 特徴量重要度抽出失敗: {e}")
        lines.append("- 重要度抽出中にエラーが発生しました")
