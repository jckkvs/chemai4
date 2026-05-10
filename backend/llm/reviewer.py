"""
backend/llm/reviewer.py

LLMを使った生成コードのレビュワー。
「生成LLM が作ったコード」を別のLLM（または同じモデル）が批判的に検査する
セカンドオピニオン機構。

チェック観点:
  1. 化学的正しさ  - 記述子の計算ロジックが化学的に妥当か
  2. コードの堅牢性 - None ハンドリング、戻り値の型と長さ
  3. 実装の完全性  - ユーザーの意図を正しく実装しているか
  4. パフォーマンス - ループ内での不要な重い処理がないか
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


# ── レビュー結果 ──────────────────────────────────────────────────────────────

@dataclass
class ReviewIssue:
    severity: str     # "ERROR" | "WARN" | "INFO"
    category: str     # "chemistry" | "robustness" | "completeness" | "performance"
    message: str      # 問題の説明（日本語）
    suggestion: str = ""  # 改善案


@dataclass
class CodeReviewResult:
    verdict: str                          # "PASS" | "WARN" | "FAIL"
    score: int                            # 0~100 (100=完璧)
    summary: str                          # 総評（1〜2文）
    issues: list[ReviewIssue] = field(default_factory=list)
    raw_response: str = ""                # LLMの生の出力（デバッグ用）

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARN")


# ── システムプロンプト ─────────────────────────────────────────────────────────

_REVIEW_SYSTEM_PROMPT = """\
あなたはChemAI ML Studioの記述子プラグインコードを厳格にレビューする専門家です。
化学情報学・RDKit・機械学習パイプラインに精通しています。

## レビュー対象
ユーザーが自然言語で指定した「計算したい記述子」に対して生成されたPythonコードです。

## 必ずJSON形式で回答してください（コードブロックなし）:
{
  "verdict": "PASS" | "WARN" | "FAIL",
  "score": 0〜100の整数,
  "summary": "1〜2文の総評（日本語）",
  "issues": [
    {
      "severity": "ERROR" | "WARN" | "INFO",
      "category": "chemistry" | "robustness" | "completeness" | "performance",
      "message": "問題の説明（日本語）",
      "suggestion": "改善策（日本語）"
    }
  ]
}

## 判定基準
- PASS (score≥80): そのまま使用可能
- WARN (score 50-79): 軽微な問題あり、使用可能だが改善推奨
- FAIL (score<50): ERROR が1件以上あり、修正必須

## チェック観点
1. **化学的正しさ**: 記述子の定義・計算が化学的に正しいか
2. **堅牢性**: 全SMILES に None ハンドリングがあるか、戻り値が list[float|None] か
3. **完全性**: ユーザーの意図（説明文）通りに実装されているか
4. **パフォーマンス**: ループ内で重い計算を繰り返していないか

## 重要ルール
- JSON以外は出力しない
- issuesが0件でもPASSを返す
"""


# ── レビュワークラス ──────────────────────────────────────────────────────────

class LLMCodeReviewer:
    """
    生成されたコードをLLMでレビューする。

    使用例:
        from backend.llm import get_llm_provider
        from backend.llm.reviewer import LLMCodeReviewer
        provider = get_llm_provider("huggingface")
        reviewer = LLMCodeReviewer(provider)
        result = reviewer.review(code, user_intent="logPを計算する記述子")
        print(result.verdict, result.score, result.summary)
    """

    def __init__(self, provider: "LLMProvider") -> None:
        self.provider = provider

    def review(
        self,
        code: str,
        user_intent: str = "",
        max_tokens: int = 1024,
    ) -> CodeReviewResult:
        """
        コードをレビューして結果を返す。

        Args:
            code: レビュー対象のPythonコード
            user_intent: ユーザーが最初に入力した「作りたい記述子の説明」
            max_tokens: 最大生成トークン数

        Returns:
            CodeReviewResult
        """
        from backend.llm.provider import LLMRequest

        intent_section = f"\n## ユーザーの意図\n{user_intent}" if user_intent else ""
        user_prompt = (
            f"{intent_section}\n\n"
            f"## レビュー対象コード\n```python\n{code}\n```\n\n"
            "上記のコードをレビューしてJSON形式で結果を返してください。"
        )

        request = LLMRequest(
            user_prompt=user_prompt,
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=0.1,  # レビューは決定論的に
        )

        try:
            response = self.provider.generate(request)
            raw = response.content.strip()
            return _parse_review_response(raw)
        except Exception as e:
            logger.warning(f"[Reviewer] LLMレビュー失敗: {e}")
            # フォールバック: 静的チェックのみ
            return _static_fallback_review(code)


# ── レスポンスパース ──────────────────────────────────────────────────────────

def _parse_review_response(raw: str) -> CodeReviewResult:
    """LLMの生出力をCodeReviewResultに変換する。"""
    # コードブロックがあれば除去
    clean = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("```").strip()

    # JSON抽出（出力に前後テキストが混じる場合）
    json_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if json_match:
        clean = json_match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.warning(f"[Reviewer] JSONパース失敗: {e}\nraw: {raw[:300]}")
        return CodeReviewResult(
            verdict="WARN",
            score=50,
            summary="レビュー結果のパースに失敗しました。静的チェックのみ実施済みです。",
            raw_response=raw,
        )

    issues = []
    for item in data.get("issues", []):
        issues.append(ReviewIssue(
            severity=item.get("severity", "INFO"),
            category=item.get("category", "completeness"),
            message=item.get("message", ""),
            suggestion=item.get("suggestion", ""),
        ))

    return CodeReviewResult(
        verdict=data.get("verdict", "WARN"),
        score=int(data.get("score", 50)),
        summary=data.get("summary", ""),
        issues=issues,
        raw_response=raw,
    )


def _static_fallback_review(code: str) -> CodeReviewResult:
    """LLMが使えない場合の静的チェックによるフォールバックレビュー。"""
    issues = []

    # None ハンドリングチェック
    if "None" not in code and "none" not in code.lower():
        issues.append(ReviewIssue(
            severity="WARN",
            category="robustness",
            message="Noneハンドリングが見当たりません",
            suggestion="無効なSMILESに対してNoneを返すtry/exceptを追加してください",
        ))

    # compute関数チェック
    if "def compute(" not in code:
        issues.append(ReviewIssue(
            severity="ERROR",
            category="completeness",
            message="compute()関数が定義されていません",
            suggestion="def compute(smiles_list: list[str]) -> list を実装してください",
        ))

    # DESCRIPTOR_NAMEチェック
    if "DESCRIPTOR_NAME" not in code:
        issues.append(ReviewIssue(
            severity="ERROR",
            category="completeness",
            message="DESCRIPTOR_NAMEが未定義です",
            suggestion='先頭に DESCRIPTOR_NAME = "記述子名" を追加してください',
        ))

    score = max(0, 80 - len([i for i in issues if i.severity == "ERROR"]) * 30
                - len([i for i in issues if i.severity == "WARN"]) * 10)
    verdict = "PASS" if score >= 80 else ("WARN" if score >= 50 else "FAIL")

    return CodeReviewResult(
        verdict=verdict,
        score=score,
        summary="LLMレビュー不可のため静的チェックのみ実施しました。",
        issues=issues,
    )
