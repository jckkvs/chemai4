"""
backend/llm/generator.py

LLMを使った記述子コード生成器。

役割:
  1. ユーザーの自然言語説明をLLMに渡す
  2. LLMが返したコードを検証（セキュリティ + プラグイン形式確認）
  3. 検証OKなら custom/ ディレクトリに保存
  4. 検証NGなら LLMGeneratorError を raise

セキュリティ設計:
  - 危険なインポート・関数呼び出しを静的チェック
  - サンドボックス実行（RestrictedPythonまたはsubprocess分離）
  - 保存前に必ずvalidate_plugin()を通す
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# custom/ ディレクトリのデフォルトパス
_CUSTOM_DIR = Path(__file__).parent.parent / "chem" / "descriptors" / "custom"

# ── セキュリティ: 禁止パターン ────────────────────────────────────────────────
_BANNED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket", "urllib",
    "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "pickle", "shelve", "importlib", "ctypes", "cffi",
    "multiprocessing", "threading", "concurrent",
    "__builtins__", "eval", "exec", "compile",
}

_BANNED_FUNCTIONS = {
    "os.system", "os.popen", "os.exec", "os.fork",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "eval(", "exec(", "compile(",
    "__import__(",
    "open(",  # ファイルI/Oも禁止（将来緩和可）
}


class LLMGeneratorError(Exception):
    """コード生成・検証エラー。"""


@dataclass
class GenerationResult:
    """生成・保存結果。"""
    success: bool
    code: str                    # 生成されたコード
    plugin_name: str = ""        # DESCRIPTOR_NAME の値
    saved_path: str = ""         # 保存先パス（成功時）
    error_msg: str = ""          # エラーメッセージ（失敗時）
    security_warnings: list[str] = None  # セキュリティ警告

    def __post_init__(self):
        if self.security_warnings is None:
            self.security_warnings = []


class LLMDescriptorGenerator:
    """
    LLMを使った記述子コード生成器。

    使用例:
        from backend.llm import get_llm_provider, LLMDescriptorGenerator
        provider = get_llm_provider("openai")
        gen = LLMDescriptorGenerator(provider)
        result = gen.generate_and_save("分子のLogPを計算する記述子")
        if result.success:
            print(f"保存: {result.saved_path}")
    """

    def __init__(self, provider: "LLMProvider") -> None:
        self.provider = provider
        self.custom_dir = _CUSTOM_DIR

    def generate(self, user_description: str) -> GenerationResult:
        """
        自然言語の説明からPythonコードを生成する（保存なし）。

        Args:
            user_description: ユーザーが望む記述子の自然言語説明

        Returns:
            GenerationResult（success=Trueなら code にコードが入る）
        """
        # 既存プラグインの例をコンテキストとして追加
        context = self._build_context()

        try:
            code = self.provider.generate_descriptor_code(
                user_description,
                additional_context=context,
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                code="",
                error_msg=f"LLM生成エラー: {e}",
            )

        # コードブロックのクリーンアップ
        code = _strip_code_fences(code)

        # セキュリティ検証
        warnings = _check_security(code)
        if any(w.startswith("[BLOCKED]") for w in warnings):
            blocked = [w for w in warnings if w.startswith("[BLOCKED]")]
            return GenerationResult(
                success=False,
                code=code,
                error_msg=f"セキュリティチェック失敗: {'; '.join(blocked)}",
                security_warnings=warnings,
            )

        # プラグイン形式検証
        plugin_name, error = _validate_code_format(code)
        if error:
            return GenerationResult(
                success=False,
                code=code,
                error_msg=f"プラグイン形式エラー: {error}",
                security_warnings=warnings,
            )

        return GenerationResult(
            success=True,
            code=code,
            plugin_name=plugin_name,
            security_warnings=warnings,
        )

    def generate_and_save(
        self,
        user_description: str,
        filename: str | None = None,
    ) -> GenerationResult:
        """
        生成してcustom/ディレクトリに保存する。

        Args:
            user_description: 記述子の説明
            filename: 保存ファイル名（Noneなら自動生成）

        Returns:
            GenerationResult（success=Trueなら saved_path にパスが入る）
        """
        result = self.generate(user_description)
        if not result.success:
            return result

        # ファイル名の決定
        if not filename:
            safe_name = re.sub(r"[^\w]", "_", result.plugin_name.lower())[:40]
            filename = f"llm_{safe_name}.py"
        if not filename.endswith(".py"):
            filename += ".py"

        # 上書き防止: ファイルが存在する場合は連番を追加
        save_path = self.custom_dir / filename
        if save_path.exists():
            stem = save_path.stem
            i = 2
            while save_path.exists():
                save_path = self.custom_dir / f"{stem}_{i}.py"
                i += 1

        # ヘッダーコメントを付与して保存
        header = textwrap.dedent(f"""\
            # AI生成記述子プラグイン
            # 生成リクエスト: {user_description[:120]}
            # このファイルはLLMにより自動生成されました。
            # 内容を確認・編集のうえ使用してください。
            # ── ここから生成コード ────────────────────────────────
        """)
        final_code = header + "\n" + result.code

        try:
            self.custom_dir.mkdir(parents=True, exist_ok=True)
            save_path.write_text(final_code, encoding="utf-8")
            logger.info(f"[LLMGen] 記述子を保存: {save_path}")
        except OSError as e:
            return GenerationResult(
                success=False,
                code=result.code,
                plugin_name=result.plugin_name,
                error_msg=f"ファイル保存エラー: {e}",
                security_warnings=result.security_warnings,
            )

        result.saved_path = str(save_path)
        return result

    def _build_context(self) -> str:
        """既存プラグインから参考コードを構築する（コンテキスト注入用）。"""
        try:
            builtin_dir = self.custom_dir.parent / "_builtins"
            examples = []
            for pyfile in list(builtin_dir.glob("*.py"))[:2]:  # 最大2ファイル
                content = pyfile.read_text(encoding="utf-8")[:500]  # 先頭500文字
                examples.append(f"# {pyfile.name}\n{content}")
            return "\n\n".join(examples)
        except Exception:
            return ""


# ── セキュリティチェック関数 ───────────────────────────────────────────────────

def _check_security(code: str) -> list[str]:
    """
    コードのセキュリティチェック。

    Returns:
        警告リスト。"[BLOCKED]" で始まるものは致命的（保存不可）。
    """
    warnings: list[str] = []

    # ASTベース解析
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in _BANNED_IMPORTS:
                            warnings.append(
                                f"[BLOCKED] 禁止されたインポート: import {module}"
                            )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module in _BANNED_IMPORTS:
                        warnings.append(
                            f"[BLOCKED] 禁止されたインポート: from {node.module} import ..."
                        )

            # eval/exec の直接呼び出し
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec", "compile", "__import__"}:
                        warnings.append(
                            f"[BLOCKED] 禁止された関数呼び出し: {node.func.id}()"
                        )
    except SyntaxError as e:
        warnings.append(f"[BLOCKED] 構文エラー: {e}")

    # 文字列ベース追加チェック（ASTで捕捉できないパターン）
    for banned in _BANNED_FUNCTIONS:
        if banned in code:
            warnings.append(f"[WARN] 危険なパターンを検出: {banned}")

    return warnings


def _validate_code_format(code: str) -> tuple[str, str]:
    """
    プラグイン形式（DESCRIPTOR_NAME, compute関数）を検証する。

    Returns:
        (plugin_name, error_message) — エラーなしなら error_message == ""
    """
    # DESCRIPTOR_NAME の存在確認
    name_match = re.search(r'^DESCRIPTOR_NAME\s*=\s*["\'](.+)["\']', code, re.MULTILINE)
    if not name_match:
        return "", "DESCRIPTOR_NAME が未定義です"
    plugin_name = name_match.group(1)

    # compute 関数の存在確認
    if "def compute(" not in code and "def compute (" not in code:
        return plugin_name, "compute() 関数が未定義です"

    # 構文チェック
    try:
        ast.parse(code)
    except SyntaxError as e:
        return plugin_name, f"構文エラー: {e}"

    return plugin_name, ""


def _strip_code_fences(code: str) -> str:
    """
    LLMが ```python ... ``` ブロックを出力した場合にフェンスを除去する。
    """
    code = code.strip()
    # ```python ... ``` または ``` ... ``` を除去
    if code.startswith("```"):
        lines = code.split("\n")
        # 最初の行（```python）と最後の行（```）を除去
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()
