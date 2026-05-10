"""
backend/llm/provider.py

LLMプロバイダーの抽象基底クラスと組込みスタブ実装。

新しいLLMバックエンド（OpenAI, Anthropic, ローカルLLM等）を追加するには:
  1. LLMProvider を継承したクラスを作成
  2. generate() メソッドを実装
  3. __init__.register_llm_provider("myname", MyProvider) で登録

設計原則:
  - StatelessなI/F（状態を持たない）
  - エラーはLLMProviderError例外で統一
  - 生成コードはPython文字列として返すのみ（実行判断はGeneratorが行う）
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """LLMプロバイダー共通例外。"""


@dataclass
class LLMRequest:
    """LLMへのリクエスト定義。"""
    user_prompt: str
    system_prompt: str = ""
    max_tokens: int = 2048
    temperature: float = 0.2
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """LLMからのレスポンス定義。"""
    content: str                   # 生成されたテキスト（Pythonコード等）
    model: str = ""                # 使用されたモデル名
    tokens_used: int = 0           # 消費トークン数
    is_truncated: bool = False     # レスポンスが途中で切れたか
    raw: Any = None                # 生のAPIレスポンス（デバッグ用）


class LLMProvider(ABC):
    """
    LLMプロバイダーの抽象基底クラス。

    全てのプロバイダーはこのクラスを継承し、generate()を実装する。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """このプロバイダーの識別名。"""

    @property
    def is_available(self) -> bool:
        """
        このプロバイダーが現在利用可能かどうか。
        APIキーの有無・ライブラリのインストール状況等を確認。
        サブクラスでオーバーライド推奨。
        """
        return True

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        LLMにリクエストを送り、レスポンスを返す。

        Args:
            request: LLMRequest オブジェクト

        Returns:
            LLMResponse オブジェクト

        Raises:
            LLMProviderError: APIエラー・接続エラー等
        """

    def generate_descriptor_code(
        self,
        user_description: str,
        *,
        additional_context: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """
        記述子計算コードを生成するための高レベルI/F。

        Args:
            user_description: ユーザーが望む記述子の説明（日本語可）
            additional_context: 既存プラグインの例等の追加コンテキスト
            max_tokens: 最大トークン数

        Returns:
            Pythonコード文字列（検証前の生成物）

        Raises:
            LLMProviderError: 生成に失敗した場合
        """
        system_prompt = _DESCRIPTOR_SYSTEM_PROMPT
        if additional_context:
            system_prompt += f"\n\n## 参考：既存プラグインの例\n{additional_context}"

        request = LLMRequest(
            user_prompt=user_description,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        response = self.generate(request)
        return response.content


# ── システムプロンプト（記述子生成専用） ──────────────────────────────────────
_DESCRIPTOR_SYSTEM_PROMPT = """\
あなたはChemAI ML Studioの記述子プラグインを生成する専門家です。

## 必須ルール
以下の形式のPythonコードを**コードブロックなしで**生成してください：

DESCRIPTOR_NAME = "記述子の識別名（英語）"
DESCRIPTOR_CATEGORY = "カテゴリ（例: 物理化学, 電子状態, トポロジー）"
DESCRIPTOR_ENGINE = "使用エンジン（例: RDKit, XTB, カスタム）"
DESCRIPTOR_DESCRIPTION = "この記述子の日本語説明"
MULTI_DESCRIPTOR = True  # 複数の記述子を返す場合のみ

def compute(smiles_list: list[str]) -> list[float | None]:
    '''記述子の計算'''
    import pandas as pd
    from rdkit import Chem
    
    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
                continue
            # 計算ロジックをここに記述
            value = ...
            results.append(float(value))
        except Exception:
            results.append(None)
    return results

## 制約
- 外部APIや危険なシステムコール（os.system, subprocess等）は使用禁止
- 計算に失敗した分子は None を返す（例外を握りつぶす）
- RDKitが利用可能な前提で記述
- コードブロック(```python)は付けない
- 説明文は不要。コードのみ出力する
"""


# ── スタブ実装（開発・テスト用） ────────────────────────────────────────────
class StubLLMProvider(LLMProvider):
    """
    スタブ（ダミー）実装。
    LLM APIが設定されていない環境での動作確認用。
    実際のLLM呼び出しは行わず、固定のサンプルを返す。
    - 記述子生成モード: サンプルPythonコードを返す
    - 特徴量選択モード: 選択済み特徴量名のリストを返す
    """

    def __init__(self):
        self._mode = "code"  # "code" or "select"

    def set_mode(self, mode: str) -> None:
        """動作モードを設定（"code"=記述子生成, "select"=特徴量選択）"""
        self._mode = mode

    @property
    def name(self) -> str:
        return "stub"

    @property
    def is_available(self) -> bool:
        return True  # スタブは常に利用可能

    def generate(self, request: LLMRequest) -> LLMResponse:
        logger.info("[StubLLM] generate() called (mode=%s)", self._mode)
        if self._mode == "select":
            # 特徴量選択モード: サンプル選択結果を返す
            content = _generate_stub_selection(request.user_prompt)
        else:
            # 記述子生成モード: サンプルコードを返す
            content = _generate_stub_code(request.user_prompt)
        return LLMResponse(
            content=content,
            model="stub-v1",
            tokens_used=0,
            is_truncated=False,
        )


def _generate_stub_code(user_prompt: str) -> str:
    """スタブ用サンプルコードを返す。"""
    return f'''\
DESCRIPTOR_NAME = "AI生成記述子（スタブ）"
DESCRIPTOR_CATEGORY = "カスタム"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = """
ユーザーリクエスト: {user_prompt[:100]}
（注意: これはスタブ実装です。実際のLLMを接続するとリアルなコードが生成されます）
"""

def compute(smiles_list: list[str]) -> list[float | None]:
    """分子量を返すサンプル実装（スタブ）。"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
                continue
            results.append(float(Descriptors.MolWt(mol)))
        except Exception:
            results.append(None)
    return results
'''


def _generate_stub_selection(user_prompt: str) -> str:
    """スタブ用の特徴量選択結果を返す。"""
    # ユーザープロンプトからカテゴリ名を抽出して、適当なサンプルを返す
    if "Fingerprint" in user_prompt:
        return "選択特徴量: fingerprint"
    elif "Quantum" in user_prompt:
        return "選択特徴量: HOMO_Energy, LUMO_Energy, HOMO_LUMO_Gap, Polarizability, DipoleMoment"
    elif "Vibrational" in user_prompt:
        return "選択特徴量: Freq_1, Freq_2, Freq_3"
    elif "RDKit" in user_prompt:
        return "選択特徴量: MolWt, MolLogP, TPSA, NumHAcceptors, NumHDonors"
    elif "Mordred" in user_prompt:
        return "選択特徴量: Mordred_1, Mordred_2, Mordred_3"
    elif "Physical" in user_prompt:
        return "選択特徴量: Octanol_Water_Partition_Coeff, Surface_Tension, Viscosity"
    else:
        return "選択特徴量: MolWt, Polarizability, DipoleMoment"
