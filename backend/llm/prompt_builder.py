"""
backend/llm/prompt_builder.py

外部LLM（ChatGPT, Copilot, Claude等）に渡すプロンプトを生成するモジュール。

役割:
  - ユーザーの「やりたいこと」をプラグイン作成プロンプトに変換
  - 外部AIが形式を間違えないよう、完全な仕様+例をプロンプトに含める
  - 生成されたコードをアプリに貼り付けて使えるよう検証もサポート

設計方針:
  - 1特徴量 = 1関数 を徹底（RDKitのDescriptors.descListと同じパターン）
  - compute()は内部で各関数を呼び出してDataFrameにまとめる
  - 外部AIには「各関数を個別に定義 → compute()でまとめる」パターンを強制
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DescriptorIntent:
    """ユーザーの記述子作成意図を表すデータクラス。"""
    library: str = ""          # 使用ライブラリ（例: rdkit, mordred, padelpy）
    what_to_calc: str = ""     # 計算したい物性（例: 分子量、LogP、HOMO-LUMOギャップ）
    output_type: str = "single"  # "single"=1値, "multi"=複数値(1特徴量1関数)
    extra_notes: str = ""      # 追加の注意事項・制約

    @property
    def is_valid(self) -> bool:
        """最低限の情報が揃っているか。"""
        return bool(self.what_to_calc.strip())


def build_external_llm_prompt(intent: DescriptorIntent) -> str:
    """
    外部LLM（ChatGPT等）に渡すプロンプトを生成する。

    Args:
        intent: ユーザーの意図

    Returns:
        外部AIに貼り付けるための完全なプロンプト文字列
    """
    library_note = f"使用ライブラリ: **{intent.library}**" if intent.library else ""
    multi_note = _MULTI_OUTPUT_NOTE if intent.output_type != "single" else ""

    prompt = (_PROMPT_TEMPLATE
              .replace("{what_to_calc}", intent.what_to_calc.strip())
              .replace("{library_note}", library_note)
              .replace("{multi_note}", multi_note)
              .replace("{extra_notes}", intent.extra_notes.strip() or "特になし"))
    return prompt.strip()


# ── プロンプトテンプレート ────────────────────────────────────────────────────

_PLUGIN_SPEC = '''\
## 📋 プラグインファイルの仕様

以下のモジュールレベル定数と関数群を持つ Python ファイルを作成してください。

### 必須定数
```python
DESCRIPTOR_NAME = "記述子の英語識別名"     # 必須: 他と重複しない短い名前
DESCRIPTOR_CATEGORY = "カテゴリ名"          # 必須: 例 "物理化学", "電子状態", "トポロジー"
DESCRIPTOR_ENGINE = "エンジン名"            # 必須: 例 "RDKit", "PaDEL", "カスタム"
DESCRIPTOR_DESCRIPTION = "この記述子の説明" # 推奨: 日本語でOK
```

### ⚠️ 最重要ルール: 1特徴量 = 1関数

**RDKitのDescriptors.descListと同じ設計パターン**を使ってください。

複数の特徴量を計算する場合でも、**1つの関数が1つのスカラー値を返す**ように設計してください。
DataFrameを直接返す関数は禁止です。

各関数のシグネチャ:
```python
def calc_特徴量名(mol) -> float | None:
    """1分子に対して1つの特徴量値を返す。"""
    ...
```

- 引数 `mol` はライブラリ固有の分子オブジェクト（例: RDKitなら `Chem.Mol`）
- 戻り値は `float` または計算失敗時に `None`
- 関数名は `calc_` プレフィクスで統一

### compute() 関数（必須・エントリポイント）

`compute()` は呼び出し側のエントリポイントです。
内部で各 `calc_*` 関数を呼び出し、結果を `pd.DataFrame` にまとめます。

```python
# 関数レジストリ: (列名, 計算関数) のリスト
DESCRIPTOR_FUNCTIONS = [
    ("FeatureName1", calc_feature1),
    ("FeatureName2", calc_feature2),
    # ... 特徴量ごとに1行追加
]

def compute(smiles_list: list[str]) -> "pd.DataFrame":
    """全記述子を計算してDataFrameで返す。"""
    import pandas as pd
    from rdkit import Chem  # ← ライブラリに応じて変更

    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)  # ← ライブラリに応じて変更
            if mol is None:
                rows.append({name: None for name, _ in DESCRIPTOR_FUNCTIONS})
                continue
            row = {}
            for name, func in DESCRIPTOR_FUNCTIONS:
                try:
                    row[name] = func(mol)
                except Exception:
                    row[name] = None
            rows.append(row)
        except Exception:
            rows.append({name: None for name, _ in DESCRIPTOR_FUNCTIONS})
    return pd.DataFrame(rows)
```

### ⚠️ 禁止事項
- `os.system()`, `subprocess`, `os.popen()` など外部プロセス実行**禁止**
- `eval()`, `exec()`, `compile()` **禁止**
- `open()` などファイルI/O **禁止**（ライブラリI/Oは除く）
- **1つの関数で複数の値を返す（dictやDataFrameを直接返す）のは禁止**
- エラー時は例外をraiseするのではなく `None` を返すこと

### ✅ 推奨事項
- 1関数が1つのスカラー値（float）のみを返す設計を徹底する
- DESCRIPTOR_FUNCTIONS に全テーブルを定義し、追加・削除を1行で管理可能にする
- 各 `calc_*` 関数は独立してテスト可能なように、副作用のない純粋関数にする
- RDKit を使う場合は `Chem.MolFromSmiles(smi)` でMolオブジェクトを作成し、`None` チェックを行う
- NumPy/pandas のインポートはループ外で行う（パフォーマンス向上）
'''

_EXAMPLE_SINGLE = '''\
## 💡 実装例（1特徴量の場合: RDKit で分子量を計算）

```python
DESCRIPTOR_NAME = "MolWeight"
DESCRIPTOR_CATEGORY = "物理化学"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = "分子量を計算します"

def calc_mol_weight(mol) -> float | None:
    """分子量を計算する。"""
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(mol)

DESCRIPTOR_FUNCTIONS = [
    ("MolWeight", calc_mol_weight),
]

def compute(smiles_list: list[str]) -> list[float | None]:
    """記述子の計算（1値のみ）。"""
    from rdkit import Chem
    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
                continue
            results.append(calc_mol_weight(mol))
        except Exception:
            results.append(None)
    return results
```
'''

_EXAMPLE_MULTI = '''\
## 💡 実装例（複数特徴量の場合: RDKit で分子量・LogP・TPSAを計算）

**重要: 1特徴量 = 1関数 のパターンを厳守してください。**

```python
DESCRIPTOR_NAME = "BasicPhysChem"
DESCRIPTOR_CATEGORY = "物理化学"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = "分子量・LogP・TPSAを個別関数で計算します"
MULTI_DESCRIPTOR = True

# ── 各特徴量を個別関数として定義 ──

def calc_mol_weight(mol) -> float | None:
    """分子量(Da)を計算する。"""
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(mol)

def calc_logp(mol) -> float | None:
    """Wildman-Crippen LogP（油水分配係数）を計算する。"""
    from rdkit.Chem import Descriptors
    return Descriptors.MolLogP(mol)

def calc_tpsa(mol) -> float | None:
    """Topological Polar Surface Area（極性表面積, Å²）を計算する。"""
    from rdkit.Chem import Descriptors
    return Descriptors.TPSA(mol)

def calc_hba(mol) -> float | None:
    """水素結合受容体数を計算する。"""
    from rdkit.Chem import Descriptors
    return float(Descriptors.NumHAcceptors(mol))

def calc_hbd(mol) -> float | None:
    """水素結合供与体数を計算する。"""
    from rdkit.Chem import Descriptors
    return float(Descriptors.NumHDonors(mol))

# ── 関数レジストリ: (列名, 関数) のリスト ──
# ここに追加するだけで新しい特徴量を追加できる
DESCRIPTOR_FUNCTIONS = [
    ("MolWeight", calc_mol_weight),
    ("LogP",      calc_logp),
    ("TPSA",      calc_tpsa),
    ("HBA",       calc_hba),
    ("HBD",       calc_hbd),
]

# ── エントリポイント ──
def compute(smiles_list: list[str]) -> "pd.DataFrame":
    """全記述子を計算してDataFrameで返す。"""
    import pandas as pd
    from rdkit import Chem

    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({name: None for name, _ in DESCRIPTOR_FUNCTIONS})
                continue
            row = {}
            for name, func in DESCRIPTOR_FUNCTIONS:
                try:
                    row[name] = func(mol)
                except Exception:
                    row[name] = None
            rows.append(row)
        except Exception:
            rows.append({name: None for name, _ in DESCRIPTOR_FUNCTIONS})
    return pd.DataFrame(rows)
```

### 📌 この設計のメリット
1. **テスト容易性**: 各 `calc_*` 関数を個別にユニットテストできる
2. **保守性**: 特徴量の追加/削除は `DESCRIPTOR_FUNCTIONS` を1行編集するだけ
3. **RDKit互換**: `Descriptors.descList` と同じ `(name, function)` パターン
4. **デバッグ性**: どの特徴量でエラーが起きたか即座に特定可能
'''

_MULTI_OUTPUT_NOTE = """\

### 📌 注意: 複数の特徴量を返す場合
**1特徴量 = 1関数** のパターンを厳守してください。
`MULTI_DESCRIPTOR = True` を追加し、各特徴量を個別の `calc_*` 関数として定義してください。
`DESCRIPTOR_FUNCTIONS = [(列名, 関数), ...]` でレジストリを作り、`compute()` でまとめてください。
"""

_PROMPT_TEMPLATE = """\
# ChemAI ML Studio 用 SMILES 記述子プラグイン作成依頼

あなたは化学情報処理の専門家として、ChemAI ML Studio のカスタム記述子プラグインを作成してください。

## 🎯 作成したい記述子
{what_to_calc}

{library_note}
{multi_note}

## 📝 追加の要件・制約
{extra_notes}

---

{plugin_spec}

---

{example_single}

---

{example_multi}

---

## ✏️ 出力形式
- **Pythonコードのみ**を出力してください（説明文・コードブロック記号 ``` は不要）
- コードは上記仕様に従い、`DESCRIPTOR_NAME`, `DESCRIPTOR_CATEGORY`, `DESCRIPTOR_ENGINE`, `DESCRIPTOR_FUNCTIONS`, `compute()` を必ず含めてください
- **1特徴量 = 1関数 (`calc_*`)** のパターンを厳守してください
- **複数の特徴量を1つの関数でまとめて返すのは禁止**です
- コードをそのまま `.py` ファイルとして保存すれば動作するよう、完全な実装にしてください
""".replace("{plugin_spec}", _PLUGIN_SPEC).replace(
    "{example_single}", _EXAMPLE_SINGLE
).replace(
    "{example_multi}", _EXAMPLE_MULTI
)
