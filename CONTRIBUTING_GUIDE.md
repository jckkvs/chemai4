# ChemAI ML Studio — 開発者拡張ガイド

このドキュメントでは、ChemAI ML Studio に**新機能を追加する方法**を解説します。
化学記述子・MLモデル・前処理・特徴選択・評価/解釈手法のすべてがプラグイン形式で追加可能です。

---

## 📐 全体アーキテクチャ

```
backend/
├── chem/                    # 化学記述子アダプタ（SMILES特徴量）
│   ├── base.py              # BaseChemAdapter（抽象基底クラス）  ← ★拡張ポイント①
│   ├── rdkit_adapter.py     # RDKit 記述子
│   ├── xtb_adapter.py       # XTB 量子化学記述子
│   ├── uma_adapter.py       # Meta UMA (fairchem)
│   ├── mordred_adapter.py   # Mordred 記述子
│   ├── cosmo_adapter.py     # COSMO-RS 溶媒和記述子
│   ├── unipka_adapter.py    # UniPKa pKa予測
│   ├── group_contrib_adapter.py  # 基団寄与法
│   ├── molai_adapter.py     # CNN+PCA 潜在ベクトル
│   └── __init__.py          # 自動検出・安全import
│
├── models/
│   ├── factory.py           # モデルレジストリ（辞書登録）  ← ★拡張ポイント②
│   ├── automl.py            # AutoML実行エンジン
│   ├── linear_tree.py       # FullScratch: LinearTree/Forest/Boost
│   └── rgf.py               # FullScratch: Regularized Greedy Forest
│
├── pipeline/
│   ├── pipeline_builder.py  # パイプライン組立  ← ★拡張ポイント③
│   ├── feature_selector.py  # 特徴選択手法
│   ├── feature_generator.py # 特徴量生成
│   └── col_preprocessor.py  # 前処理
│
├── interpret/
│   ├── shap_explainer.py    # SHAP 分析  ← ★拡張ポイント④
│   └── sri.py               # SHAP Interactions
│
└── data/
    ├── preprocessor.py      # データ前処理
    ├── feature_engineer.py  # 特徴エンジニアリング
    └── eda.py               # 探索的データ分析
```

---

## ★ 拡張ポイント① 化学記述子アダプタの追加

### 手順（3ステップ）

#### Step 1: アダプタクラスを作成

`backend/chem/my_adapter.py` を作成し、`BaseChemAdapter` を継承:

```python
"""backend/chem/my_adapter.py — カスタム記述子アダプタ"""
from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult
import pandas as pd

class MyAdapter(BaseChemAdapter):

    @property
    def name(self) -> str:
        return "my_engine"   # ユニークな識別子

    @property
    def description(self) -> str:
        return "カスタム記述子エンジン。pip install my-package が必要。"

    def is_available(self) -> bool:
        try:
            import my_package  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        # ここで実際の記述子計算を実装
        rows, failed = [], []
        for i, smi in enumerate(smiles_list):
            try:
                row = {"my_desc_1": 1.0, "my_desc_2": 2.0}  # 計算結果
                rows.append(row)
            except Exception:
                failed.append(i)
                rows.append({"my_desc_1": float("nan"), "my_desc_2": float("nan")})
        return DescriptorResult(
            descriptors=pd.DataFrame(rows),
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(name="my_desc_1", meaning="説明1", is_count=False),
            DescriptorMetadata(name="my_desc_2", meaning="説明2", is_count=True),
        ]
```

#### Step 2: `__init__.py` に登録

`backend/chem/__init__.py` に追加:

```python
try:
    from backend.chem.my_adapter import MyAdapter
except Exception:
    MyAdapter = _make_unavailable_adapter("MyAdapter")
```

`__all__` にも `"MyAdapter"` を追加。

#### Step 3: UI に表示（任意）

`frontend_streamlit/app.py` のエンジンリスト `_lib_info` に追加:

```python
{"key": "use_my_engine", "name": "MyEngine", "icon": "🔬", "cost": "🟡", "cost_label": "中",
 "dims": "~10種", "desc": "カスタム記述子エンジン。",
 "adapter": MyAdapter(), "auto_on": False},
```

事前計算にも対応させる場合は `smiles_transformer.py` の `_engine_adapters` 辞書にも追加:

```python
"use_my_engine": ("MyEngine", "backend.chem.my_adapter", "MyAdapter", {}),
```

### テンプレ（コピペ用）

既存アダプタの参考:
- **最小限**: `group_contrib_adapter.py`（~100行）
- **標準的**: `rdkit_adapter.py`（~300行）
- **外部プロセス呼出**: `xtb_adapter.py`（~375行）
- **MLモデル呼出**: `uma_adapter.py`（~240行）

---

## ★ 拡張ポイント② MLモデルの追加

### 手順（1ステップ ← 辞書に追加するだけ）

`backend/models/factory.py` のレジストリ辞書に追加:

```python
# _REGRESSION_REGISTRY に追加
"my_model": {
    "name": "My Custom Model",       # GUI表示名
    "class": MyModelRegressor,        # sklearn互換クラス
    "default_params": {"alpha": 1.0}, # デフォルトパラメータ
    "available": True,                # 常時利用可能
    "tags": ["linear", "custom"],     # フィルタ用タグ
},
```

#### factory パターン（遅延import用）

外部ライブラリのモデルの場合:

```python
def _my_model_regressor(**kw):
    from my_package import MyModelRegressor
    return MyModelRegressor(**kw)

"my_model": {
    "name": "My Custom Model",
    "factory": _my_model_regressor,    # classの代わりにfactory
    "default_params": {"alpha": 1.0},
    "available": is_available("my_package"),
    "tags": ["custom"],
},
```

#### 分類モデルも追加する場合

`_CLASSIFICATION_REGISTRY` にも同様に追加。キーの末尾に `_c` をつける慣習:

```python
"my_model_c": { ... }
```

#### AutoML デフォルトに含める場合

`get_default_automl_models()` の `regression_defaults` / `classification_defaults` リストにキーを追加。

### 必要な条件

モデルクラスは **scikit-learn 互換** であること:
- `fit(X, y)` / `predict(X)` を実装
- `get_params()` / `set_params()` を実装
- `BaseEstimator` を継承するのが最も簡単

---

## ★ 拡張ポイント③ 前処理・特徴選択の追加

### 前処理（Scaler / Encoder）

`backend/pipeline/pipeline_builder.py` の `_build_preprocessor()` を参照。

新しいスケーラーを追加する場合:
```python
# pipeline_builder.py の SCALER_MAP に追加
SCALER_MAP = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "minmax":   MinMaxScaler,
    "my_scaler": MyCustomScaler,   # ← 追加
}
```

### 特徴選択

`backend/pipeline/feature_selector.py` に新しい特徴選択手法を追加:

```python
# SELECTOR_MAP に追加
SELECTOR_MAP = {
    "select_k_best":  SelectKBest,
    "boruta":         BorutaSelector,
    "my_selector":    MyCustomSelector,  # ← 追加
}
```

---

## ★ 拡張ポイント④ 評価・解釈手法の追加

### 解釈性手法

`backend/interpret/` に新しいモジュールを追加:

```python
# backend/interpret/my_explainer.py
def explain(model, X, y=None, **kwargs):
    """
    Returns:
        dict with keys like 'fig' (matplotlib/plotly figure), 'data' (DataFrame), etc.
    """
    result = compute_my_explanation(model, X)
    return {"fig": fig, "data": result_df}
```

UIに表示する場合は `frontend_streamlit/components/interpretability_ui.py` にタブを追加:

```python
# interpretability_ui.py の render_interpretability_ui() 内
with tabs[n]:
    from backend.interpret.my_explainer import explain
    result = explain(model, X)
    st.plotly_chart(result["fig"])
```

### 評価指標

`backend/models/automl.py` の `_evaluate_model()` で使用する指標を追加:

```python
from sklearn.metrics import my_custom_scorer
```

---

## 🧪 テストの書き方

### 化学アダプタのテスト

```python
# tests/test_my_adapter.py
from backend.chem.my_adapter import MyAdapter

class TestMyAdapter:
    def test_name(self):
        assert MyAdapter().name == "my_engine"

    def test_is_available(self):
        assert isinstance(MyAdapter().is_available(), bool)

    def test_compute(self):
        adapter = MyAdapter()
        if adapter.is_available():
            result = adapter.compute(["CCO", "c1ccccc1"])
            assert result.descriptors.shape[0] == 2
```

### モデルのテスト

```python
from backend.models.factory import get_model, list_models

def test_my_model_creation():
    model = get_model("my_model", task="regression")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
```

テスト実行: `python -m pytest tests/ -q`

---

## 📁 ファイル命名規則

| 種類 | ファイル名パターン | 例 |
|------|---------------------|-----|
| 化学アダプタ | `backend/chem/{name}_adapter.py` | `uma_adapter.py` |
| MLモデル | `backend/models/{name}.py` | `linear_tree.py` |
| 解釈手法 | `backend/interpret/{name}.py` | `shap_explainer.py` |
| UIコンポーネント | `frontend_streamlit/components/{name}_ui.py` | `charge_config_ui.py` |
| テスト | `tests/test_{name}.py` | `test_uma_adapter.py` |

---

## 🔌 安全import パターン

ライブラリが未インストールでもアプリが起動するよう、全てのオプショナル依存は安全importを使用:

```python
# backend/utils/optional_import.py
from backend.utils.optional_import import safe_import, is_available

_my_lib = safe_import("my_package", "my_package")
_available = is_available("my_package")
```

---

## チェックリスト（新機能追加時）

- [ ] 抽象基底クラスまたはレジストリに準拠しているか
- [ ] `is_available()` で未インストール時に `False` を返すか
- [ ] `__init__.py` に安全importで登録したか
- [ ] テストを作成したか（モック使用可）
- [ ] UIに表示する場合、`app.py` のリストに追加したか
- [ ] `requirements.txt` / `environment.yml` に依存を追加したか

---

## ★ 拡張ポイント⑤ リーケージ検出手法の追加

### 新しい類似度推定手法の追加

`backend/data/leakage_detector.py` に新しい類似度関数を追加:

```python
def compute_my_similarity(X: np.ndarray, **kwargs) -> np.ndarray:
    """カスタム類似度行列を計算。
    Returns: (n_samples, n_samples) の対称行列、値域 [0, 1]
    """
    S = ...  # 計算ロジック
    return S
```

`detect_leakage()` の method 分岐に追加:

```python
elif method == "my_method":
    S = compute_my_similarity(X_scaled, **kwargs)
```

### UIへの追加

`leakage_check_ui.py` の手法選択 selectbox に選択肢を追加するだけ。
