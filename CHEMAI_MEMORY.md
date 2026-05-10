# ChemAI プロジェクト記憶ファイル
# このファイルはAIが毎会話開始時に必ず読む。ライブラリではなくrepoに保存。
# 変更時は必ずgit commitに含めること。

## ⚠️ 永続ルール（絶対に守ること）

1. **実装済み機能は絶対に削除しない** - UIを再構成する際も、既存の render_* 呼び出しを全て維持する
2. **変更前に必ず `python scripts/audit_ui_features.py` を実行**して機能喪失がないか確認する
3. **タブ構造を変えるときは特に注意** - ネストが変わるとコンテンツが消えて見える
4. **インデント修正スクリプトを使う前にバックアップを取る**

---

## 📐 UIタブ構造（現在の正確な構造）

```
外側タブ (main_tabs):
  ├── 📂 データ設定 (data_tab)
  │     内側タブ:
  │       ├── 📂 データ読込    → _render_data_load()
  │       ├── 🏷️ 列の役割      → _render_column_roles()
  │       ├── ⚗️ SMILES特徴量  → _render_smiles_features()
  │       │     └── カスタムプラグイン → _render_custom_plugins() [descriptor_plugins_ui.py]
  │       └── 📊 EDA           → _render_eda()
  │
  ├── 🔬 EDA (eda_tab)           → render_eda_panel()
  ├── ⚙️ 設定 (pipeline_tab)
  │     ├── render_leakage_check_panel()
  │     ├── render_cv_config()        ← ★ここを消したことがある。絶対消すな
  │     └── render_pipeline_config()
  ├── 📊 結果確認 (results_tab)   → render_results_tab()
  ├── 🔮 逆解析 (inverse_tab)    → render_inverse_analysis_tab()
  └── 🧪 実験計画 (doe_tab)      → render_doe_tab()
```

---

## ✅ 実装済み機能一覧

### バックエンド
- `backend/automl/` - AutoMLエンジン (ランダム/グリッド/ベイズ/GA)
- `backend/cv_manager.py` - CV管理 (KFold/StratifiedKFold/GroupKFold/LOO/etc.)
- `backend/leakage_detector.py` - データリーク検出
- `backend/type_detector.py` - 列型自動判定
- `backend/preset_manager.py` - 設定プリセット保存/読込/YAML export
- `backend/chem/` - 14種SMILESエンジン:
  - RDKit, Mordred, GroupContrib, DescriptaStorus, MolAI
  - scikit-FP, UMA, Mol2Vec, PaDEL, Molfeat
  - XTB, UniPKa, COSMO-RS, Chemprop
- `backend/chem/descriptors/` - プラグイン型記述子システム (_builtins/ + custom/)
- `backend/llm/` - LLM連携インフラ:
  - `provider.py` - 抽象基底クラス + StubLLMProvider
  - `generator.py` - コード生成 + セキュリティ検証 + 保存
  - `reviewer.py` - **LLMコードレビュワー** (PASS/WARN/FAIL + 問題リスト)
  - `providers/hf_provider.py` - **HuggingFaceローカル推論プロバイダー**
    - モデルカタログ: Qwen2.5-Coder-1.5B/7B, Gemma-3-1B, Phi-4-Mini, IBM Granite
    - トークン設定: `.hf_config.json` (gitignore済み)
    - snapshot_download → load_model → generate のパイプライン
- `backend/doe/` - 実験計画法 (D/E/I最適, 直交表L4〜L27)

### フロントエンド
- `descriptor_plugins_ui.py` - SMILES特徴量UI全体
  - エンジン別ON/OFF切替
  - 記述子セット管理（推奨プリセット3種）
  - カスタムプラグイン管理（登録/表示/削除）
  - 既存アダプタコピー機能（RDKit/XTB/COSMO-RS等）
  - AI記述子生成（実験的・折りたたみ表示）← 目立たなくした
    - 外部AI（ChatGPT/Copilot）: プロンプト生成 + コード貼付 + 検証保存 + **LLMレビュー**
    - 内部AI（HuggingFace）: トークン設定 + モデルDL + コード生成 + **LLMレビュー**
- `internal_llm_ui.py` - **HuggingFace LLM設定・推論UI** (新規2026-03-30)
- `cv_config_ui.py` - 交差検証設定（方法/分割数/詳細パラメータ）
- `pipeline_config_ui.py` - 前処理/パイプライン設定
- `leakage_check_ui.py` - データリーク検出UI
- `eda_panel.py` - EDA（分布/相関/欠損値）
- `results_tab.py` - 結果表示（Sakurai Method UIアニメーション込み）
- `inverse_analysis_tab.py` - 逆解析
- `doe_tab.py` - 実験計画法（D/E/I最適 + 直交表, 2モード）

---

## ⚡ ユーザーの強い要望（必ず守る）

- **EDAは解析前後で両方表示する**（内側タブにも外側タブにも）
- **「AI で記述子を生成」は実験的機能として目立たなく**（アコーディオン非展開）
- **全4401記述子デフォルト選択は禁止** → 実務では数十〜数百件しかデータがない
- **勝手に機能を削除しない。UIリファクタ時も既存機能を必ず維持する**
- **回答は必ず日本語**

---

## 🔧 開発環境

- サーバー: `python frontend_nicegui/main.py` → http://localhost:8085
- ポート競合時: `netstat -ano | findstr :8085` でPIDを確認してkill
- 機能監査: `python scripts/audit_ui_features.py`

---

## 📝 最終確認日
2026-03-30

## 🆕 最近の変更（2026-03-30）
- HuggingFace ローカルLLMプロバイダー追加 (`backend/llm/providers/hf_provider.py`)
- LLMコードレビュワー追加 (`backend/llm/reviewer.py`) - 生成コードを第2LLMが査読
- 内部AI UIコンポーネント追加 (`frontend_nicegui/components/internal_llm_ui.py`)
- 全4401記述子フォールバック禁止 + 過学習リスク警告バナー追加
- CV設定の設定タブへの復元
