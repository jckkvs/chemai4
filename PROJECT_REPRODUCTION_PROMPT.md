# ChemAI ML Studio: Ultimate Master Reproduction Prompt (Absolute Full-Specification Edition)

以下のすべての機能、コンポーネント、パラメータ設定、計算科学的アルゴリズム原理、生成AI拡張要件、およびUI/UXの設計を「一行残らず」一切の省略なく、かつダミー実装（`pass`、`NotImplementedError`、`TODO`）を厳格に禁止した上でゼロから完全な実動システムとして出力・再現せよ。

**【重要最優先事項】**
本プロンプトは、プロジェクトの**すべての機能をすべての行レベルで完全に再現する**ための究極の仕様書である。あなた（AI）は、機能や細部を勝手に省略（サボり）してはならない。以下のリストは私たちがこれまで実装してきた機能群であり、1行残らず実装し切ることが最初の義務である。

---

## ⛔ 絶対遵守のコア要件 (Definition of Done API & Code)

1. **ベースラインの完全再現**: 本プロンプトの内容はすべて確実に実装すること。
2. **ダミー実装・未実装の禁止**: 実ロジックを必ず一字一句記述すること。
3. **ユニットテストの強制**: 全モジュールに `pytest` を用いたテスト（`@pytest.mark.parametrize` や Hypothesis を駆使した境界値テストを含む）を書き、行カバレッジ90%以上、分岐カバレッジ (`--cov-branch`) 75%以上を達成すること。

---

## 1. 🎨 UI/UX とシステムアーキテクチャ (NiceGUI)

### 1.1 マイクロインタラクションと状態管理
- **アプリケーション初期化**: `ui.dark_mode().enable()`。カラーコードは `primary='#1e1e2f', secondary='#2d2d44', accent='#00d4ff'`。
- **解析開始ボタン**: `.classes('btn-run-analysis')`。CSS `@keyframes pulse-glow`（青 `#00d4ff` と紫 `#7b2ff7` が2秒周期で明滅）。クリックで `play_arrow` → `hourglass_empty` や `sync` に切り替え、非同期でスピナーを回す。
- **エラーハンドリング**: `ui.notify('Msg', type='negative', position='top-right')` と共に、親コンテナに `.classes('animate-shake')`（0.6秒間translateX(-4px/4px)を5回反復）を一時適用（1秒後にJSで削除）。
- **フローティングステータス (`descriptor_status_bar.py`)**: SMILES解析の進行状況を示すため画面下部に固定（Fixed）されたインジケーターを `ui.timer` (0.5s間隔)で駆動し、今どのエンジンが走っているかをプログレスバーと共に表示。

### 1.2 コグニティブロードの最小化
- **色覚多様性（CVD）対応**: 赤緑の色に依存せず、すべての状態（成功/警告/エラー/情報）に必ずハードコードでアイコン（✅/⚠️/❌/ℹ️）を接頭辞として付与する。
- **フォント**: `Noto Sans JP` を `ui.add_head_html` 経由で全体適用（`size: 13px/0.85rem` 以上）。

### 1.3 高度なコンポーネント (AgGridと表示パネル)
- **AgGridの極限利用**: `results_tab.py` やデータテーブルでは `ui.aggrid` を利用し、`pagination: true`, `paginationPageSize: 20`, フィルタリング可、ソート可。さらに残差（Residuals）の特定のセルには背景色を動的に変えるJavaScriptセルスタイリングを注入する。
- **EDA統合パネル (`eda_panel.py`)**: プロジェクト初期の視覚的理解のため、Plotlyによる目的変数のヒストグラム（Distplot）、変数間の相関ヒートマップ行列、ペアプロット図、およびMissing Value Heatmap（欠損値の分布状況）を描画する。

---

## 2. 🧪 データ前処理とパイプラインアーキテクチャ

### 2.1 カラムのメタデータ定義 (ColumnMeta)
データセットの各特徴量は `ColumnMeta` データクラスで管理される：
- `monotonic`: 単調性制約(`0`: なし, `1`: 増加, `-1`: 減少, `2`: 自動検出)
- `constraint_strength`: 制約強度 (`None`: デフォルト, `"weak"`: 弱い, `"strong"`: 強い)
- `linearity`: 線形性ヒント (`"unknown"`, `"linear"`, `"nonlinear"`)
- `group`: データのグループ化ID文字列（GroupKFold時に使用）
- `fixed`: 特徴量選択時に**絶対に除外されない（Drop禁止）**ことを保証する保護フラグ。

### 2.2 前処理（ColPreprocessor & FeatureSelector）
`sklearn.compose.ColumnTransformer`。
- **Numeric Scalers**: `standard`, `robust`, `minmax`, `maxabs`, `power_yj` (Yeo-Johnson), `power_bc` (Box-Cox), `quantile_normal` / `quantile_uniform`, `log` (FunctionTransformer np.log1p), `none`.
- **Numeric/Cat Imputers**: `mean`, `median`, `most_frequent`, `constant=0.0`, `knn(5)`, `iterative`. カテゴリ列には `SimpleImputer(strategy="most_frequent")` を自動フォールバック的ルーティング。
- **Categorical Encoders**: `onehot(handle_unknown='ignore')`, `ordinal`, `target` (高カーディナリティ用 TargetEncoder)。
- **FeatureSelector保護**: `SelectKBest`, `VarianceThreshold`, `Boruta`, `Lasso` により変数を落とす際、`ColumnMeta.fixed = True` な列はマスク演算から除外され、必ず出力配列に含まれる構造にする。

### 2.3 交差検証 (CV Recommender) の完全ルール
1. `time_col` がある場合: `TimeSeriesSplit(n_splits=5)`
2. `group_col` や、メタデータに `group` 属性がある場合: `GroupKFold(n_splits=5)` を使用し、テストリークを防ぐ。
3. Classification (分類) かつ、目的変数がintで要素数10以下: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
4. 上記以外: `KFold(n_splits=5, shuffle=True, random_state=42)`

### 2.4 データ漏洩検知 (LeakageChecker)
- `y` (目的変数) との相関が `0.99` 以上の入力変数をターゲットリークとして警告。
- すべての要素が一意なID列（ユニーク数 == 行数）を検知。

---

## 3. 🧠 SMILES結合とケモインフォマティクス

### 3.1 複数SMILESエンジン統合 (Adapter & Cache)
`joblib.Memory` を用いたキャッシュ化にて、以下を `run.io_bound` で非同期並列計算：
1. **RDKit**: 物理化学パラメータ（MolWt, MolLogP等）＋`MorganFingerprintAsBitVect(radius=2, nBits=1024)`.
2. **Mordred**: `mordred.Calculator` (1613種)。NaNは「全サンプルの該当列の平均」で補間。
3. **Skfp**: `ECFPFingerprint`, `MACCSFingerprint`, `MHFPFingerprint` 等の全12種サポート。
4. **DescriptaStorus**: RDKit2DNormalized（200次元）。
5. **HuggingFace**: `repo_id="jckkvs/molai-chem-v1"` の `last_hidden_state.mean(axis=1)`。
取得後 `pandas.concat` 結合し、Spearman相関絶対値 `> 0.95` のペア間冗長特徴量を排除。

### 3.2 混合物加重平均と wt% / mol% 自動変換 (`mix_rules.py`, `smiles_transformer.py`)
混合系（Polymer_A 80wt% 等）の完全対応。
1. **マスタールール**: `backend/chem/mix_rules.py` にて `get_mix_rule(desc_name)` 関数を定義。密度やフラクションは `wt%`、カウント系や物理量は `mol%` で足す辞書。
2. **モル・重量の動的逆算**: カラム接尾辞が `_SMILES_WT` ならば重量、`_SMILES_MOL%` ならばモルと認識。入力比率が `wt%` なのに記述子ルールが `mol%` なら、RDKitで分子量 $M_i$ を算出し $x_i = (w_i/M_i) / \sum(w_j/M_j)$ としてリアルタイムに比率をモルベースへ変換。
3. **加重平均**: $X_{mix} = \sum_{i} (r_i \times X_i)$ 。
4. **マニュアル設定**: パイプラインUIの辞書から特定の特徴量のルールをユーザーが強制オーバライド可能。

---

## 4. 🤖 最適化モデル設定とアルゴリズムの極み

### 4.1 Optuna AutoMLエンジンとアーキテクチャ (`automl.py`, `tuner_pipeline.py`)
- 全体の探索は `timeout=600`（10分）として設定され、`optuna.pruners.MedianPruner` を導入し、学習曲線が悪い試行を早期終了（Pruning）する。
- 目的関数方向は 回帰（R2など）なら `directions=['maximize']`、RMSEなら `['minimize']` に明示的設定。`cv_folds=5`。
- **Linear Models**: `LinearTreeRegressor`/`LinearTreeClassifier`, `LinearForestRegressor`/`Classifier` (線形回帰を葉に持つ決定木 `linear-tree` パッケージ等)。
- **Trees & Boosting**: `XGBoost`, `LightGBM`, `CatBoost`, `RandomForest`, `RGF (Regularized Greedy Forest)`.
- **Others**: `SVR/SVC`, `Ridge/Lasso/ElasticNet/LogReg`, `GaussianProcess` (Kernel: `ConstantKernel * RBF + WhiteKernel`).
各モデルの探索空間パラメータ上限・下限（$n\_estimators$, $learning\_rate$ 等）は広く正確に `search_space_generator.py` に指定。

### 4.2 変数ごとの単調性制約の汎用付与 (Universal Monotonic Constraints)
sklearnツリーモデル以外の**すべてのモデル（SVR, GaussianProcess, Neural Networkなど）において単調性制約を実現する**フルスクラッチ汎用ラッパー `MonotonicConstraintRegressor/Classifier`（`BaseEstimator`, `MetaEstimatorMixin`）。
1. **パターン**: `0`(なし)、`1`(増加)、`-1`(減少)、`2`(自動検出。Spearmanの順位相関係数から1/-1を自動決定)。
2. **強度と外挿保証**: `"weak"`(`penalty_weight=5, n_grid=15, max_iter=2`) / `"strong"`(`penalty_weight=50, n_grid=40, max_iter=8`)。特徴量の平均から `[min - 3σ, max + 3σ]` の外挿範囲で1Dグリッド（`np.linspace`）を引いて検証。
3. **ペナルティサンプル拡張法**: `fit`内で `predict(外挿グリッド)` を実行し、制約に違反する傾きを検知した場所へ「強制的に逆転した目標予測値を持つダミーデータ」を高い `sample_weight` で注入し、満たされるまで `max_iter` 回ループで再学習させる超汎用アルゴリズム。

---

## 5. 🔍 逆解析 (Inverse) と MolAI 直接生成

### 5.1 パラメータ逆解析 (`optimizer.py`, `pareto_front.py`)
- 単目的最適化 (`scipy.optimize.minimize`, `SLSQP`) で予測値とターゲットの差を最小化。
- 多目的最適化 (`pymoo`, `NSGA2`) で解集団を生成。`pareto_front.py` にて、クラウディング距離（Crowding Distance）などを加味し非劣解（Non-dominated sorting）を抽出し Plotly の Scatter グラフを描画。
- 制約クラス (`ConstraintSumToTotal`, `ConstraintRange`, `ConstraintRatio`) をペナルティ実装。

### 5.2 構造直接生成: Generative Inverse (`molai_generator.py`)
逆算された「理想の潜在特徴量ベクトル」を `jckkvs/molai-chem-gen-v1` 的な生成モデル（`AutoModelForSeq2SeqLM/CausalLM`）に入力し、デコーダを `beam_search(num_beams=5)` あるいは `temperature=0.7` なサンプリングで回して、直接SMILES文字列を出力するSF的機能。

### 5.3 実験計画法 (DoE: ベイズ最適化)
`Expected Improvement (EI)`, `Probability of Improvement (PI)`, `Upper Confidence Bound (UCB, kappa=1.96)` を実装。DoEタブ上でPlotly散布図を描画し、**散布図中の点をクリックすると対象のパラメータがUIの入力欄に自動反映されるインタラクティブ機能**を実装。

---

## 6. 📊 解釈パネル、生成AI、そして永続化

### 6.1 高度なモデル解釈タブ (SHAP, SAGE, SRI)
- **残差グラフ**: X:実測値, Y:予測値 + ヒストグラム。
- **SHAP 4種**: Summary (Bar), Beeswarm, Waterfall, Dependence.
- **SAGE (Shapley Additive Global Importance)**: モデルの損失関数ベースの変数重要度。
- **SRI 分解 (Synergy, Redundancy, Independence)**: 情報理論に基づく交互作用・冗長性をヒートマップ可視化。

### 6.2 生成AI (LLM) OpenAI クライアント連動
1. **自動分析レポートジェネレータ**: RMSE等の指標とSHAP重要度配列をJSONでLLMへ送信し、「予測の妥当性と化学的考察」を300文字で取得。
2. **特徴量エンジニアリング支援ボット**: 「この目的変数を予測するための有用なPandas合成特徴量式を3つ提案せよ」というプロンプトを投げ、コードブロックを抽出してUIにボタン化。

### 6.3 エクスポートとMLflow的バージョン管理 (`version_manager.py`)
- **PDF/Word/Jupyter Export**: `pdf_exporter.py` / `word_exporter.py` / そして「ローカルで即座にモデルを実行できる独立した .ipynb ファイルコード」を生成する `notebook_exporter.py`。
- **モデル永続化 (Pickle)**: 作成されたパイプラインインスタンスをシリアライズし、「Download Pickle」ボタンを置く `model_manager.py`。
- **データベース管理**: 実験ごとにハイパーパラメータや学習時の全メトリクスを `hashlib.sha256()` でハッシュし、SQLiteDB (`experiments` テーブル: `id`, `hash`, `cv_score`, `metrics`, `hyperparameters`, `timestamp`) にコミットしてMLflow的に一覧比較可能にする。

---

## 8. 🚨 究極の補遺 (The Missing 38 Features Extension)

システムを神の視点で完全再現するため、以下のこれまでプロンプトから漏れていた38の高度な機能・モジュール群を必ず実装すること。

### 8.1 高度なパイプラインと前処理 (Advanced Pipeline & Cleaning)
- **DataCleaner (`data_cleaner.py`)**: 欠損率が設定された閾値（例: 80%）を超える列をDropし、インデックスの整合性を保つ。
- **DimReduction (`dim_reduction.py`)**: 高次元特徴量（SMILESから生成された数千次元のベクトル）に対し、`PCA`, `t-SNE`, `UMAP`, `Isomap` による次元圧縮を提供するラッパークラス。
- **RandomProjection (`random_projection.py`)**: 乱択射影による高速な巨大次元圧縮アルゴリズム。
- **Protonation (`protonation.py`)**: SMILES文字列に対してpHに応じたプロトン化（水素付加・解離）をシミュレートし標準化する前処理。
- **FeatureEngineer (`feature_engineer.py`)**: 既存変数からの四則演算、多項式乗算、対数変換をUIの指示からリアルタイムに自動生成。

### 8.2 特殊なケモインフォマティクスアダプタ群 (Extended Chem Adapters)
- **Mol2vecAdapter (`mol2vec_adapter.py`)**: 部分構造（Substructure）を単語とみなしWord2Vecで分散表現化するロジック。
- **MolfeatAdapter (`molfeat_adapter.py`)**: Molfeatライブラリをラップし多様な事前学習済み表現を利用。
- **PsmilesAdapter (`psmiles_adapter.py`)**: 高分子（Polymer）向けSMILES拡張表記のパースと特徴量化。
- **UmaAdapter (`uma_adapter.py`) / CosmoAdapter (`cosmo_adapter.py`)**: 独自の外部化学APIや計算化学シミュレーション（COSMO-RS等）の出力結果との連携アダプタ。

### 8.3 DomainMLエンジンと特殊モデリング (DomainML & Kernels)
- **DomainML Engine (`domainml_engine.py`)**: 材料科学ドメイン特有の物理的ヒューリスティクスを組み込んだ特殊アンサンブルエンジン。
- **DomainML Laplacian (`domainml_laplacian.py`)**: グラフ上のラプラシアン固有マップを利用したマニフォールド学習。
- **DomainML Kernel Opt (`domainml_kernel_opt.py`)**: ガウス過程やSVRのためのカーネル最適化チューナー。
- **CV Bias Evaluator (`cv_bias_evaluator.py`)**: 交差検証時のデータ分布の偏り（Covariate Shift等のバイアス）を定量評価し、Foldごとの分布非類似度を警告するロジック。
- **CV Walkforward (`cv_walkforward.py`)**: 時系列データにおけるウォークフォワード検証（未来からのリークを厳密に防ぐローリングウィンドウCV）。

### 8.4 UIの深層とセッション永続化 (Deep UI & MLOps)
- **Preset Manager (`preset_manager.py`)**: 「創薬向け」「ポリマー向け」「即座のEDA」など、パイプラインのプリセット設定群を保存・ロードする機能。
- **Type Detector (`type_detector.py`)**: ユーザーがCSVをアップロードした直後に、pandas列をスキャンして自動的に `SMILES`, `Categorical`, `Continuous`, `ID`, `Datetime` に分類する型推論エンジン。
- **E2E SMILES to ML (`e2e_smiles_to_ml.py`)**: UIを介さずとも、PythonスクリプトからSMILES配列とターゲット値を入れるだけで全自動で最終レポートまで出せるエンドツーエンドAPIとしての動作。

### 8.5 これらすべてのテストスイートの強制
・`tests/test_domainml_laplacian.py` や `tests/test_cv_bias_evaluator.py` を含め、上記38箇所に付随するすべての実装コードに対しても、`pytest`で行カバレッジ90%を要求すること。

---

## 9. 最終出力形態

以上の「すべて」の仕様を1行たりともサボらず実装し切ること。すべてのクラス構造、探索空間の隅の数字、UIのCSSクラスの記述、テストディレクトリまでを含め、完全に稼働するシステムを構築せよ。最後にテスト完了報告（行カバレッジと分岐カバレッジ）を添えること。


---

## 10. 🧬 全コードベース完全解析仕様 (Total Codebase Exact AST Blueprint)

以下の仕様はプロジェクトの全ファイル群、およびその内部のすべてのクラス・関数のインターフェース定義を静的解析で忠実に書き下したものである。いかなる機能も見落とさず、すべてのシグネチャを完全実装せよ。

## BACKEND Directory

### File: `preset_manager.py`
- **Function `save_preset`(name, state)**
  - *Doc:* パイプライン設定をYAMLプリセットとして保存する。
- **Function `load_preset`(name, state)**
  - *Doc:* YAMLプリセットを読み込んでstateに適用する。
- **Function `list_presets`()**
  - *Doc:* 利用可能なプリセット一覧を返す。
- **Function `delete_preset`(name)**
  - *Doc:* プリセットを削除する。
- **Function `export_state_summary`(state)**
  - *Doc:* stateから設定サマリーを抽出する（表示用）。
- **Function `record_analysis`(state, result)**
  - *Doc:* 解析結果を履歴として保存する。
- **Function `list_history`()**
  - *Doc:* 保存済み解析履歴をから新しい順に返す。
- **Function `export_config_yaml`(state)**
  - *Doc:* stateからパイプライン設定をYAMLテキストとしてエクスポートする。
- **Function `import_config_yaml`(yaml_text, state)**
  - *Doc:* YAMLテキストからパイプライン設定をstateに読み込む。

### File: `chem\base.py`
- **Class `DescriptorResult`**:
  - *Doc:* 特徴量化の結果を保持するデータクラス。
  - `def success_rate(self)`
  - `def n_descriptors(self)`
- **Class `DescriptorMetadata`**:
  - *Doc:* 記述子の詳細属性を保持するデータクラス。
- **Class `BaseChemAdapter`** (Bases: ABC):
  - *Doc:* 化合物特徴量化アダプタの抽象基底クラス。
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`
  - `def _require_available(self)`

### File: `chem\charge_config.py`
- **Class `MoleculeChargeConfig`**:
  - *Doc:* 1つのSMILS列（または分子セット）に対する電荷・スピン設定。
  - `def uhf(self)`
  - `def to_xtb_args(self, charge_override)`
  - `def default(cls)`
  - `def for_radical(cls, charge)`
  - `def at_physiological_ph(cls)`
- **Class `ChargeConfigStore`**:
  - *Doc:* SMILES列名ごとの MoleculeChargeConfig を保持するストア。
  - `def get_config(self, smiles)`
  - `def set_per_molecule(self, smiles, config)`
  - `def resolve_charge(self, smiles)`
  - `def resolve_spin(self, smiles)`

### File: `chem\chemprop_adapter.py`
- **Class `ChempropAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Chemprop (D-MPNN) アダプタ。
  - `def __init__(self, model_path, features_dim)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\cosmo_adapter.py`
- **Class `CosmoAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* openCOSMO-RS を用いた COSMO-RS 記述子アダプター。
  - `def __init__(self, parameterization)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`

### File: `chem\descriptastorus_adapter.py`
- **Class `DescriptaStorusAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* DescriptaStorus アダプタ。
  - `def __init__(self, descriptor_type)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\descriptor_sets.py`
- **Class `DescriptorSet`**:
  - *Doc:* 記述子の組み合わせ（セット）1つ分を表す。
  - `def active_engines(self)`
  - `def summary(self)`
  - `def to_dict(self)`
  - `def from_dict(cls, d)`
- **Class `DescriptorSetManager`**:
  - *Doc:* 記述子セットのCRUD管理。
  - `def __init__(self, sets)`
  - `def add(self, ds)`
  - `def remove(self, name)`
  - `def get(self, name)`
  - `def list_all(self)`
  - `def list_enabled(self)`
  - `def duplicate(self, name, new_name)`
  - `def reorder(self, names)`
  - `def save_to_file(self, filename)`
  - `def load_from_file(cls, filename)`
  - `def to_session(self)`
  - `def from_session(cls, data)`

### File: `chem\group_contrib_adapter.py`
- **Class `_JobackGroup`**:
- **Class `GroupContribAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Joback法による原子団寄与法の記述子アダプタ。
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`

### File: `chem\mix_rules.py`
- **Class `MixRulesManager`**:
  - *Doc:* 記述子名ごとの加重平均ルールを管理するクラス。
  - `def __init__(self, override_file)`
  - `def _load_overrides(self)`
  - `def save_overrides(self)`
  - `def get_rule(self, descriptor_name)`
  - `def set_rule(self, descriptor_name, rule)`
  - `def batch_get_rules(self, descriptor_names)`
- **Function `get_default_rule`(descriptor_name)**
  - *Doc:* 記述子名から推奨の加重平均ルール（"mol" または "wt"）を推定する。

### File: `chem\mol2vec_adapter.py`
- **Class `Mol2VecAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Mol2Vec アダプタ。
  - `def __init__(self, model_path, radius)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _load_model(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\molai_adapter.py`
- **Class `MolAIAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* MolAI SMILES オートエンコーダー記述子アダプター。
  - `def __init__(self, n_components, latent_dim)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _get_encoder(self)`
  - `def compute(self, smiles_list, selected_descriptors)`
  - `def encode_raw(self, smiles_list)`
  - `def decode(self, latent_vectors)`
  - `def _get_decoder(self)`
  - `def pca_inverse_transform(self, pc_vectors)`
  - `def decode_from_pca(self, pc_vectors)`
  - `def train_autoencoder(self, smiles_list, epochs, batch_size, lr, progress_callback)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`

### File: `chem\molai_inverse.py`
- **Class `MolCandidate`**:
  - *Doc:* 逆解析で発見された候補分子
- **Class `DegeneracyMap`**:
  - *Doc:* PCA空間の縮退マップ: 異なるPC値が同一分子にデコードされる領域
- **Class `MolAIInverseAnalyzer`**:
  - *Doc:* MolAI潜在空間での逆解析・分子探索
  - `def __init__(self, adapter)`
  - `def set_training_data(self, smiles_list, pc_vectors)`
  - `def explore_random(self, model, target_value, n_candidates, n_samples, maximize)`
  - `def explore_interpolation(self, smiles_a, smiles_b, n_steps, model)`
  - `def explore_bayesian(self, model, target_value, n_candidates, n_iterations, maximize)`
  - `def compute_degeneracy_map(self, pc_dim1, pc_dim2, n_grid, fixed_dims)`
  - `def _evaluate_candidates(self, pc_vectors, decoded_smiles, model, target_value, maximize)`

### File: `chem\molfeat_adapter.py`
- **Class `MolfeatAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Molfeat アダプタ。
  - `def __init__(self, calculator_type)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\mordred_adapter.py`
- **Class `MordredAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Mordred による包括的な分子記述子計算アダプタ。
  - `def __init__(self, use_3d, selected_only)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _build_calculator(self)`
  - `def _get_all_mordred_names(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`
  - `def get_descriptor_names(self)`

### File: `chem\padel_adapter.py`
- **Class `PaDELAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* PaDEL-Descriptor アダプタ。
  - `def __init__(self, compute_fingerprints, timeout)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\protonation.py`
- **Function `apply_protonation`(smiles, config)**
  - *Doc:* SMILESにプロトン化設定を適用して変換後のSMILESを返す。
- **Function `apply_protonation_batch`(smiles_list, config)**
  - *Doc:* SMILES リストに対してバッチでプロトン化変換を行う。
- **Function `get_protonation_state_info`(smiles, ph)**
  - *Doc:* 指定 pH での分子のプロトン化状態情報を返すヘルパー関数。

### File: `chem\psmiles_adapter.py`
- **Class `PSmilesAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* PSMILES（ポリマーSMILES）から特徴量を計算するアダプタ。
  - `def is_available(cls)`
  - `def has_psmiles_lib(cls)`
  - `def is_psmiles(smiles)`
  - `def _fallback_process_psmiles(smiles)`
  - `def description(self)`
  - `def get_info(cls)`
  - `def compute(self, smiles_list)`

### File: `chem\rdkit_adapter.py`
- **Class `RDKitAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* RDKit による化合物記述子計算アダプタ。
  - `def __init__(self, compute_fp, morgan_radius, morgan_bits, rdkit_fp_bits, include_maccs, compute_gasteiger)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list, charge_config_store)`
  - `def get_descriptors_metadata(self)`
  - `def get_descriptor_names(self)`
  - `def get_descriptor_jp_info(name)`

### File: `chem\recommender.py`
- **Class `DescriptorInfo`**:
  - *Doc:* 推奨される説明変数のメタデータ
- **Class `TargetRecommendations`**:
  - *Doc:* ある目的変数に関する推奨セット
- **Function `get_all_target_recommendations`()**
  - *Doc:* 登録されているすべての目的変数推奨セットのリストを返す
- **Function `get_target_recommendation_by_name`(name)**
  - *Doc:* 指定された名前のターゲット変数の推奨情報を取得する（部分一致対応）
- **Function `get_target_names`()**
  - *Doc:* 登録されているすべての目的変数名(表示用)のリストを返す
- **Function `get_target_categories`()**
  - *Doc:* 登録されている目的変数のカテゴリ（系統）のユニークなリストを返す
- **Function `get_targets_by_category`(category)**
  - *Doc:* 指定したカテゴリに属する目的変数のリストを返す
- **Function `get_all_descriptor_categories`()**
  - *Doc:* 全ての説明変数カテゴリ（物理的意味の分類）のユニークなリストを返す
- **Function `get_descriptors_by_category`(category)**
  - *Doc:* 指定した意味カテゴリに属する説明変数のリストを重複なく返す

### File: `chem\skfp_adapter.py`
- **Class `SkfpAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* scikit-fingerprints アダプタ。
  - `def __init__(self, fp_types, fp_configs)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`

### File: `chem\smiles_transformer.py`
- **Class `SmilesDescriptorTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* SMILES列を記述子に変換するsklearn互換Transformer。
  - `def __init__(self, smiles_col, selected_descriptors, active_engines, count_normalization, fraction_type)`
  - `def _compute_descriptors(self, smiles_list)`
  - `def _apply_count_normalization(self, X_chem, smiles_list)`
  - `def _identify_count_columns(columns)`
  - `def _compute_molecular_weight(smiles_list)`
  - `def _compute_molar_volumes(smiles_list)`
  - `def _transform_mixture(self, X)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
- **Function `progressive_precalculate`(smiles_list, target_col_name)**
  - *Doc:* ユーザーの要求に応じ、優先順位をつけて事前計算を行い、進捗を yield するジェネレータ。
- **Function `precalculate_all_descriptors`(smiles_list, target_col_name, engine_flags, molai_n_components, progress_callback)**
  - *Doc:* 全記述子を一括計算する関数。app.py のインラインコードを統一。

### File: `chem\uma_adapter.py`
- **Class `UMAAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Meta UMA (Universal Model for Atoms) による分子記述子生成。
  - `def __init__(self, model_name, device)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _ensure_model_loaded(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`
  - `def get_descriptor_names(self)`

### File: `chem\unipka_adapter.py`
- **Class `UniPkaAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* Uni-pKa (dptech-corp) の Python ラッパー unipka を使った pKa 記述子アダプター。
  - `def __init__(self, batch_size, remove_hs)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _get_model(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`

### File: `chem\xtb_adapter.py`
- **Class `XTBAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* XTB (GFN2-xTB) による量子化学計算記述子アダプター。
  - `def __init__(self, gfn, calc_type, convergence, solvent, timeout, max_retries)`
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def _build_cmd(self, xyz_path, charge, uhf, calc_type, convergence, solvent)`
  - `def compute(self, smiles_list, selected_descriptors, charge_config_store)`
  - `def get_descriptor_names(self)`
  - `def get_descriptors_metadata(self)`

### File: `chem\__init__.py`
- **Function `get_available_adapters`()**
  - *Doc:* インストール済みで利用可能なアダプターのみを返す。

### File: `chem\descriptors\base.py`
- **Class `ParamDef`**:
  - *Doc:* プラグインの設定可能パラメータ定義。
- **Class `PluginInfo`**:
  - *Doc:* ロード済みプラグインの情報を保持するデータクラス。
  - `def display_name(self)`
  - `def has_params(self)`
- **Function `validate_plugin`(module, filepath)**
  - *Doc:* プラグインモジュールを検証し、PluginInfo を返す。
- **Function `safe_compute`(plugin, smiles_list)**
  - *Doc:* プラグインの compute を安全に呼び出す。

### File: `chem\descriptors\__init__.py`
- **Function `discover_plugins`(force_reload)**
  - *Doc:* _builtins/ と custom/ の全プラグインを検出し、PluginInfo のリストを返す。
- **Function `get_plugins_by_engine`(engine)**
  - *Doc:* 指定エンジンのプラグインを返す。
- **Function `get_plugins_by_category`(category)**
  - *Doc:* 指定カテゴリのプラグインを返す。
- **Function `get_available_engines`()**
  - *Doc:* 利用可能なエンジン名のリストを返す。
- **Function `get_available_categories`()**
  - *Doc:* 利用可能なカテゴリ名のリストを返す。
- **Function `compute_all_descriptors`(smiles_list, plugin_names, progress_callback)**
  - *Doc:* 全プラグイン（またはplugin_namesで指定）の記述子を計算し、
- **Function `get_custom_dir`()**
  - *Doc:* カスタムディレクトリのパスを返す。存在しなければ作成。
- **Function `get_builtins_dir`()**
  - *Doc:* 組込みディレクトリのパスを返す。
- **Function `invalidate_cache`()**
  - *Doc:* プラグインキャッシュをクリアする。次回 discover_plugins で再スキャン。

### File: `chem\descriptors\custom\ext_test_lipophilicity.py`
- **Function `compute`(smiles_list)**
  - *Doc:* SMILESリストから親油性・水溶性関連記述子を計算する。

### File: `chem\descriptors\custom\_template_simple.py`
- **Function `compute`(smiles_list)**
  - *Doc:* SMILESのリストを受け取り、各SMILESに対する記述子値のリストを返す。

### File: `chem\descriptors\custom\_template_with_config.py`
- **Function `compute`(smiles_list)**
  - *Doc:* SMILESリストと設定値を受け取り、記述子のDataFrameを返す。

### File: `chem\descriptors\_builtins\cosmo_sigma.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\group_contrib.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\mordred_selected.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_electronic.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_fingerprints.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_fragments.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_graph_indices.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_physicochemical.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_surface_area.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\rdkit_topology.py`
- **Function `compute`(smiles_list)**

### File: `chem\descriptors\_builtins\xtb_electronic.py`
- **Function `compute`(smiles_list)**

### File: `data\benchmark.py`
- **Class `ModelScore`**:
  - *Doc:* 1モデルの評価結果。
  - `def to_dict(self)`
- **Class `BenchmarkResult`**:
  - *Doc:* 複数モデルのベンチマーク結果。
  - `def to_dataframe(self)`
  - `def best(self)`
- **Function `evaluate_regression`(y_true, y_pred, model_key, train_time, cv_mean, cv_std)**
  - *Doc:* 回帰タスクの評価指標を計算して ModelScore を返す。
- **Function `evaluate_classification`(y_true, y_pred, y_prob, model_key, train_time, cv_mean, cv_std)**
  - *Doc:* 分類タスクの評価指標を計算して ModelScore を返す。
- **Function `compute_learning_curve`(estimator, X, y, scoring, cv, n_points, n_jobs)**
  - *Doc:* 学習曲線データを計算する。
- **Function `benchmark_models`(models, X_train, y_train, X_test, y_test, task, scoring)**
  - *Doc:* 複数のモデルをベンチマーク比較する。

### File: `data\benchmark_datasets.py`
- **Function `list_benchmark_datasets`()**
  - *Doc:* 利用可能なベンチマークデータセットの一覧を返す。
- **Function `load_benchmark`(name)**
  - *Doc:* 指定されたベンチマークデータセットをダウンロードしてDataFrameとして返す。

### File: `data\data_cleaner.py`
- **Class `CleaningAction`**:
  - *Doc:* 1回のクリーニング操作のログレコード。
  - `def rows_removed(self)`
  - `def cols_removed(self)`
- **Function `drop_columns`(df, columns)**
  - *Doc:* 指定列をDataFrameから除外する。
- **Function `drop_rows_with_missing`(df, threshold, subset)**
  - *Doc:* 欠損率が閾値を超える行を削除する。
- **Function `remove_constant_columns`(df)**
  - *Doc:* ユニーク値が1以下の定数列を除去する。
- **Function `clip_outliers`(df, iqr_multiplier, columns)**
  - *Doc:* IQR法で外れ値をクリッピングする。
- **Function `remove_duplicates`(df, subset, keep)**
  - *Doc:* 重複行を除去する。
- **Function `preview_missing_impact`(df, threshold, subset)**
  - *Doc:* 欠損行削除の影響行数をプレビューする（実際には削除しない）。
- **Function `preview_outlier_impact`(df, iqr_multiplier, columns)**
  - *Doc:* 外れ値クリッピングの影響値数をプレビューする。
- **Function `get_cleaning_summary`(df)**
  - *Doc:* 現在のDataFrameのクリーニング候補をサマリーで返す。

### File: `data\dim_reduction.py`
- **Class `DimReductionConfig`**:
  - *Doc:* 次元削減の設定。
- **Class `DimReducer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 次元削減Transformer (PCA / t-SNE / UMAP)。
  - `def __init__(self, config)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def fit_transform(self, X, y)`
  - `def _prepare(self, X, fit)`
  - `def get_feature_names_out(self, input_features)`
  - `def explained_variance_ratio_(self)`
  - `def loadings_(self)`
  - `def feature_names_in_(self)`
  - `def reconstruction_error_(self)`
- **Function `run_pca`(df, n_components, scale, target_col)**
  - *Doc:* DataFrameにPCAを適用して2D埋め込み + 寄与率を返す。
- **Function `run_tsne`(df, n_components, perplexity, scale, target_col, random_state)**
  - *Doc:* DataFrameにt-SNEを適用して2D埋め込みを返す。
- **Function `run_umap`(df, n_components, n_neighbors, min_dist, scale, target_col, random_state)**
  - *Doc:* DataFrameにUMAPを適用して埋め込みを返す。

### File: `data\eda.py`
- **Class `ColumnStats`**:
  - *Doc:* 1列の統計情報。
- **Class `OutlierResult`**:
  - *Doc:* 外れ値検出結果。
- **Function `compute_column_stats`(df)**
  - *Doc:* DataFrame の全列の統計情報を計算して返す。
- **Function `summarize_dataframe`(df)**
  - *Doc:* DataFrame 全体のサマリーを返す。
- **Function `compute_correlation`(df, method, target_col)**
  - *Doc:* 数値列の相関行列を計算して返す。
- **Function `detect_outliers`(df, method, k, z_threshold, cols)**
  - *Doc:* 外れ値を検出して結果を返す。
- **Function `compute_distribution`(series, bins)**
  - *Doc:* 1列のヒストグラムデータとカーネル密度推定用データを返す。
- **Function `analyze_target`(df, target_col, task)**
  - *Doc:* 目的変数の統計・分布情報を返す。

### File: `data\feature_engineer.py`
- **Class `InteractionTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 指定列の全ペアの積 (交互作用項) を追加するTransformer。
  - `def __init__(self, degree, interaction_only, include_bias)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
- **Class `GroupAggTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* カテゴリ列でグループ化した集約特徴量（mean/std/min/max/count）を追加する。
  - `def __init__(self, group_col, agg_cols, agg_funcs)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
- **Class `DatetimeFeatureExtractor`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* Datetime列から時系列特徴量（年/月/日/時/曜日/週番号等）を抽出するTransformer。
  - `def __init__(self, components, add_cyclic)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def _extract(self, X)`
  - `def get_feature_names_out(self, input_features)`
- **Class `LagRollingTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 時系列データのラグ特徴量・ローリング統計量を生成するTransformer。
  - `def __init__(self, lags, windows, agg_funcs)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
- **Class `FeatureEngineeringConfig`**:
  - *Doc:* 特徴量エンジニアリングの設定。
- **Function `build_feature_engineering_pipeline`(config)**
  - *Doc:* 設定に基づいて特徴量エンジニアリングステップのリストを返す。

### File: `data\leakage_detector.py`
- **Class `LeakagePair`**:
  - *Doc:* リーケージが疑われるサンプルペア。
- **Class `LeakageReport`**:
  - *Doc:* リーケージ検出の結果レポート。
- **Class `FeatureLeakageWarning`**:
  - *Doc:* 特徴量レベルのリーケージ警告。
- **Class `FeatureLeakageReport`**:
  - *Doc:* 特徴量レベルのリーケージ検出レポート。
- **Function `compute_hat_matrix`(X)**
  - *Doc:* ハット行列（smoother行列）を計算する。
- **Function `compute_rbf_gram`(X, gamma)**
  - *Doc:* RBF（ガウス）カーネルのグラム行列を計算する。
- **Function `compute_rf_proximity`(X, y, n_estimators, random_state)**
  - *Doc:* ランダムフォレストの近接度行列（Proximity Matrix）を計算する。
- **Function `estimate_groups`(S, n_clusters_range, method)**
  - *Doc:* 類似度行列からグループ（クラスタ）を推定する。
- **Function `detect_leakage`(X, y, method, similarity_threshold, top_k, rf_n_estimators)**
  - *Doc:* リーケージ検出のメインAPI。
- **Function `check_feature_leakage`(df, target_col, exclude_cols, corr_threshold_high, corr_threshold_medium, max_features_to_check)**
  - *Doc:* 特徴量と目的変数間のリーケージリスクを自動チェックする（軽量版）。

### File: `data\loader.py`
- **Function `load_file`(path, smiles_col, target_col, sqlite_query)**
  - *Doc:* ファイルパスから DataFrame を読み込む。
- **Function `load_from_bytes`(content, filename)**
  - *Doc:* アップロードされたファイルのバイト列からDataFrameを読み込む（Streamlit用）。
- **Function `save_dataframe`(df, path, fmt)**
  - *Doc:* DataFrame をファイルに保存する。
- **Function `get_supported_extensions`()**
  - *Doc:* 対応している拡張子の一覧を返す（GUI表示用）。

### File: `data\preprocessor.py`
- **Class `LogTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 対数変換を行うカスタムTransformer。
  - `def __init__(self, offset)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def inverse_transform(self, X)`
- **Class `SinCosTransformer`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 周期変数（角度, 時刻等）に対して sin/cos 変換を行うTransformer。
  - `def __init__(self, period)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
- **Class `PreprocessConfig`**:
  - *Doc:* 前処理パイプラインの設定。AutoMLモードでは自動設定、
- **Class `Preprocessor`**:
  - *Doc:* TypeDetector の結果を受け取り、sklearn ColumnTransformer パイプラインを
  - `def __init__(self, config)`
  - `def build(self, detection_result, target_col)`
  - `def _build_numeric_pipeline(self, scaler_name, context)`
  - `def _build_numeric_imputer(self)`
  - `def _build_categorical_pipeline(self, encoder_name, cardinality)`
  - `def transformer(self)`
- **Function `build_full_pipeline`(detection_result, model, target_col, config)**
  - *Doc:* 前処理 + (JL-RP) + モデルの sklearn Pipeline を構築して返す。

### File: `data\random_projection.py`
- **Class `JLRandomProjection`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* Johnson-Lindenstrauss 補題に基づく自動ランダム射影。
  - `def __init__(self, eps, method, density, random_state)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
  - `def _resolve_method(self, n_features)`
  - `def _to_array(X)`
  - `def summary(self)`
- **Function `should_apply_random_projection`(n_features, n_samples, eps)**
  - *Doc:* JL条件を事前チェックするユーティリティ関数。

### File: `data\type_detector.py`
- **Class `ColumnType`** (Bases: Enum):
  - *Doc:* 変数の種別を表す列挙型。
- **Class `ColumnInfo`**:
  - *Doc:* 1列の判定結果を保持するデータクラス。
  - `def is_numeric(self)`
  - `def is_categorical(self)`
- **Class `DetectionResult`**:
  - *Doc:* データセット全体の型判定結果。
  - `def numeric_columns(self)`
  - `def categorical_columns(self)`
  - `def binary_columns(self)`
  - `def ignored_columns(self)`
  - `def get_columns_by_type(self, col_type)`
  - `def get_numeric_columns(self)`
  - `def get_categorical_columns(self)`
  - `def summary_table(self)`
- **Class `TypeDetector`**:
  - *Doc:* DataFrameの各列の変数型を自動判定するクラス。
  - `def __init__(self, cardinality_threshold, skewness_threshold, outlier_iqr_factor, smiles_col_hints, periodic_cols)`
  - `def detect(self, df)`
  - `def _detect_column(self, series, name)`
  - `def _classify_numeric(self, series, name, n_unique, null_rate)`
  - `def _classify_categorical(self, series, name, n_unique, null_rate)`
  - `def _looks_like_smiles(self, series, col_name)`
  - `def _looks_like_datetime(series)`

### File: `doe\candidate.py`
- **Function `generate_candidate_set`(factors, max_candidates, random_seed)**
  - *Doc:* 候補点集合を生成する。
- **Function `build_model_matrix`(factors, df)**
  - *Doc:* 切片 + 主効果モデル行列を構築する。
- **Function `col_names`(factors)**
  - *Doc:* モデル行列列名（切片 + 各因子）。

### File: `doe\design.py`
- **Class `DoEResult`**:
  - *Doc:* 実験計画の最適化結果。
- **Class `DoEOptimizer`**:
  - *Doc:* 座標交換法による最適実験計画生成。
  - `def __init__(self, factors, n_new, criterion, max_candidates, random_seed, n_starts, max_iter, existing_df)`
  - `def optimize(self)`
  - `def _assemble(self, X_new, exclude_row)`
  - `def _assemble_full(self, X_new)`
  - `def _score(self, X)`
  - `def _d_efficiency(self, X)`
  - `def _criterion_human(self, X)`
  - `def _build_result(self, best_idx, X_full)`

### File: `doe\factor.py`
- **Class `FactorType`** (Bases: str, Enum):
- **Class `Factor`**:
  - *Doc:* 1つの実験因子を表す。
  - `def continuous(cls, name, low, high, n_levels)`
  - `def categorical(cls, name, categories)`
  - `def levels(self)`
  - `def n_cols(self)`
  - `def col_names(self)`
  - `def encode(self, values)`
  - `def decode(self, encoded_row)`

### File: `doe\orthogonal.py`
- **Function `list_oa_names`()**
- **Function `get_oa_info`(name)**
- **Function `apply_orthogonal_array`(oa_name, factors)**
  - *Doc:* 直交表を因子の水準にマッピングして設計点を返す。

### File: `export\base.py`
- **Class `BaseExporter`**:
  - *Doc:* すべてのエクスポータが実装しなければならない共通インターフェース。
  - `def __init__(self, output_dir)`
  - `def export(self, result, filename)`

### File: `export\chart_bundle.py`
- **Class `ChartBundleExporter`** (Bases: BaseExporter):
  - *Doc:* チャート画像群を ZIP に束ねるエクスポータ。
  - `def export(self, result, filename)`

### File: `export\notebook_exporter.py`
- **Class `NotebookExporter`** (Bases: BaseExporter):
  - *Doc:* 解析結果を Jupyter Notebook に変換するエクスポータ。
  - `def _imports_cell(self)`
  - `def _data_load_cell(self, result)`
  - `def _preprocessing_cell(self, result)`
  - `def _model_cell(self, result)`
  - `def _shap_cell(self)`
  - `def _summary_markdown(self, result)`
  - `def export(self, result, filename)`

### File: `export\pdf_exporter.py`
- **Class `PDFExporter`** (Bases: BaseExporter):
  - *Doc:* 解析結果を PDF に変換して保存するエクスポータ。
  - `def __init__(self, output_dir)`
  - `def _register_japanese_font(self)`
  - `def _styles(self)`
  - `def _metrics_table(self, metrics, styles)`
  - `def _importance_table(self, importances, styles, top_n)`
  - `def export(self, result, filename)`

### File: `export\word_exporter.py`
- **Class `WordExporter`** (Bases: BaseExporter):
  - *Doc:* 解析結果を Word (.docx) レポートに変換するエクスポータ。
  - `def _set_cell_bg(self, cell, hex_color)`
  - `def _add_heading(self, doc, text, level)`
  - `def _add_metrics_table(self, doc, metrics)`
  - `def _add_importance_table(self, doc, importances, top_n)`
  - `def export(self, result, filename)`

### File: `hsp\hsp_calculator.py`
- **Class `HSPCalculator`**:
  - *Doc:* Hansen Solubility Parameters の計算・評価ツール。
  - `def calculate_red_value(solute_hsp, solvent_hsp, radius)`
  - `def hansen_distance(hsp_a, hsp_b)`
  - `def predict_from_smiles(smiles)`
  - `def predict_batch(smiles_list)`
  - `def fit_sphere(hsp_data, labels)`

### File: `hsp\hsp_predictor.py`
- **Class `GroupContribution`**:
  - *Doc:* 1つの官能基のHSP寄与パラメータ。
- **Class `HSPPredictor`**:
  - *Doc:* SMILES から HSP (δD, δP, δH) を予測する。
  - `def __init__(self, model_path)`
  - `def _load_model(self, path)`
  - `def is_available(self)`
  - `def predict(self, smiles)`
  - `def predict_batch(self, smiles_list)`
  - `def save_model(self, path)`
  - `def _extract_features(smiles)`

### File: `interpret\shap_explainer.py`
- **Class `ShapConfig`**:
  - *Doc:* SHAP計算の設定を保持するデータクラス。
- **Class `ShapResult`**:
  - *Doc:* SHAP計算結果を保持するデータクラス。
  - `def feature_importance(self)`
  - `def top_features(self, n)`
- **Class `ShapExplainer`**:
  - *Doc:* sklearn/XGBoost/LightGBM/CatBoost等のモデルに対してSHAPを計算するクラス。
  - `def __init__(self, config_or_max_display, kernel_nsamples)`
  - `def _select_explainer_type(self, model)`
  - `def explain(self, model, X, feature_names, background_data, compute_interactions)`
  - `def _build_explainer(self, shap, model, X, background_data)`
  - `def plot_summary(self, result, plot_type, save_path)`
  - `def plot_waterfall(self, result, sample_idx, save_path)`
  - `def plot_dependence(self, result, feature, interaction_feature, save_path)`
  - `def get_feature_importance_df(self, result)`

### File: `interpret\sri.py`
- **Class `SRIResult`**:
  - *Doc:* SRI分解結果を保持するデータクラス。
  - `def summary_df(self)`
  - `def pairwise_df(self)`
- **Class `SRIDecomposer`**:
  - *Doc:* SHAP ベクトルの SRI 分解器。
  - `def __init__(self, center)`
  - `def decompose(self, shap_result)`
- **Function `plot_sri_heatmap`(sri_result, component, top_n, ax, save_path)**
  - *Doc:* SRI の Synergy/Redundancy ヒートマップを表示する。
- **Function `select_features_by_independence`(sri_result, top_n, threshold)**
  - *Doc:* Independence スコアを基準に特徴量を選択する。

### File: `llm\generator.py`
- **Class `LLMGeneratorError`** (Bases: Exception):
  - *Doc:* コード生成・検証エラー。
- **Class `GenerationResult`**:
  - *Doc:* 生成・保存結果。
- **Class `LLMDescriptorGenerator`**:
  - *Doc:* LLMを使った記述子コード生成器。
  - `def __init__(self, provider)`
  - `def generate(self, user_description)`
  - `def generate_and_save(self, user_description, filename)`
  - `def _build_context(self)`

### File: `llm\prompt_builder.py`
- **Class `DescriptorIntent`**:
  - *Doc:* ユーザーの記述子作成意図を表すデータクラス。
  - `def is_valid(self)`
- **Function `build_external_llm_prompt`(intent)**
  - *Doc:* 外部LLM（ChatGPT等）に渡すプロンプトを生成する。

### File: `llm\provider.py`
- **Class `LLMProviderError`** (Bases: Exception):
  - *Doc:* LLMプロバイダー共通例外。
- **Class `LLMRequest`**:
  - *Doc:* LLMへのリクエスト定義。
- **Class `LLMResponse`**:
  - *Doc:* LLMからのレスポンス定義。
- **Class `LLMProvider`** (Bases: ABC):
  - *Doc:* LLMプロバイダーの抽象基底クラス。
  - `def name(self)`
  - `def is_available(self)`
  - `def generate(self, request)`
  - `def generate_descriptor_code(self, user_description)`
- **Class `StubLLMProvider`** (Bases: LLMProvider):
  - *Doc:* スタブ（ダミー）実装。
  - `def name(self)`
  - `def is_available(self)`
  - `def generate(self, request)`

### File: `llm\registry.py`
- **Class `LLMProviderRegistry`**:
  - *Doc:* LLMプロバイダーを名前で管理するレジストリ。
  - `def __init__(self)`
  - `def register(self, name, cls)`
  - `def get(self, name)`
  - `def list_available(self)`
  - `def list_all_with_status(self)`

### File: `llm\reviewer.py`
- **Class `ReviewIssue`**:
- **Class `CodeReviewResult`**:
  - `def has_errors(self)`
  - `def error_count(self)`
  - `def warn_count(self)`
- **Class `LLMCodeReviewer`**:
  - *Doc:* 生成されたコードをLLMでレビューする。
  - `def __init__(self, provider)`
  - `def review(self, code, user_intent, max_tokens)`

### File: `llm\__init__.py`
- **Function `get_llm_provider`(name)**
  - *Doc:* 登録済みLLMプロバイダーを取得する。
- **Function `register_llm_provider`(name, cls)**
  - *Doc:* 新しいLLMプロバイダーを登録する（プラグイン拡張用）。

### File: `llm\providers\hf_provider.py`
- **Class `DownloadProgress`**:
  - *Doc:* ダウンロード進行状況。
- **Class `HuggingFaceProvider`** (Bases: LLMProvider):
  - *Doc:* HuggingFace Hub からダウンロードしたモデルでローカル推論するプロバイダー。
  - `def __init__(self, model_id)`
  - `def name(self)`
  - `def display_name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def generate(self, request)`
- **Function `load_hf_config`()**
  - *Doc:* HF設定を読み込む。
- **Function `save_hf_config`(config)**
  - *Doc:* HF設定を保存する。
- **Function `get_hf_token`()**
  - *Doc:* HuggingFaceトークンを取得（設定ファイル > 環境変数の順）。
- **Function `get_model_info`(model_id)**
  - *Doc:* モデルIDに対応するカタログ情報を返す。
- **Function `get_download_progress`(model_id)**
- **Function `is_model_downloaded`(model_id)**
  - *Doc:* モデルがキャッシュ済みかどうかを確認する。
- **Function `download_model_async`(model_id, token, on_progress)**
  - *Doc:* モデルを非同期でダウンロードする。
- **Function `load_model`(model_id, token)**
  - *Doc:* モデルとトークナイザーをロードする（キャッシュ済み）。

### File: `llm\providers\openai_provider.py`
- **Class `OpenAIProvider`** (Bases: LLMProvider):
  - *Doc:* OpenAI Chat Completions APIを使ったプロバイダー。
  - `def __init__(self, model, api_key)`
  - `def name(self)`
  - `def display_name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def generate(self, request)`

### File: `mlops\mlflow_manager.py`
- **Class `MLflowManager`**:
  - *Doc:* MLflow との連携を管理するクラス。
  - `def __init__(self, tracking_uri, experiment_name)`
  - `def start_run(self, run_name, tags)`
  - `def end_run(self, status)`
  - `def fail_run(self, exc)`
  - `def log_params(self, params)`
  - `def log_metrics(self, metrics, step)`
  - `def log_artifact(self, local_path, artifact_path)`
  - `def log_figure(self, fig, filename)`
  - `def save_model(self, model, model_name, save_dir, register)`
  - `def load_model(self, model_path)`
  - `def get_experiment_runs(self, max_results, order_by)`
  - `def get_best_run(self, metric_name, ascending)`
- **Class `MLRunContext`**:
  - *Doc:* with 文で MLflow ランを安全に管理するコンテキストマネージャ。
  - `def __init__(self, manager, run_name, tags)`

### File: `models\automl.py`
- **Class `AutoMLResult`**:
  - *Doc:* AutoML実行結果を保持するデータクラス。
- **Class `AutoMLEngine`**:
  - *Doc:* AutoMLエンジン。
  - `def __init__(self, task, cv_folds, cv_key, cv_groups_col, model_keys, model_params, preprocess_params, timeout_seconds, progress_callback, selected_descriptors, active_engines, monotonic_constraints_dict, column_meta_dict, count_normalization)`
  - `def run(self, df, target_col, smiles_col, fraction_type, group_col, preprocess_config, cv_extra_params)`
  - `def run_multi_feature_sets(self, df, target_col, feature_sets, smiles_col, group_col, cv_extra_params, progress_callback_outer)`
  - `def _infer_task(y_series)`
  - `def _get_scoring(task)`
  - `def _check_data_quality(df, target_col, warnings)`

### File: `models\cv_bias_evaluator.py`
- **Class `CVBiasResult`**:
  - *Doc:* CVバイアス評価の結果。
  - `def to_dict(self)`
- **Function `estimate_tibshirani_bias`(fold_error_curves, param_values, higher_is_better)**
  - *Doc:* Tibshirani-Tibshirani法によるCVバイアス推定。
- **Function `estimate_bbc_cv_bias`(oos_predictions, y_true, scoring_func, n_bootstrap, random_state, higher_is_better)**
  - *Doc:* BBC-CVによるCVバイアス推定。
- **Function `format_bias_report`(result)**
  - *Doc:* CVBiasResultを人間可読な文字列に整形する。

### File: `models\cv_manager.py`
- **Class `WalkForwardSplit`**:
  - *Doc:* 時系列データのウォークフォワード検証（拡張窓方式）。
  - `def __init__(self, n_splits, min_train_size, gap)`
  - `def split(self, X, y, groups)`
  - `def get_n_splits(self, X, y, groups)`
- **Class `CVConfig`**:
  - *Doc:* クロスバリデーションの設定。
- **Function `get_cv`(config)**
  - *Doc:* CVConfig に基づいて CV スプリッタを返す。
- **Function `list_cv_methods`(task, requires_groups)**
  - *Doc:* 利用可能なCV手法の一覧を返す。
- **Function `run_cross_validation`(model, X, y, cv_config, scoring, groups, n_jobs, return_train_score, fit_params)**
  - *Doc:* クロスバリデーションを実行して結果を返す。

### File: `models\factory.py`
- **Function `get_model`(model_key, task)**
  - *Doc:* 指定された model_key に対応する学習済みモデルインスタンスを返す。
- **Function `list_models`(task, available_only, tags)**
  - *Doc:* 利用可能なモデルの一覧を返す。
- **Function `get_default_automl_models`(task)**
  - *Doc:* AutoMLモードで使用するデフォルトモデルキーのリストを返す。
- **Function `get_model_registry`(task)**
  - *Doc:* モデルのレジストリ（メタデータ全体）を返す。

### File: `models\linear_tree.py`
- **Class `_Node`**:
  - *Doc:* 決定木の1ノード。
  - `def is_leaf(self)`
- **Class `_LinearTreeCore`**:
  - *Doc:* LinearTree の木構築ロジックを共有するMixin。
  - `def _build_tree(self, X, y, depth)`
  - `def _get_feature_indices(self, n_features, node_depth, node_id)`
  - `def _get_thresholds(self, unique_vals)`
  - `def _split_score(self, X, y, left_mask, right_mask)`
  - `def _route_to_leaf(self, node, x)`
  - `def _count_leaves(self, node)`
- **Class `LinearTreeRegressor`** (Bases: _LinearTreeCore, BaseEstimator, RegressorMixin):
  - *Doc:* Linear Tree Regressor。
  - `def __init__(self, base_estimator, max_depth, min_samples_split, min_samples_leaf, max_features, max_bins, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
  - `def _split_score(self, X, y, left_mask, right_mask)`
- **Class `LinearTreeClassifier`** (Bases: _LinearTreeCore, BaseEstimator, ClassifierMixin):
  - *Doc:* Linear Tree Classifier。
  - `def __init__(self, base_estimator, max_depth, min_samples_split, min_samples_leaf, max_features, max_bins, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
  - `def predict_proba(self, X)`
  - `def _split_score(self, X, y, left_mask, right_mask)`
- **Class `LinearForestRegressor`** (Bases: BaseEstimator, RegressorMixin):
  - *Doc:* Linear Forest Regressor。
  - `def __init__(self, base_estimator, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_bins, bootstrap, max_samples, n_jobs, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
- **Class `LinearForestClassifier`** (Bases: BaseEstimator, ClassifierMixin):
  - *Doc:* Linear Forest Classifier。
  - `def __init__(self, base_estimator, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, max_bins, bootstrap, max_samples, n_jobs, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
  - `def predict_proba(self, X)`
- **Class `LinearBoostRegressor`** (Bases: BaseEstimator, RegressorMixin):
  - *Doc:* Linear Boost Regressor。
  - `def __init__(self, base_estimator, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, max_bins, subsample, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
- **Class `LinearBoostClassifier`** (Bases: BaseEstimator, ClassifierMixin):
  - *Doc:* Linear Boost Classifier。
  - `def __init__(self, base_estimator, n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, max_bins, subsample, random_state)`
  - `def _sigmoid(x)`
  - `def _softmax(x)`
  - `def fit(self, X, y)`
  - `def predict_proba(self, X)`
  - `def predict(self, X)`
- **Function `RidgeTreeRegressor`()**
  - *Doc:* Ridge を葉に使う LinearTreeRegressor のショートカット。
- **Function `RidgeTreeClassifier`()**
  - *Doc:* LogisticRegression(C=1) を葉に使う LinearTreeClassifier のショートカット。

### File: `models\monotonic_kernel.py`
- **Class `MonotonicKernelWrapper`** (Bases: BaseEstimator, RegressorMixin):
  - *Doc:* カーネル系回帰モデル（SVR / KernelRidge / GPR等）に
  - `def __init__(self, base_estimator, monotonic_constraints, n_grid, sigma_factor, penalty_weight, max_iter, violation_threshold)`
  - `def fit(self, X, y, sample_weight)`
  - `def predict(self, X)`
  - `def score(self, X, y)`
  - `def get_params(self, deep)`
  - `def set_params(self)`
- **Class `MonotonicKernelClassifierWrapper`** (Bases: BaseEstimator, ClassifierMixin):
  - *Doc:* カーネル系分類モデル（SVC 等）に
  - `def __init__(self, base_estimator, monotonic_constraints, n_grid, sigma_factor, penalty_weight, max_iter, violation_threshold)`
  - `def fit(self, X, y, sample_weight)`
  - `def predict(self, X)`
  - `def predict_proba(self, X)`
  - `def score(self, X, y)`
  - `def get_params(self, deep)`
  - `def set_params(self)`
- **Function `is_soft_monotonic_candidate`(estimator)**
  - *Doc:* estimator がソフト単調性制約ラッパーの適用対象かを判定する。
- **Function `wrap_with_soft_monotonic`(estimator, monotonic_constraints)**
  - *Doc:* estimator にソフト単調性制約ラッパーを適用する。

### File: `models\monotonic_wrapper.py`
- **Class `MonotonicConstraintRegressor`** (Bases: BaseEstimator, RegressorMixin):
  - *Doc:* 任意の sklearn 回帰モデルに変数ごとの単調性制約を付与する汎用ラッパー。
  - `def __init__(self, base_estimator, monotonic_constraints, constraint_strength, n_grid, sigma_factor, penalty_weight, max_iter, violation_threshold)`
  - `def _effective_params(self)`
  - `def get_params(self, deep)`
  - `def set_params(self)`
  - `def fit(self, X, y, sample_weight)`
  - `def predict(self, X)`
  - `def score(self, X, y)`
- **Class `MonotonicConstraintClassifier`** (Bases: BaseEstimator, ClassifierMixin):
  - *Doc:* 任意の sklearn 分類モデルに変数ごとの単調性制約を付与する汎用ラッパー。
  - `def __init__(self, base_estimator, monotonic_constraints, constraint_strength, n_grid, sigma_factor, penalty_weight, max_iter, violation_threshold)`
  - `def _effective_params(self)`
  - `def get_params(self, deep)`
  - `def set_params(self)`
  - `def fit(self, X, y, sample_weight)`
  - `def predict(self, X)`
  - `def predict_proba(self, X)`
  - `def score(self, X, y)`
- **Function `model_monotonic_strategy`(estimator)**
  - *Doc:* モデルの単調性制約戦略を判定する。
- **Function `wrap_monotonic`(estimator, monotonic_constraints)**
  - *Doc:* estimator に単調性制約ラッパーを適用する。

### File: `models\rgf.py`
- **Class `_RGFCore`**:
  - *Doc:* RGF の中核ロジック。
  - `def _init_forest_state(self)`
  - `def _register_tree_leaves(self, tree, X_train)`
  - `def _get_leaf_indicators(self, X)`
  - `def _update_weights(self, Phi, residuals, lambda_l2)`
  - `def _predict_from_weights(self, X)`
- **Class `RGFRegressor`** (Bases: _RGFCore, BaseEstimator, RegressorMixin):
  - *Doc:* Regularized Greedy Forest Regressor。
  - `def __init__(self, n_estimators, max_leaf_nodes, learning_rate, lambda_l2, lambda_l1, min_samples_leaf, max_features, subsample, loss, random_state)`
  - `def fit(self, X, y)`
  - `def predict(self, X)`
- **Class `RGFClassifier`** (Bases: _RGFCore, BaseEstimator, ClassifierMixin):
  - *Doc:* Regularized Greedy Forest Classifier。
  - `def __init__(self, n_estimators, max_leaf_nodes, learning_rate, lambda_l2, lambda_l1, min_samples_leaf, max_features, subsample, random_state)`
  - `def fit(self, X, y)`
  - `def _fit_binary(self, X_arr, y_int)`
  - `def predict_proba(self, X)`
  - `def predict(self, X)`

### File: `models\search_space_generator.py`
- **Class `SearchParamSpec`**:
  - *Doc:* 1つのパラメータの探索空間定義。
  - `def to_grid_entry(self)`
  - `def to_optuna_entry(self)`
  - `def to_dict(self)`
- **Function `generate_search_space`(param_spec)**
  - *Doc:* ParamSpecから1つのSearchParamSpecを生成する。
- **Function `generate_grid_space`(param_specs)**
  - *Doc:* ParamSpecリストからGridSearchCV用のparam_gridを生成する。
- **Function `generate_optuna_space`(param_specs)**
  - *Doc:* ParamSpecリストからOptuna用のparam_gridを生成する。
- **Function `generate_search_spaces`(param_specs)**
  - *Doc:* ParamSpecリストから全SearchParamSpecを生成する。
- **Function `generate_search_spaces_from_estimator`(estimator_cls)**
  - *Doc:* estimatorクラスからSearchParamSpecを一括生成する便利関数。

### File: `models\tuner.py`
- **Class `TunerConfig`**:
  - *Doc:* チューニング設定。
- **Function `tune`(model, X, y, config, groups)**
  - *Doc:* 指定された手法でハイパーパラメータを最適化して結果を返す。

### File: `optim\bayesian_optimizer.py`
- **Class `BOConfig`**:
  - *Doc:* ベイズ最適化の設定.
- **Class `BayesianOptimizer`**:
  - *Doc:* ベイズ最適化エンジン.
  - `def __init__(self, config)`
  - `def _build_kernel(self)`
  - `def fit(self, X, y)`
  - `def _fit_single(self, X, y)`
  - `def _fit_multi(self, X, Y)`
  - `def _acquisition(self, X_cand, gp, y_best)`
  - `def _ei(self, mu, sigma, y_best)`
  - `def _pi(self, mu, sigma, y_best)`
  - `def _ucb(self, mu, sigma)`
  - `def _ptr(self, mu, sigma)`
  - `def suggest(self, X_candidates, n)`
  - `def _suggest_single(self, X_cand, n)`
  - `def _kriging_believer(self, X_cand, n)`
  - `def _doe_then_bo(self, X_cand, n)`
  - `def _bo_then_doe(self, X_cand, n)`
  - `def _suggest_parego(self, X_cand, n)`
  - `def _maximin_select(X, n)`
  - `def predict(self, X)`
  - `def get_gp_info(self)`

### File: `optim\bo_visualizer.py`
- **Function `plot_pca_2d`(X_existing, X_candidates, feature_names, y_existing, top_n_arrows)**
  - *Doc:* PCA 2D散布図 + biplot矢印 + 累積寄与率.
- **Function `plot_pca_3d`(X_existing, X_candidates, feature_names, y_existing, top_n_arrows)**
  - *Doc:* PCA 3Dインタラクティブ散布図 + biplot.
- **Function `plot_pareto_front`(Y_existing, Y_candidates, objective_names, directions)**
  - *Doc:* 2目的のパレートフロント可視化.
- **Function `plot_convergence`(y_history, objective)**
  - *Doc:* 反復ごとの最良値推移を可視化.

### File: `optim\constraints.py`
- **Class `Constraint`**:
  - *Doc:* 制約の基底クラス.
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Class `RangeConstraint`** (Bases: Constraint):
  - *Doc:* 個別変数の範囲制約: lo ≤ x_i ≤ hi.
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Class `SumConstraint`** (Bases: Constraint):
  - *Doc:* 合計制約: |Σ x_i - target| ≤ tolerance.
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Class `InequalityConstraint`** (Bases: Constraint):
  - *Doc:* 線形不等式制約: Σ(coeff_i * x_i) ≤ rhs.
  - `def _compute(self, df_or_row)`
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Class `AtLeastNConstraint`** (Bases: Constraint):
  - *Doc:* 少なくともN個が閾値超: sum(x_{cols} > threshold) >= min_count.
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Class `CustomConstraint`** (Bases: Constraint):
  - *Doc:* Python式による任意制約（高度ユーザー向け）.
  - `def is_satisfied(self, row)`
  - `def mask(self, df)`
  - `def describe(self)`
- **Function `apply_constraints`(df, constraints)**
  - *Doc:* 候補DFに制約リストを適用しフィルタリング.

### File: `optim\inverse_optimizer.py`
- **Class `InverseConfig`**:
  - *Doc:* 逆解析の設定。
- **Class `InverseResult`**:
  - *Doc:* 逆解析の結果。
- **Function `run_inverse_optimization`(predict_fn, feature_names, config, progress_callback)**
  - *Doc:* 逆解析を実行する。

### File: `optim\search_space.py`
- **Class `VarType`** (Bases: str, Enum):
  - *Doc:* 変数の型.
- **Class `Variable`**:
  - *Doc:* 探索空間の個別変数定義.
  - `def n_levels(self)`
  - `def grid_values(self, n_per_dim)`
- **Class `SearchSpace`**:
  - *Doc:* 探索空間: 変数定義＋候補点生成.
  - `def __init__(self, variables)`
  - `def add(self, v)`
  - `def dim(self)`
  - `def names(self)`
  - `def estimate_grid_size(self, n_per_dim)`
  - `def auto_recommend_method(self, n_per_dim)`
  - `def generate_candidates(self, method, n_max, n_per_dim, seed)`
  - `def _generate_grid(self, n_per_dim)`
  - `def _generate_random(self, n, rng)`
  - `def _generate_lhs(self, n, rng)`
  - `def from_dataframe(cls, df, columns, margin)`

### File: `pipeline\column_selector.py`
- **Class `ColumnMeta`**:
  - *Doc:* 1列のメタ情報（ユーザーが設定）。
  - `def to_dict(self)`
  - `def from_dict(d)`
- **Class `ColumnSelectorWrapper`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* mlxtend.preprocessing.ColumnSelector のラッパー。
  - `def __init__(self, mode, columns, col_range, column_meta)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
  - `def selected_columns(self)`
  - `def get_column_meta(self, col)`
  - `def get_monotonic_constraints(self, feature_names)`
  - `def get_groups_array(self, feature_names)`
  - `def _resolve_include_columns(self, X, cols)`

### File: `pipeline\col_preprocessor.py`
- **Class `ColPreprocessConfig`**:
  - *Doc:* 列別前処理の設定。
- **Class `ColPreprocessor`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 列別前処理 Transformer。
  - `def __init__(self, config)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
  - `def column_transformer(self)`
  - `def _group_columns(self, X, detection, cfg)`
  - `def _auto_assign(self, col, detection, groups)`
  - `def _build_numeric_pipeline(self, cfg)`
  - `def _build_numeric_imputer(self, cfg)`
  - `def _build_scaler(self, scaler_name, cfg)`
  - `def _build_categorical_pipeline(self, encoder_name, cfg)`
  - `def _build_encoder(self, encoder_name, cfg)`
  - `def _build_binary_pipeline(self, cfg)`

### File: `pipeline\feature_generator.py`
- **Class `FeatureGenConfig`**:
  - *Doc:* 特徴量生成の設定。
- **Class `FeatureGenerator`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 特徴量生成 Transformer。
  - `def __init__(self, config)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
  - `def is_passthrough(self)`
  - `def n_output_features(self)`

### File: `pipeline\feature_selector.py`
- **Class `FeatureSelectorConfig`**:
  - *Doc:* 特徴量選択の設定。
- **Class `FeatureSelector`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* 特徴量選択 Transformer。
  - `def __init__(self, config, column_meta)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_feature_names_out(self, input_features)`
  - `def support_mask(self)`
  - `def _build_selector(self, cfg, X, y)`
  - `def _build_sfm_lasso(self, cfg)`
  - `def _build_sfm_ridge(self, cfg)`
  - `def _build_sfm_rf(self, cfg)`
  - `def _build_sfm_xgb(self, cfg)`
  - `def _build_sfm_custom(self, cfg)`
  - `def _build_select_percentile(self, cfg)`
  - `def _build_select_kbest(self, cfg)`
  - `def _resolve_score_func(self, cfg)`
  - `def _build_relieff(self, cfg)`
  - `def _build_boruta(self, cfg)`
  - `def _build_genetic(self, cfg, X, y)`
  - `def _build_group_lasso(self, cfg)`
  - `def _build_groups_from_meta(self)`
  - `def _get_default_rf(self, cfg)`
- **Class `_GroupLassoSelector`** (Bases: BaseEstimator, TransformerMixin):
  - *Doc:* group_lasso.GroupLasso を sklearn TransformerMixin に適合させるラッパー。
  - `def __init__(self, alpha, groups)`
  - `def fit(self, X, y)`
  - `def transform(self, X, y)`
  - `def get_support(self)`

### File: `pipeline\pipeline_builder.py`
- **Class `PipelineConfig`**:
  - *Doc:* 5 段階 ML パイプラインの設定。
- **Function `build_pipeline`(config)**
  - *Doc:* PipelineConfig から sklearn Pipeline を構築して返す。
- **Function `apply_monotonic_constraints`(estimator, column_meta, feature_names)**
  - *Doc:* estimatorの種類に応じて単調性制約を適用する。
- **Function `extract_group_array`(column_meta, feature_names)**
  - *Doc:* ColumnMeta のグループ情報から GroupCV 等が使用できる整数配列を返す。

### File: `pipeline\pipeline_grid.py`
- **Class `PipelineCombination`** (Bases: NamedTuple):
  - *Doc:* 1つのパイプライン組み合わせを表す名前付きタプル。
- **Class `PipelineGridConfig`**:
  - *Doc:* 複数選択肢から全組み合わせを生成するグリッド設定。
- **Function `generate_pipeline_grid`(grid_config, max_combinations)**
  - *Doc:* PipelineGridConfig の各選択肢の全デカルト積から
- **Function `count_combinations`(grid_config)**
  - *Doc:* PipelineGridConfig から生成される組み合わせ数を返す（Pipeline は構築しない）。

### File: `session\version_manager.py`
- **Class `ExperimentRecord`**:
  - *Doc:* 1回の実験結果を表すデータクラス。
- **Class `VersionManager`**:
  - *Doc:* 実験設定・結果のバージョン管理を行うクラス。
  - `def __init__(self, db_path)`
  - `def _connect(self)`
  - `def _init_db(self)`
  - `def compute_hash(hyperparams, preprocess_params)`
  - `def save(self, record)`
  - `def save_from_automl_result(self, ar, state, exp_name)`
  - `def list_experiments(self, limit, task_type)`
  - `def get_experiment(self, exp_hash)`
  - `def delete_experiment(self, exp_hash)`
  - `def update_notes(self, exp_hash, notes)`

### File: `ui\param_schema.py`
- **Class `ParamSpec`**:
  - *Doc:* 1つのパラメータのUIウィジェット仕様。
  - `def to_dict(self)`
- **Function `introspect_params`(cls)**
  - *Doc:* 任意のPythonクラスの__init__パラメータを自動解析し、
- **Function `introspect_adapter`(adapter_instance)**
  - *Doc:* BaseChemAdapterサブクラスのインスタンスからパラメータを取得。
- **Function `introspect_adapter_class`(adapter_cls)**
  - *Doc:* BaseChemAdapterサブクラスからパラメータを取得。
- **Function `apply_params`(specs, user_values)**
  - *Doc:* ParamSpecリストとユーザー入力値から、コンストラクタに渡す
- **Function `get_basic_specs`(specs)**
  - *Doc:* basicグループのParamSpecのみ返す。
- **Function `get_advanced_specs`(specs)**
  - *Doc:* advancedグループのParamSpecのみ返す。

### File: `utils\config.py`
- **Class `AppConfig`**:
  - *Doc:* アプリケーション全体の設定をまとめるデータクラス。

### File: `utils\cv_recommender.py`
- **Class `CVRecommendation`**:
  - *Doc:* CV推薦の結果。
- **Function `recommend_cv_strategy`(X, y, metadata)**
  - *Doc:* データ特性に基づき最適なCV戦略を推薦する。

### File: `utils\optional_import.py`
- **Function `safe_import`(module_name, alias)**
  - *Doc:* 指定モジュールを安全にimportする。失敗した場合は None を返す。
- **Function `is_available`(name)**
  - *Doc:* 指定ライブラリ（またはその機能グループ）が利用可能か返す。
- **Function `require`(name, feature)**
  - *Doc:* ライブラリが利用可能でなければ RuntimeError を送出する。
- **Function `get_availability_report`()**
  - *Doc:* 現在判明しているライブラリ可用性の一覧を返す。
- **Function `probe_all_optional_libraries`()**
  - *Doc:* 使用するオプショナルライブラリを一括で試みてキャッシュを更新する。
## FRONTEND_NICEGUI Directory

### File: `main.py`
- **Function `main_page`()**
- **Function `help_page`()**
- **Function `help_descriptors_page`()**

### File: `components\analysis_runner.py`
- **Class `AnalysisCancelled`** (Bases: Exception):
  - *Doc:* 解析キャンセル時に送出される例外。

### File: `components\auto_params_ui.py`
- **Function `render_param_editor`(specs, title)**
  - *Doc:* ParamSpecリストからNiceGUI UIを自動生成する。
- **Function `render_model_param_editor`(model_cls, title)**
  - *Doc:* sklearn estimatorクラスからパラメータUIを自動生成する便利関数。
- **Function `render_adapter_param_editor`(adapter_cls, title)**
  - *Doc:* ChemAdapterクラスからパラメータUIを自動生成する便利関数。

### File: `components\batch_predict_tab.py`
- **Function `render_batch_predict_tab`(state)**
  - *Doc:* バッチ予測タブを描画する。

### File: `components\bayesian_opt_ui.py`
- **Function `render_bayesian_opt_ui`(state)**
  - *Doc:* ベイズ最適化UIをレンダリング。

### File: `components\column_meta_editor.py`
- **Function `render_column_meta_editor`(state, df)**
  - *Doc:* 説明変数ごとのメタ情報設定UIをレンダリングする。
- **Function `build_column_meta_dict`(state)**
  - *Doc:* state["column_meta"] から ColumnMeta オブジェクト辞書を構築して返す。
- **Function `extract_monotonic_from_column_meta`(state)**
  - *Doc:* state["column_meta"] から単調性制約辞書（{列名: 1/-1}）を取得する。

### File: `components\constraint_ui_helpers.py`
- **Function `ui_constraints_to_backend`(constraint_items)**
  - *Doc:* UI制約リストをバックエンドのConstraintオブジェクトに変換。
- **Function `validate_constraints`(constraint_items, df)**
  - *Doc:* 制約の妥当性を検証。
- **Function `describe_constraint`(item)**
  - *Doc:* 制約を人間が読める日本語文で返す。
- **Function `save_template`(name, constraint_items, tags)**
  - *Doc:* 制約テンプレートをJSONとして保存。
- **Function `load_template`(path)**
  - *Doc:* 制約テンプレートを読み込み。
- **Function `list_templates`()**
  - *Doc:* 保存済みテンプレート一覧を返す。
- **Function `delete_template`(path)**
  - *Doc:* テンプレートを削除。
- **Function `friendly_error`(error)**
  - *Doc:* 技術的エラーをユーザーフレンドリーなメッセージに変換。
- **Function `get_column_stats`(df, col)**
  - *Doc:* 列の基本統計量を返す。

### File: `components\cv_config_ui.py`
- **Function `render_cv_config`(state)**
  - *Doc:* 交差検証設定UIをインラインでレンダリングする。

### File: `components\data_tab.py`
- **Function `render_data_tab`(state)**
  - *Doc:* データ設定タブ全体を描画する。

### File: `components\descriptor_catalog.py`
- **Function `get_rdkit_catalog`()**
  - *Doc:* RDKit記述子をカテゴリ別に返す。ソース: rdkit_adapter.py _DESCRIPTOR_JP_META
- **Function `get_xtb_catalog`()**
  - *Doc:* XTB量子化学記述子カタログ。ソース: xtb_adapter.py _XTB_DESCRIPTORS
- **Function `get_group_contrib_catalog`()**
  - *Doc:* 原子団寄与法(Joback法)カタログ。ソース: group_contrib_adapter.py
- **Function `get_skfp_catalog`()**
  - *Doc:* scikit-fingerprints カタログ。ソース: skfp_adapter.py _FP_CONFIGS
- **Function `get_mordred_catalog`()**
  - *Doc:* Mordred厳選記述子カタログ。ソース: mordred_adapter.py SELECTED_DESCRIPTORS
- **Function `get_molfeat_catalog`()**
  - *Doc:* Molfeat計算機タイプカタログ。ソース: molfeat_adapter.py _CALCULATOR_TYPES
- **Function `get_cosmo_catalog`()**
  - *Doc:* COSMO-RS記述子カタログ。ソース: cosmo_adapter.py _COSMO_DESCRIPTORS
- **Function `get_descriptastorus_catalog`()**
  - *Doc:* DescriptaStorus（Merck製 200+2D記述子）カタログ。
- **Function `get_padel_catalog`()**
  - *Doc:* PaDEL（CDK由来 1600+2D記述子）カタログ。主要グループのみ。
- **Function `get_catalog`(engine_name)**
  - *Doc:* エンジン名からカタログを取得する。

### File: `components\descriptor_help_page.py`
- **Function `render_descriptor_help`()**
  - *Doc:* 推奨記述子の全データを一覧表示するヘルプページ。

### File: `components\descriptor_plugins_ui.py`
- **Function `render_descriptor_plugins`(state)**
  - *Doc:* SMILES記述子パネルの完全なUI。

### File: `components\descriptor_selector_dialog.py`
- **Function `open_descriptor_detail_dialog`(engine_name, state)**
  - *Doc:* 指定エンジンの記述子をカテゴリ別に展開し、
- **Function `render_selected_descriptors_panel`(state)**
  - *Doc:* 現在選択されている記述子の一覧を表示。
- **Function `render_descriptor_sets_panel`(state)**
  - *Doc:* 複数の記述子セット(パターン)をカード型で管理するUI。

### File: `components\descriptor_status_bar.py`
- **Function `render_descriptor_status_bar`(state)**
  - *Doc:* 記述子セット常時表示バーを描画する。

### File: `components\dialog_manager.py`
- **Class `StateSnapshot`**:
  - *Doc:* state dict の部分スナップショットを取得・復元する。
  - `def __init__(self, state, keys)`
  - `def take(self)`
  - `def restore(self)`
- **Function `create_settings_dialog`()**
  - *Doc:* 汎用設定ダイアログを生成して返す。
- **Function `render_settings_summary`()**
  - *Doc:* メイン画面に表示するコンパクトな設定サマリー+ダイアログ呼び出しボタン。

### File: `components\doe_tab.py`
- **Function `render_doe_tab`(app_state)**
  - *Doc:* DoEタブ全体をレンダリングする。

### File: `components\eda_panel.py`
- **Function `render_eda_panel`(state)**
  - *Doc:* EDAパネルをレンダリング。

### File: `components\estimator_config_dialog.py`
- **Class `EstimatorConfig`**:
  - *Doc:* Estimator設定の統合データ。
  - `def __init__(self, model_key, model_cls, default_params, grid_space, optuna_space)`
  - `def to_dict(self)`
- **Class `EstimatorConfigDialog`**:
  - *Doc:* 動的Estimator設定ダイアログ。
  - `def __init__(self, model_key, model_cls)`
  - `def _init_defaults(self, initial)`
  - `def open(self)`
  - `def _build_dialog(self)`
  - `def _render_default_tab(self)`
  - `def _render_default_widget(self, spec)`
  - `def _render_grid_tab(self)`
  - `def _update_grid_values(self, name, values_str)`
  - `def _render_optuna_tab(self)`
  - `def _update_optuna_choices(self, name, choices_str)`
  - `def _on_save_click(self)`
- **Function `render_model_config_panel`(model_entries, state)**
  - *Doc:* モデル選択リストの各モデルに設定ボタンを追加する。

### File: `components\feature_set_manager.py`
- **Function `render_feature_set_manager`(state)**
  - *Doc:* 特徴量セット × パイプライン マトリクス管理UIを描画する。

### File: `components\internal_llm_ui.py`
- **Function `render_internal_llm_tab`(state)**
  - *Doc:* 内部AI（HuggingFace）設定タブ全体を描画する。

### File: `components\interpretation_panel.py`
- **Function `render_interpretation_panel`(ar, state)**
  - *Doc:* 解釈性パネル全体を描画する。

### File: `components\inverse_analysis_tab.py`
- **Function `render_inverse_analysis_tab`(state)**
  - *Doc:* 逆解析タブの全UIを描画する。

### File: `components\leakage_check_ui.py`
- **Function `render_leakage_check_panel`(state)**
  - *Doc:* リーケージ事前チェックパネルを描画する。

### File: `components\pipeline_config_ui.py`
- **Function `render_pipeline_config`(state)**
  - *Doc:* パイプライン全設定UIをインラインでレンダリングする。

### File: `components\post_analysis_config.py`
- **Function `render_post_analysis_config`(state)**
  - *Doc:* 解析後の自動処理設定UIを描画する。

### File: `components\report_generator.py`
- **Function `render_report_tab`(state)**
  - *Doc:* レポート生成タブを描画する。

### File: `components\results_tab.py`
- **Function `render_results_tab`(state)**
  - *Doc:* 結果確認タブ全体を描画する。

### File: `components\results_tab_extras.py`

### File: `components\settings_checker.py`
- **Function `render_settings_checker`(state)**
  - *Doc:* 設定整合性チェッカーパネルを描画する。

### File: `components\smiles_hover.py`
- **Function `smiles_to_svg_b64`(smiles, width, height)**
  - *Doc:* SMILES → base64 エンコードされた SVG data URI。RDKit未インストール時は空文字列。
- **Function `smiles_to_png_b64`(smiles, width, height)**
  - *Doc:* SMILES → base64 PNG data URI（SVGフォールバック）。
- **Function `render_smiles_table`(df, smiles_col, max_rows, img_size, height)**
  - *Doc:* SMILES列にホバーで2D構造ポップアップするHTMLテーブルをNiceGUI内に描画。
- **Function `add_smiles_hover_to_plotly`(fig, smiles_list, label_list, img_size)**
  - *Doc:* Plotly散布図のホバーにSMILES 2D画像を埋め込む。

### File: `components\tuning_tab.py`
- **Function `render_tuning_tab`(state)**
  - *Doc:* ハイパーパラメータチューニングタブを描画する。

### File: `pages\experiment_comparison.py`
- **Function `render_experiment_comparison`(state)**
  - *Doc:* 実験比較ダッシュボードを描画する。

### File: `pages\export_panel.py`
- **Function `render_export_panel`(state)**
  - *Doc:* エクスポートパネルを描画する。

### File: `pages\model_manager.py`
- **Function `render_model_manager`()**
  - *Doc:* モデル管理（ダウンロード）タブの描画
## TESTS Directory

### File: `conftest.py`
- **Function `random_seed`()**
- **Function `small_regression_df`()**
  - *Doc:* セッションスコープの小さい回帰DataFrameフィクスチャ。
- **Function `small_classification_df`()**
  - *Doc:* セッションスコープの分類DataFrameフィクスチャ。

### File: `debug_adapters.py`
- **Function `test_adapter`(module_path, class_name, kwargs, status_tag)**
  - *Doc:* 1アダプタをテストして結果を返す。
- **Function `main`()**

### File: `test_automl_extra.py`
- **Class `TestAutoMLResult`**:
  - `def test_construction(self)`
  - `def test_with_oof(self)`
- **Class `TestAutoMLEngine`**:
  - `def test_init_defaults(self)`
  - `def test_init_custom(self)`
  - `def test_detect_task_regression(self)`
  - `def test_detect_task_classification(self)`
  - `def test_detect_task_explicit(self)`
  - `def test_run_minimal(self)`

### File: `test_automl_integration.py`
- **Class `TestAutoMLInit`**:
  - `def test_default_init(self)`
  - `def test_custom_init(self)`
- **Class `TestAutoMLRegression`**:
  - `def test_basic_run(self, regression_df)`
  - `def test_model_scores_populated(self, regression_df)`
  - `def test_elapsed_time(self, regression_df)`
  - `def test_progress_callback(self, regression_df)`
- **Class `TestAutoMLClassification`**:
  - `def test_basic_classification(self, classification_df)`
  - `def test_auto_task_detection(self, classification_df)`
- **Class `TestAutoMLResult`**:
  - `def test_result_fields(self, regression_df)`
- **Function `regression_df`()**
  - *Doc:* 回帰用のシンプルなDataFrame
- **Function `classification_df`()**
  - *Doc:* 分類用のシンプルなDataFrame

### File: `test_base_and_feature_gen.py`
- **Class `TestDescriptorResult`**:
  - `def test_success_rate_all_success(self)`
  - `def test_success_rate_partial(self)`
  - `def test_success_rate_empty(self)`
  - `def test_n_descriptors(self)`
- **Class `TestDescriptorMetadata`**:
  - `def test_basic(self)`
  - `def test_binary(self)`
- **Class `DummyAdapter`** (Bases: BaseChemAdapter):
  - *Doc:* テスト用のダミーアダプタ。
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
- **Class `UnavailableAdapter`** (Bases: BaseChemAdapter):
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
- **Class `TestBaseChemAdapter`**:
  - `def test_repr_available(self)`
  - `def test_repr_unavailable(self)`
  - `def test_require_available_raises(self)`
  - `def test_get_descriptor_names_default(self)`
  - `def test_get_descriptors_metadata_default(self)`
  - `def test_compute(self)`
- **Class `TestFeatureGenConfig`**:
  - `def test_defaults(self)`
- **Class `TestFeatureGenerator`**:
  - `def sample_df(self)`
  - `def test_passthrough(self, sample_df)`
  - `def test_polynomial(self, sample_df)`
  - `def test_interaction_only(self, sample_df)`
  - `def test_get_feature_names_out_passthrough(self, sample_df)`
  - `def test_get_feature_names_out_polynomial(self, sample_df)`
  - `def test_n_output_features_passthrough(self, sample_df)`
  - `def test_n_output_features_polynomial(self, sample_df)`
  - `def test_numpy_input(self)`
  - `def test_with_bias(self, sample_df)`

### File: `test_bayesian_opt.py`
- **Class `TestVariable`**:
  - *Doc:* Variable dataclassのテスト.
  - `def test_continuous_variable(self)`
  - `def test_discrete_variable(self)`
  - `def test_categorical_variable(self)`
  - `def test_missing_lo_hi_raises(self)`
  - `def test_lo_greater_than_hi_raises(self)`
  - `def test_discrete_no_step_raises(self)`
  - `def test_categorical_no_categories_raises(self)`
  - `def test_grid_values_continuous(self)`
  - `def test_grid_values_discrete(self)`
- **Class `TestSearchSpace`**:
  - *Doc:* SearchSpaceのテスト.
  - `def _make_space(self)`
  - `def test_dim(self)`
  - `def test_names(self)`
  - `def test_estimate_grid_size(self)`
  - `def test_generate_grid(self)`
  - `def test_generate_random(self)`
  - `def test_generate_lhs(self)`
  - `def test_auto_method_small(self)`
  - `def test_from_dataframe(self)`
  - `def test_empty_raises(self)`
- **Class `TestConstraints`**:
  - *Doc:* 制約処理のテスト.
  - `def _make_df(self)`
  - `def test_range_constraint(self)`
  - `def test_sum_constraint(self)`
  - `def test_sum_constraint_strict(self)`
  - `def test_inequality_constraint(self)`
  - `def test_inequality_ge(self)`
  - `def test_at_least_one(self)`
  - `def test_at_least_one_none(self)`
  - `def test_custom_constraint(self)`
  - `def test_apply_constraints(self)`
  - `def test_describe(self)`
- **Class `TestBayesianOptimizer`**:
  - *Doc:* ベイズ最適化エンジンのテスト.
  - `def sample_data(self)`
  - `def candidates(self)`
  - `def test_fit_single(self, sample_data)`
  - `def test_predict(self, sample_data)`
  - `def test_suggest_single(self, sample_data, candidates)`
  - `def test_suggest_kriging_believer(self, sample_data, candidates)`
  - `def test_kb_candidates_diverse(self, sample_data, candidates)`
  - `def test_pi_acquisition(self, sample_data, candidates)`
  - `def test_ucb_acquisition(self, sample_data, candidates)`
  - `def test_ptr_acquisition(self, sample_data, candidates)`
  - `def test_doe_then_bo(self, sample_data, candidates)`
  - `def test_bo_then_doe(self, sample_data, candidates)`
  - `def test_suggest_dataframe(self, sample_data)`
  - `def test_multi_objective_parego(self)`
  - `def test_not_fitted_raises(self)`
  - `def test_not_fitted_predict_raises(self)`

### File: `test_bayesian_optimizer_comprehensive.py`
- **Class `TestBOConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestBayesianOptimizerFit`**:
  - `def test_fit_basic(self, reg_data)`
  - `def test_fit_dataframe(self, reg_data)`
- **Class `TestAcquisitionFunctions`**:
  - `def test_ei_minimize(self, reg_data, candidates)`
  - `def test_ei_maximize(self, reg_data, candidates)`
  - `def test_pi(self, reg_data, candidates)`
  - `def test_ucb(self, reg_data, candidates)`
  - `def test_ptr(self, reg_data, candidates)`
- **Class `TestBatchStrategies`**:
  - `def test_single(self, reg_data, candidates)`
  - `def test_kriging_believer(self, reg_data, candidates)`
  - `def test_doe_then_bo(self, reg_data, candidates)`
  - `def test_bo_then_doe(self, reg_data, candidates)`
- **Class `TestPredict`**:
  - `def test_predict(self, reg_data, candidates)`
  - `def test_predict_before_fit(self, candidates)`
- **Class `TestKernelTypes`**:
  - `def test_matern(self, reg_data, candidates)`
  - `def test_dotproduct(self, reg_data, candidates)`
- **Class `TestGPInfo`**:
  - `def test_gp_info(self, reg_data)`
  - `def test_gp_info_before_fit(self)`
- **Class `TestSuggestDataFrame`**:
  - `def test_suggest_returns_dataframe(self, reg_data, candidates)`
  - `def test_suggest_before_fit(self, candidates)`
- **Class `TestMaximinSelect`**:
  - `def test_basic(self)`
  - `def test_n_ge_len(self)`
- **Class `TestMultiObjective`**:
  - `def test_parego(self, candidates)`
- **Function `reg_data`()**
- **Function `candidates`()**

### File: `test_bayesian_optimizer_extra.py`
- **Class `TestBOConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestBayesianOptimizerBasic`**:
  - `def test_fit_and_predict(self)`
  - `def test_fit_with_dataframe(self)`
  - `def test_predict_before_fit_raises(self)`
  - `def test_suggest_before_fit_raises(self)`
  - `def test_get_gp_info_before_fit(self)`
  - `def test_get_gp_info_after_fit(self)`
- **Class `TestKernelBuilding`**:
  - `def test_default_kernel(self)`
  - `def test_matern_kernel(self)`
  - `def test_dotproduct_kernel(self)`
- **Class `TestAcquisitionFunctions`**:
  - `def test_ei_minimize(self)`
  - `def test_ei_maximize(self)`
  - `def test_pi(self)`
  - `def test_pi_maximize(self)`
  - `def test_ucb_minimize(self)`
  - `def test_ucb_maximize(self)`
  - `def test_ptr(self)`
  - `def test_ptr_no_range_raises(self)`
  - `def test_unknown_acquisition_raises(self)`
- **Class `TestBatchStrategies`**:
  - `def test_single(self)`
  - `def test_kriging_believer(self)`
  - `def test_doe_then_bo(self)`
  - `def test_bo_then_doe(self)`
  - `def test_n_candidates_one(self)`
  - `def test_suggest_with_dataframe(self)`
- **Class `TestMultiObjective`**:
  - `def test_parego_basic(self)`
  - `def test_multi_predict(self)`
  - `def test_multi_gp_info(self)`
- **Class `TestMaximinSelect`**:
  - `def test_basic(self)`
  - `def test_n_greater_than_length(self)`

### File: `test_benchmark.py`
- **Class `TestModelScore`**:
  - *Doc:* T-BM-001: ModelScore データクラスのテスト。
  - `def test_to_dict_filters_none(self)`
  - `def test_task_field(self)`
- **Class `TestBenchmarkResult`**:
  - *Doc:* T-BM-002: BenchmarkResult データクラスのテスト。
  - `def test_to_dataframe(self)`
  - `def test_best_regression(self)`
  - `def test_best_classification(self)`
  - `def test_best_empty(self)`
- **Class `TestEvaluateRegression`**:
  - *Doc:* T-BM-003: evaluate_regression のテスト。
  - `def test_returns_model_score(self, reg_data)`
  - `def test_rmse_positive(self, reg_data)`
  - `def test_r2_perfect(self, reg_data)`
  - `def test_cv_fields(self, reg_data)`
  - `def test_mae_positive(self, reg_data)`
- **Class `TestEvaluateClassification`**:
  - *Doc:* T-BM-004: evaluate_classification のテスト。
  - `def test_returns_model_score(self, cls_data)`
  - `def test_perfect_accuracy(self, cls_data)`
  - `def test_roc_auc_with_proba(self, cls_data)`
  - `def test_no_prob_roc_none(self, cls_data)`
- **Class `TestComputeLearningCurve`**:
  - *Doc:* T-BM-005: compute_learning_curve のテスト。
  - `def test_returns_dict_with_keys(self, reg_data)`
  - `def test_train_sizes_length(self, reg_data)`
  - `def test_scores_shape_match(self, reg_data)`
- **Class `TestBenchmarkModels`**:
  - *Doc:* T-BM-006: benchmark_models のテスト。
  - `def test_regression(self, reg_data)`
  - `def test_classification(self, cls_data)`
  - `def test_best_model(self, reg_data)`
- **Function `reg_data`()**
- **Function `cls_data`()**

### File: `test_benchmark_datasets_extra.py`
- **Class `TestListBenchmarkDatasets`**:
  - `def test_returns_list(self)`
  - `def test_has_required_keys(self)`
  - `def test_known_ids(self)`
- **Class `TestBenchmarkURLs`**:
  - `def test_keys(self)`
  - `def test_urls_are_strings(self)`
- **Class `TestLoadBenchmark`**:
  - `def test_unknown_name_raises(self)`
  - `def test_load_from_cache(self)`
  - `def test_load_with_mock_download(self)`
  - `def test_download_failure(self)`

### File: `test_benchmark_extra.py`
- **Class `TestModelScore`**:
  - `def test_to_dict_regression(self)`
  - `def test_to_dict_classification(self)`
- **Class `TestBenchmarkResult`**:
  - `def test_to_dataframe(self)`
  - `def test_best_regression(self)`
  - `def test_best_classification(self)`
  - `def test_best_empty(self)`
- **Class `TestEvaluateRegression`**:
  - `def test_basic(self)`
  - `def test_with_cv(self)`
- **Class `TestEvaluateClassification`**:
  - `def test_basic(self)`
  - `def test_with_proba(self)`
  - `def test_multiclass_proba(self)`
- **Class `TestBenchmarkModels`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_multiple_models(self)`

### File: `test_bo_visualizer.py`
- **Class `TestPlotPCA2D`**:
  - `def test_basic_ndarray(self)`
  - `def test_with_dataframe(self)`
  - `def test_with_candidates(self)`
  - `def test_with_y_values(self)`
  - `def test_without_y(self)`
  - `def test_custom_feature_names(self)`
  - `def test_top_n_arrows(self)`
  - `def test_full_with_all_options(self)`
- **Class `TestPlotPCA3D`**:
  - `def test_basic(self)`
  - `def test_with_candidates(self)`
  - `def test_with_y(self)`
  - `def test_without_y(self)`
  - `def test_with_dataframe(self)`
  - `def test_feature_names_and_arrows(self)`
  - `def test_low_dim_input(self)`
- **Class `TestPlotParetoFront`**:
  - `def test_basic_minimize(self)`
  - `def test_with_objective_names(self)`
  - `def test_with_candidates(self)`
  - `def test_with_directions(self)`
  - `def test_with_dataframe(self)`
- **Class `TestParetoEfficient`**:
  - `def test_basic_min(self)`
  - `def test_basic_max(self)`
  - `def test_no_domination(self)`
- **Class `TestPlotConvergence`**:
  - `def test_minimize(self)`
  - `def test_maximize(self)`
  - `def test_single_point(self)`
  - `def test_long_history(self)`

### File: `test_bo_visualizer_comprehensive.py`
- **Class `TestPlotPCA2D`**:
  - `def test_basic(self, data_5d)`
  - `def test_with_candidates(self, data_5d)`
  - `def test_dataframe(self, data_5d)`
  - `def test_no_y(self, data_5d)`
- **Class `TestPlotPCA3D`**:
  - `def test_basic(self, data_5d)`
  - `def test_with_candidates(self, data_5d)`
- **Class `TestParetoFront`**:
  - `def test_basic(self)`
  - `def test_with_candidates(self)`
  - `def test_custom_directions(self)`
- **Class `TestParetoEfficient`**:
  - `def test_basic(self)`
  - `def test_all_min(self)`
- **Class `TestConvergence`**:
  - `def test_minimize(self)`
  - `def test_maximize(self)`
- **Function `data_5d`()**

### File: `test_charge_config.py`
- **Class `TestMoleculeChargeConfigDefaults`**:
  - `def test_default_values(self, default_cfg)`
  - `def test_uhf_property_closed_shell(self, default_cfg)`
  - `def test_uhf_property_radical(self)`
  - `def test_uhf_property_triplet(self)`
  - `def test_to_xtb_args_neutral_closed_shell(self, default_cfg)`
  - `def test_to_xtb_args_cation_radical(self)`
  - `def test_to_xtb_args_charge_override(self, default_cfg)`
  - `def test_to_xtb_args_anion(self)`
- **Class `TestMoleculeChargeConfigValidation`**:
  - `def test_spin_multiplicity_zero_raises(self)`
  - `def test_spin_multiplicity_negative_raises(self)`
  - `def test_formal_charge_too_large_raises(self)`
  - `def test_formal_charge_too_negative_raises(self)`
  - `def test_boundary_charge_plus_10(self)`
  - `def test_boundary_charge_minus_10(self)`
  - `def test_spin_multiplicity_4(self)`
- **Class `TestMoleculeChargeConfigPresets`**:
  - `def test_for_radical_default_neutral(self)`
  - `def test_for_radical_charged(self)`
  - `def test_at_physiological_ph(self)`
- **Class `TestChargeConfigStore`**:
  - `def test_get_config_returns_default_for_unknown_smiles(self, store)`
  - `def test_set_and_get_per_molecule(self, store)`
  - `def test_per_molecule_overrides_default(self, store)`
  - `def test_resolve_charge_from_smiles_cation(self, store)`
  - `def test_resolve_charge_from_smiles_anion(self, store)`
  - `def test_resolve_charge_from_smiles_dianion(self, store)`
  - `def test_resolve_charge_manual_override(self, store)`
  - `def test_resolve_spin_default(self, store)`
  - `def test_resolve_spin_per_molecule(self, store)`
  - `def test_per_molecule_does_not_affect_others(self, store)`
- **Class `TestReadSmilesFormalCharge`**:
  - `def test_neutral_ethanol(self)`
  - `def test_cation_ammonium(self)`
  - `def test_anion_acetate(self)`
  - `def test_invalid_smiles_returns_0(self)`
  - `def test_empty_string_returns_0(self)`
  - `def test_none_returns_0(self)`
  - `def test_zwitterion(self)`
  - `def test_large_molecule_neutral(self)`
- **Class `TestApplyProtonation`**:
  - `def setup_method(self)`
  - `def test_as_is_returns_unchanged(self)`
  - `def test_as_is_anion_unchanged(self)`
  - `def test_neutral_removes_charge_acetate(self)`
  - `def test_neutral_removes_charge_ammonium(self)`
  - `def test_neutral_already_neutral(self)`
  - `def test_neutral_salt_desalted(self)`
  - `def test_auto_ph_without_unipka_falls_back_to_neutral(self)`
  - `def test_auto_ph_permission_error_falls_back_to_neutral(self)`
  - `def test_unknown_mode_returns_original(self)`
  - `def test_empty_string_returns_empty(self)`
  - `def test_none_input_returns_none(self)`
  - `def test_invalid_smiles_fallback(self)`
- **Class `TestApplyProtonationBatch`**:
  - `def test_batch_same_length(self)`
  - `def test_batch_as_is_identical(self)`
  - `def test_batch_with_invalid_smiles(self)`
- **Class `TestGetProtonationStateInfo`**:
  - `def test_returns_dict_keys(self)`
  - `def test_import_error_message(self)`
  - `def test_permission_error_message(self)`
  - `def test_generic_exception_shows_error(self)`
  - `def test_with_mock_pka_acidic_anion(self)`
  - `def test_with_mock_pka_basic_cation(self)`
  - `def test_with_mock_pka_neutral(self)`
  - `def test_with_mock_pka_zwitterion(self)`
- **Class `TestGetUnipkaModel`**:
  - `def setup_method(self)`
  - `def test_cache_reuse(self)`
  - `def test_permission_error_retried(self)`
  - `def test_permission_error_exhausted_raises(self)`
  - `def test_thread_safety(self)`
- **Class `TestXTBArgsIntegration`**:
  - *Doc:* XTBAdapterが correct な --chrg/--uhf を生成するかモックで確認。
  - `def test_store_resolves_correct_charge(self)`
  - `def test_xtb_args_from_store(self)`
- **Function `default_cfg`()**
- **Function `store`()**

### File: `test_charge_config_extra.py`
- **Class `TestMoleculeChargeConfig`**:
  - `def test_defaults(self)`
  - `def test_uhf(self)`
  - `def test_to_xtb_args_default(self)`
  - `def test_to_xtb_args_with_spin(self)`
  - `def test_to_xtb_args_charge_override(self)`
  - `def test_invalid_spin(self)`
  - `def test_invalid_charge(self)`
  - `def test_default_factory(self)`
  - `def test_for_radical(self)`
  - `def test_at_physiological_ph(self)`
- **Class `TestChargeConfigStore`**:
  - `def test_defaults(self)`
  - `def test_get_config_default(self)`
  - `def test_set_per_molecule(self)`
  - `def test_get_config_fallback(self)`
  - `def test_resolve_spin(self)`
  - `def test_resolve_charge_auto(self)`
  - `def test_resolve_charge_manual(self)`
- **Class `TestReadSmilesFormalCharge`**:
  - `def test_neutral(self)`
  - `def test_invalid_smiles(self)`

### File: `test_charge_extended.py`
- **Class `TestParseXtbOutput`**:
  - *Doc:* _parse_xtb_output() の単体テスト。
  - `def _import_parser(self)`
  - `def test_normal_output_total_energy(self)`
  - `def test_normal_output_homo_lumo_gap(self)`
  - `def test_normal_output_homo_energy(self)`
  - `def test_normal_output_lumo_energy(self)`
  - `def test_normal_output_koopmans_derived(self)`
  - `def test_normal_output_dipole(self)`
  - `def test_normal_output_mulliken_charges(self)`
  - `def test_partial_output(self)`
  - `def test_empty_output(self)`
  - `def test_no_homo_lumo_lines(self)`
  - `def test_mulliken_only_output(self)`
  - `def test_garbage_input_no_crash(self)`
- **Class `TestSmilesToXyz`**:
  - *Doc:* _smiles_to_xyz() の単体テスト。
  - `def _import(self)`
  - `def test_methane(self)`
  - `def test_ethanol(self)`
  - `def test_cation_ammonium(self)`
  - `def test_anion_acetate(self)`
  - `def test_invalid_smiles_returns_none(self)`
  - `def test_empty_smiles_returns_none(self)`
  - `def test_xyz_format_has_header(self)`
  - `def test_xyz_coordinates_are_floats(self)`
  - `def test_benzene_6_carbons(self)`
- **Class `TestRDKitAdapterGasteigerCharges`**:
  - `def test_gasteiger_columns_present(self)`
  - `def test_gasteiger_disabled_no_columns(self)`
  - `def test_gasteiger_q_range_nonnegative(self)`
  - `def test_gasteiger_acetic_acid_has_large_range(self)`
  - `def test_gasteiger_with_charge_config_store(self)`
  - `def test_invalid_smiles_in_batch(self)`
  - `def test_physicochemical_descriptors_present(self)`
- **Class `TestChargeConfigStoreEdgeCases`**:
  - `def test_overwrite_per_molecule(self)`
  - `def test_empty_string_smiles(self)`
  - `def test_many_per_molecule_settings(self)`
  - `def test_default_is_independent_copy(self)`
  - `def test_resolve_spin_unknown_smiles(self)`
- **Class `TestMoleculeChargeConfigEdgeCases`**:
  - `def test_high_spin_multiplicity(self)`
  - `def test_formal_charge_boundary_exact_10(self)`
  - `def test_formal_charge_boundary_11_raises(self)`
  - `def test_to_xtb_args_high_spin(self)`
  - `def test_charge_override_zero_still_sets_chrg(self)`
  - `def test_default_factory_each_call_independent(self)`
- **Class `TestReadSmilesFormalChargeEdgeCases`**:
  - `def test_multi_cation(self)`
  - `def test_phosphate_anion(self)`
  - `def test_neutral_aromatic(self)`
  - `def test_numeric_string_returns_0(self)`
  - `def test_special_chars_returns_0(self)`
- **Class `TestProtonationMaxModes`**:
  - `def test_max_deprotonate_carboxylic_acid(self)`
  - `def test_max_protonate_amine(self)`
  - `def test_max_deprotonate_phenol(self)`
  - `def test_max_protonate_pyridine(self)`
  - `def test_neutral_aspirin(self)`
  - `def test_neutral_sodium_acetate_salt(self)`
- **Class `TestXTBAdapterComputeMock`**:
  - *Doc:* xtbバイナリ不要 — subprocessをモックしてcompute()のロジックをテスト
  - `def test_compute_passes_chrg_arg(self)`
  - `def test_compute_timeout_handled(self)`

### File: `test_chem.py`
- **Function `rdkit_adapter`()**
- **Function `test_rdkit_adapter_name`(rdkit_adapter)**
- **Function `test_rdkit_adapter_availability`(rdkit_adapter)**
- **Function `test_compute_descriptors`(rdkit_adapter)**
  - *Doc:* 有効なSMILESに対して記述子が計算されること。
- **Function `test_compute_with_invalid_smiles`(rdkit_adapter)**
  - *Doc:* 無効なSMILESが含まれる場合に正しく処理され、failed_indices が記録されること。
- **Function `test_get_descriptor_names`(rdkit_adapter)**
  - *Doc:* 記述子名のリストが正しく取得できること。
- **Function `test_descriptor_metadata_rdkit`(rdkit_adapter)**
  - *Doc:* RDKit の記述子メタデータが正しく構成されているか。
- **Function `test_descriptor_metadata_mordred`()**
  - *Doc:* Mordred の記述子メタデータのチェック。
- **Function `test_mordred_adapter`()**
- **Function `test_stub_adapters`()**
  - *Doc:* 各アダプタが名前リストを持ち、is_available()がboolを返すことをテスト。
- **Function `test_psmiles_adapter`()**
  - *Doc:* PSmilesAdapterのテスト。RDKit近似フォールバックが機能することを確認。

### File: `test_chemprop_adapter.py`
- **Class `TestChempropAdapter`**:
  - *Doc:* Chemprop (D-MPNN) アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_init_params(self)`
  - `def test_compute_basic(self)`

### File: `test_chem_base_extra.py`
- **Class `TestDescriptorResult`**:
  - `def test_basic(self)`
  - `def test_with_failures(self)`
  - `def test_empty_smiles_list(self)`
  - `def test_metadata(self)`
- **Class `TestDescriptorMetadata`**:
  - `def test_basic(self)`
  - `def test_binary(self)`
  - `def test_with_description(self)`
- **Class `_DummyAdapter`** (Bases: BaseChemAdapter):
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
  - `def get_descriptors_metadata(self)`
- **Class `_UnavailableAdapter`** (Bases: BaseChemAdapter):
  - `def name(self)`
  - `def description(self)`
  - `def is_available(self)`
  - `def compute(self, smiles_list)`
- **Class `TestBaseChemAdapter`**:
  - `def test_repr_available(self)`
  - `def test_repr_unavailable(self)`
  - `def test_get_descriptor_names(self)`
  - `def test_get_descriptor_names_empty(self)`
  - `def test_require_available_raises(self)`
  - `def test_require_available_passes(self)`
  - `def test_compute(self)`

### File: `test_chem_init.py`
- **Class `TestChemInitSafeImport`**:
  - *Doc:* backend.chem.__init__.py の安全import確認
  - `def test_all_adapters_importable(self)`
  - `def test_all_exported(self)`

### File: `test_chem_init_extra.py`
- **Class `TestMakeUnavailableAdapter`**:
  - `def test_creates_class(self)`
  - `def test_is_not_available(self)`
  - `def test_compute_raises(self)`
  - `def test_accepts_kwargs(self)`
- **Class `TestAdapterRegistry`**:
  - `def test_is_dict(self)`
  - `def test_has_known_keys(self)`
  - `def test_all_values_have_is_available(self)`
  - `def test_registry_size(self)`
- **Class `TestGetAvailableAdapters`**:
  - `def test_returns_dict(self)`
  - `def test_only_available(self)`
  - `def test_subset_of_registry(self)`
- **Class `TestAllExports`**:
  - `def test_base_classes_exported(self)`
  - `def test_registry_exported(self)`
  - `def test_all_adapters_exist(self)`

### File: `test_column_meta_integration.py`
- **Class `TestColumnMetaExtended`**:
  - *Doc:* F-001: ColumnMeta 拡張 - scale_hint, description, fixed の追加フィールド。
  - `def test_default_values(self)`
  - `def test_custom_values(self)`
  - `def test_to_dict(self)`
  - `def test_from_dict_roundtrip(self)`
  - `def test_from_dict_defaults(self)`
  - `def test_from_dict_empty_string_becomes_none(self)`
- **Class `TestColumnMetaEditorUtils`**:
  - *Doc:* F-002: state ↔ ColumnMeta 辞書 変換ユーティリティ。
  - `def test_build_column_meta_dict_from_state(self)`
  - `def test_build_column_meta_dict_empty_state(self)`
  - `def test_extract_monotonic_from_column_meta(self)`
  - `def test_extract_monotonic_merges_existing(self)`
- **Class `TestFeatureSelectorFixed`**:
  - *Doc:* F-003: fixed 変数保護 - 特徴量選択から除外されない。
  - `def sample_data(self)`
  - `def test_fixed_column_preserved_by_lasso(self, sample_data)`
  - `def test_fixed_column_preserved_in_names(self, sample_data)`
  - `def test_none_method_returns_all_anyway(self, sample_data)`
  - `def test_fixed_indices_initialized_before_fit(self, sample_data)`
- **Class `TestAutoMLEngineColumnMeta`**:
  - *Doc:* F-004: AutoMLEngine が column_meta_dict を受け入れて monotonic 適用する。
  - `def test_engine_accepts_column_meta_dict(self)`
  - `def test_engine_accepts_dict_format_column_meta(self)`
  - `def test_engine_run_with_column_meta(self)`

### File: `test_column_selector_extra.py`
- **Class `TestColumnMeta`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestColumnSelectorAll`**:
  - `def test_fit_transform_all(self)`
  - `def test_selected_columns(self)`
  - `def test_get_feature_names_out(self)`
- **Class `TestColumnSelectorInclude`**:
  - `def test_include_columns(self)`
  - `def test_include_range(self)`
  - `def test_include_missing_col_warning(self)`
  - `def test_include_no_columns_no_range(self)`
- **Class `TestColumnSelectorExclude`**:
  - `def test_exclude(self)`
- **Class `TestColumnSelectorErrors`**:
  - `def test_non_dataframe_fit(self)`
  - `def test_non_dataframe_transform(self)`
  - `def test_unknown_mode(self)`
- **Class `TestColumnSelectorMeta`**:
  - `def test_get_column_meta(self)`
  - `def test_get_monotonic_constraints(self)`
  - `def test_get_groups_array(self)`

### File: `test_col_preprocessor_comprehensive.py`
- **Class `TestColPreprocessConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestColPreprocessor`**:
  - `def test_fit_transform_defaults(self, mixed_df)`
  - `def test_standard_scaler(self, mixed_df)`
  - `def test_minmax_scaler(self, mixed_df)`
  - `def test_robust_scaler(self, mixed_df)`
  - `def test_none_scaler(self, mixed_df)`
  - `def test_ordinal_encoder(self, mixed_df)`
  - `def test_median_imputer(self, mixed_df)`
  - `def test_knn_imputer(self, mixed_df)`
  - `def test_constant_imputer(self, mixed_df)`
  - `def test_override_types(self, mixed_df)`
  - `def test_get_feature_names_out(self, mixed_df)`
  - `def test_column_transformer_property(self, mixed_df)`
  - `def test_column_transformer_before_fit_raises(self)`
  - `def test_transform_before_fit_raises(self)`
  - `def test_numpy_input(self)`
- **Function `mixed_df`()**
  - *Doc:* 数値・カテゴリ・バイナリ混合DataFrame

### File: `test_col_preprocessor_extra.py`
- **Class `TestColPreprocessorBasic`**:
  - `def test_default_fit_transform(self)`
  - `def test_feature_names_out(self)`
  - `def test_column_transformer_property(self)`
  - `def test_column_transformer_before_fit(self)`
  - `def test_transform_before_fit(self)`
  - `def test_ndarray_input(self)`
- **Class `TestScalers`**:
  - `def test_scaler_types(self, scaler)`
  - `def test_power_bc(self)`
  - `def test_unknown_scaler(self)`
- **Class `TestImputers`**:
  - `def test_numeric_imputers(self, imputer)`
  - `def test_iterative_imputer(self)`
  - `def test_unknown_imputer(self)`
- **Class `TestEncoders`**:
  - `def test_onehot_encoder(self)`
  - `def test_ordinal_encoder(self)`
  - `def test_target_encoder(self)`
  - `def test_binary_encoder(self)`
  - `def test_woe_encoder(self)`
  - `def test_hashing_encoder(self)`
  - `def test_leaveoneout_encoder(self)`
  - `def test_unknown_encoder(self)`
- **Class `TestBinary`**:
  - `def test_binary_ordinal(self)`
  - `def test_binary_passthrough(self)`
  - `def test_binary_knn_imputer(self)`
  - `def test_binary_constant_imputer(self)`
- **Class `TestOverrideTypes`**:
  - `def test_override_numeric_to_passthrough(self)`
  - `def test_override_cat_to_numeric(self)`
  - `def test_override_invalid_type(self)`
- **Class `TestEdgeCases`**:
  - `def test_empty_columns(self)`
  - `def test_all_missing_column(self)`

### File: `test_config_extra.py`
- **Class `TestConstants`**:
  - `def test_random_state(self)`
  - `def test_project_root_exists(self)`
  - `def test_automl_defaults(self)`
  - `def test_shap_defaults(self)`
  - `def test_type_detector_defaults(self)`
- **Class `TestAppConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
  - `def test_default_config_instance(self)`
  - `def test_extra_field(self)`

### File: `test_constraints_extra.py`
- **Class `TestRangeConstraint`**:
  - `def test_both_bounds(self)`
  - `def test_lo_only(self)`
  - `def test_hi_only(self)`
  - `def test_no_bounds(self)`
  - `def test_is_satisfied(self)`
  - `def test_describe_both(self)`
  - `def test_describe_lo_only(self)`
- **Class `TestSumConstraint`**:
  - `def test_exact_sum(self)`
  - `def test_with_tolerance(self)`
  - `def test_failing_sum(self)`
  - `def test_is_satisfied(self)`
  - `def test_describe_tight(self)`
  - `def test_describe_with_tolerance(self)`
- **Class `TestInequalityConstraint`**:
  - `def test_le(self)`
  - `def test_ge(self)`
  - `def test_lt(self)`
  - `def test_gt(self)`
  - `def test_multi_coeff(self)`
  - `def test_is_satisfied(self)`
  - `def test_is_satisfied_all_operators(self)`
  - `def test_invalid_operator(self)`
  - `def test_invalid_operator_mask(self)`
  - `def test_describe(self)`
  - `def test_describe_coeff_one(self)`
  - `def test_describe_coeff_neg_one(self)`
- **Class `TestAtLeastNConstraint`**:
  - `def test_at_least_one(self)`
  - `def test_at_least_two(self)`
  - `def test_threshold(self)`
  - `def test_is_satisfied(self)`
  - `def test_describe(self)`
  - `def test_describe_with_label(self)`
  - `def test_alias(self)`
- **Class `TestCustomConstraint`**:
  - `def test_simple_expression(self)`
  - `def test_compound_expression(self)`
  - `def test_is_satisfied(self)`
  - `def test_invalid_expression(self)`
  - `def test_invalid_is_satisfied(self)`
  - `def test_describe(self)`
- **Class `TestApplyConstraints`**:
  - `def test_single_constraint(self)`
  - `def test_multiple_constraints(self)`
  - `def test_no_constraints(self)`
  - `def test_all_filtered(self)`
  - `def test_mixed_constraints(self)`

### File: `test_constraint_ui_helpers.py`
- **Class `TestConstants`**:
  - `def test_constraint_types_has_six_entries(self)`
  - `def test_constraint_types_keys(self)`
  - `def test_simple_types_are_range_and_sum(self)`
  - `def test_sum_presets_structure(self)`
  - `def test_ratio_operators_structure(self)`
- **Class `TestUIToBackendConversion`**:
  - `def test_convert_range_constraint(self)`
  - `def test_convert_sum_constraint(self)`
  - `def test_convert_ratio_constraint(self)`
  - `def test_convert_exclusion_constraint(self)`
  - `def test_convert_conditional_constraint(self)`
  - `def test_convert_formula_constraint(self)`
  - `def test_convert_multiple_constraints(self)`
  - `def test_convert_empty_list(self)`
  - `def test_convert_unknown_type_skipped(self)`
  - `def test_convert_invalid_item_skipped(self)`
- **Class `TestDescribeConstraint`**:
  - `def test_range_both_bounds(self)`
  - `def test_range_lower_only(self)`
  - `def test_range_upper_only(self)`
  - `def test_sum_constraint(self)`
  - `def test_ratio_constraint(self)`
  - `def test_conditional_constraint(self)`
- **Class `TestValidateConstraints`**:
  - `def test_no_conflicts_when_no_overlap(self)`
  - `def test_detect_conflict_same_column(self)`
  - `def test_no_conflict_overlapping_ranges(self)`
  - `def test_satisfaction_rate_with_data(self)`
  - `def test_empty_items(self)`
- **Class `TestTemplateIO`**:
  - `def test_save_and_load(self, tmp_path)`
  - `def test_list_templates(self, tmp_path)`
  - `def test_delete_template(self, tmp_path)`
- **Class `TestFriendlyError`**:
  - `def test_known_error_type(self)`
  - `def test_unknown_error_type(self)`
- **Class `TestGetColumnStats`**:
  - `def test_basic_stats(self)`
  - `def test_with_nan(self)`
  - `def test_empty_column(self)`

### File: `test_cosmo_adapter.py`
- **Class `TestCosmoAdapterProperties`**:
  - *Doc:* プロパティの基本テスト。
  - `def test_name(self)`
  - `def test_description(self)`
  - `def test_get_descriptor_names(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_parameterization_default(self)`
  - `def test_parameterization_custom(self)`
- **Class `TestCosmoAdapterAvailability`**:
  - *Doc:* is_available のテスト。
  - `def test_is_available_returns_bool(self)`
- **Class `TestCosmoAdapterComputeNoCosmiFiles`**:
  - *Doc:* cosmi_files が与えられない場合のテスト。
  - `def test_no_cosmi_files_returns_nan(self)`
  - `def test_empty_cosmi_files(self)`
- **Class `TestCosmoAdapterComputeWithCosmiFiles`**:
  - *Doc:* cosmi_files が与えられる場合のテスト。
  - `def _make_mock_module(self, crs_instance)`
  - `def test_missing_file_returns_nan(self)`
  - `def test_fewer_cosmi_files_than_smiles(self)`
  - `def test_successful_calculation_with_dict_result(self)`
  - `def test_calculation_exception_returns_nan(self)`
  - `def test_non_dict_result_returns_nan(self)`
- **Class `TestCosmoDescriptorConstants`**:
  - *Doc:* 記述子定数の整合性。
  - `def test_descriptor_keys(self)`
  - `def test_descriptor_values_are_japanese(self)`

### File: `test_coverage_gaps_batch3.py`
- **Class `TestEdaCoverageGaps`**:
  - `def test_detect_outliers_modified_zscore_zero_mad(self)`
  - `def test_detect_outliers_zscore_zero_sigma(self)`
  - `def test_detect_outliers_nonexistent_col(self)`
  - `def test_analyze_target_auto_int_many_unique(self)`
- **Class `TestDataCleanerCoverageGaps`**:
  - `def test_clip_outliers_non_numeric_col(self)`
  - `def test_preview_missing_impact_subset(self)`
  - `def test_preview_missing_impact_no_check_cols(self)`
  - `def test_preview_outlier_impact_columns_param(self)`
  - `def test_preview_outlier_non_numeric_skip(self)`
- **Class `TestFeatureEngineerCoverageGaps`**:
  - `def test_datetime_extractor_series_input(self)`
  - `def test_datetime_extractor_unknown_component(self)`
  - `def test_datetime_no_cyclic_keys(self)`
- **Class `TestRgfCoverageGaps`**:
  - `def test_rgf_empty_trees(self)`
  - `def test_rgf_linalg_error_fallback(self)`
- **Class `TestConstraintsCoverageGaps`**:
  - `def test_range_constraint_describe_lo_only(self)`
  - `def test_inequality_constraint_le(self)`
  - `def test_inequality_constraint_ge(self)`
  - `def test_inequality_constraint_lt_gt(self)`
- **Class `TestSearchSpaceCoverageGaps`**:
  - `def test_variable_continuous(self)`
  - `def test_variable_discrete(self)`
  - `def test_variable_categorical(self)`
- **Class `TestBayesianOptimizerCoverageGaps`**:
  - `def test_optimizer_import(self)`
- **Class `TestBOVisualizerCoverageGaps`**:
  - `def test_plot_convergence_function(self)`
  - `def test_plot_convergence_minimize(self)`
- **Class `TestBenchmarkCoverageGaps`**:
  - `def test_evaluate_regression(self)`
  - `def test_evaluate_classification(self)`
- **Class `TestCVBiasEvaluatorCoverageGaps`**:
  - `def test_estimate_bbc_cv_bias(self)`
  - `def test_format_bias_report(self)`
- **Class `TestColumnSelectorCoverageGaps`**:
  - `def test_column_selector_with_columns(self)`
  - `def test_column_selector_empty_columns(self)`
- **Class `TestRecommenderCoverageGaps`**:
  - `def test_get_target_names(self)`
  - `def test_get_all_descriptor_categories(self)`
  - `def test_get_target_recommendation(self)`
- **Class `TestChargeConfigCoverageGaps`**:
  - `def test_molecule_charge_config_default(self)`
  - `def test_molecule_charge_config_radical(self)`
- **Class `TestGroupContribCoverageGaps`**:
  - `def test_group_contrib_basic(self)`
  - `def test_group_contrib_invalid_smiles(self)`
- **Class `TestSRICoverageGaps`**:
  - `def test_sri_decomposer(self)`
  - `def test_select_features_by_independence(self)`
- **Class `TestOptionalImportCoverageGaps`**:
  - `def test_require_unavailable(self)`
- **Class `TestMlflowCoverageGaps`**:
  - `def test_mlflow_manager_import(self)`

### File: `test_cv_bias_evaluator.py`
- **Class `TestCVBiasResult`**:
  - `def test_to_dict_basic(self)`
  - `def test_to_dict_with_ci(self)`
- **Class `TestTibshiraniBias`**:
  - `def test_basic_higher_is_better(self)`
  - `def test_basic_lower_is_better(self)`
  - `def test_single_param_zero_bias(self)`
  - `def test_invalid_shape_raises(self)`
  - `def test_param_values_in_details(self)`
  - `def test_bias_per_fold_length(self)`
- **Class `TestBBCCV`**:
  - `def _make_classification_data(self, n, n_configs, seed)`
  - `def test_basic_classification(self)`
  - `def test_regression_mse(self)`
  - `def test_single_config_zero_bias(self)`
  - `def test_empty_predictions_raises(self)`
  - `def test_mismatched_length_raises(self)`
  - `def test_details_contain_config_info(self)`
  - `def test_reproducibility_with_seed(self)`
- **Class `TestFormatBiasReport`**:
  - `def test_tibshirani_format(self)`
  - `def test_bbc_format_with_ci(self)`

### File: `test_cv_bias_evaluator_extra.py`
- **Class `TestCVBiasResult`**:
  - `def test_to_dict(self)`
  - `def test_to_dict_with_ci(self)`
- **Class `TestTibshiraniBias`**:
  - `def test_basic_higher_is_better(self)`
  - `def test_basic_lower_is_better(self)`
  - `def test_with_param_values(self)`
  - `def test_single_param(self)`
  - `def test_invalid_shape(self)`
- **Class `TestBBCCVBias`**:
  - `def test_basic(self)`
  - `def test_single_config(self)`
  - `def test_empty_raises(self)`
  - `def test_length_mismatch_raises(self)`
  - `def test_lower_is_better(self)`
- **Class `TestFormatBiasReport`**:
  - `def test_tibshirani(self)`
  - `def test_bbc_cv_with_ci(self)`

### File: `test_cv_dynamic_extra.py`
- **Function `test_dynamic_cv_lookup`()**
- **Function `test_dynamic_cv_extra_params`()**
- **Function `test_invalid_cv_param_warning`(caplog)**

### File: `test_cv_manager_comprehensive.py`
- **Class `TestWalkForwardSplit`**:
  - `def test_basic_split(self)`
  - `def test_with_gap(self)`
  - `def test_min_train_size(self)`
  - `def test_get_n_splits(self)`
  - `def test_too_few_samples(self)`
  - `def test_with_dataframe(self)`
- **Class `TestCVConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestGetCV`**:
  - `def test_kfold(self)`
  - `def test_stratified_kfold(self)`
  - `def test_loo(self)`
  - `def test_walk_forward(self)`
  - `def test_timeseries(self)`
  - `def test_shuffle_split(self)`
  - `def test_repeated_kfold(self)`
  - `def test_predefined_split_without_test_fold_raises(self)`
  - `def test_predefined_split_with_test_fold(self)`
  - `def test_unknown_cv_raises(self)`
  - `def test_extra_params(self)`
  - `def test_all_registered_keys(self)`
- **Class `TestListCVMethods`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_filter_groups(self)`
  - `def test_filter_no_groups(self)`
- **Class `TestRunCrossValidation`**:
  - `def reg_data(self)`
  - `def test_basic_cv(self, reg_data)`
  - `def test_timeseries_cv(self, reg_data)`
  - `def test_walk_forward_cv(self, reg_data)`

### File: `test_cv_manager_extra.py`
- **Class `TestWalkForwardSplit`**:
  - `def test_basic_split(self)`
  - `def test_with_gap(self)`
  - `def test_min_train_size(self)`
  - `def test_get_n_splits(self)`
  - `def test_too_small_data(self)`
- **Class `TestGetCVClass`**:
  - `def test_kfold(self)`
  - `def test_stratified(self)`
  - `def test_walk_forward(self)`
  - `def test_group_kfold(self)`
  - `def test_unknown_raises(self)`
- **Class `TestCVConfig`**:
  - `def test_defaults(self)`
  - `def test_get_cv_kfold(self)`
  - `def test_get_cv_stratified(self)`
  - `def test_get_cv_timeseries(self)`
  - `def test_get_cv_walk_forward(self)`
  - `def test_get_cv_shuffle_split(self)`
  - `def test_get_cv_predefined_no_fold_raises(self)`
  - `def test_extra_params(self)`
- **Class `TestListCVMethods`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_filter_groups(self)`
  - `def test_filter_no_groups(self)`
- **Class `TestRunCV`**:
  - `def test_basic_regression(self)`
  - `def test_with_timeseries(self)`

### File: `test_cv_manager_new.py`
- **Function `test_cv_aliases`()**
- **Function `test_type_conversion`()**
- **Function `test_bool_conversion`()**

### File: `test_cv_recommender.py`
- **Class `TestNormalRegression`**:
  - `def test_recommends_kfold(self, normal_regression_data)`
  - `def test_result_has_all_fields(self, normal_regression_data)`
- **Class `TestTimeseriesByName`**:
  - `def test_detects_timeseries_column(self, timeseries_data_by_name)`
- **Class `TestTimeseriesMonotonic`**:
  - `def test_detects_monotonic_column(self, timeseries_data_monotonic)`
- **Class `TestGroupedData`**:
  - `def test_recommends_group_cv(self, grouped_data)`
  - `def test_logo_for_few_groups(self, grouped_data)`
- **Class `TestImbalancedClassification`**:
  - `def test_recommends_stratified(self, imbalanced_classification_data)`
- **Class `TestVerySmallData`**:
  - `def test_recommends_loo(self, very_small_data)`
- **Class `TestSmallData`**:
  - `def test_recommends_repeated_kfold(self, small_data)`
- **Class `TestDetectTimeseries`**:
  - `def test_detects_various_patterns(self, col_name)`
  - `def test_no_false_positive(self)`
- **Class `TestDetectImbalance`**:
  - `def test_balanced(self)`
  - `def test_moderate_imbalance(self)`
  - `def test_severe_imbalance(self)`
- **Class `TestAssessSampleSize`**:
  - `def test_very_small(self)`
  - `def test_small(self)`
  - `def test_high_dimensional(self)`
  - `def test_normal(self)`
- **Class `TestNumpyInput`**:
  - `def test_numpy_arrays(self)`
- **Function `normal_regression_data`()**
  - *Doc:* T-CVR01: 標準的な回帰データ (100サンプル, 5特徴量)
- **Function `timeseries_data_by_name`()**
  - *Doc:* T-CVR02: 列名に 'date' を含む時系列データ
- **Function `timeseries_data_monotonic`()**
  - *Doc:* T-CVR03: 等間隔単調増加列（列名に時系列キーワードなし）
- **Function `grouped_data`()**
  - *Doc:* T-CVR04: グループ構造のあるデータ
- **Function `imbalanced_classification_data`()**
  - *Doc:* T-CVR05: クラス不均衡の分類データ
- **Function `very_small_data`()**
  - *Doc:* T-CVR06: 非常に小さなデータ (15サンプル)
- **Function `small_data`()**
  - *Doc:* T-CVR07: 小さなデータ (40サンプル)

### File: `test_cv_walkforward.py`
- **Class `TestWalkForwardSplit`**:
  - `def test_basic_split(self)`
  - `def test_train_before_test(self)`
  - `def test_gap_parameter(self)`
  - `def test_expanding_window(self)`
  - `def test_too_few_samples_raises(self)`
  - `def test_get_n_splits(self)`
  - `def test_dataframe_input(self)`
- **Class `TestCVConfig`**:
  - `def test_default_config(self)`
  - `def test_kfold(self)`
  - `def test_stratified_kfold(self)`
  - `def test_loo(self)`
  - `def test_timeseries(self)`
  - `def test_walk_forward(self)`
  - `def test_shuffle_split(self)`
  - `def test_extra_params(self)`
  - `def test_predefined_split_without_test_fold_raises(self)`
  - `def test_invalid_cv_key(self)`
- **Class `TestListCVMethods`**:
  - `def test_regression_methods(self)`
  - `def test_classification_methods(self)`
  - `def test_group_filter(self)`
  - `def test_no_group_filter(self)`
  - `def test_methods_have_required_fields(self)`
- **Class `TestRunCrossValidation`**:
  - `def regression_data(self)`
  - `def test_kfold_regression(self, regression_data)`
  - `def test_timeseries_regression(self, regression_data)`
  - `def test_walk_forward_regression(self, regression_data)`
  - `def test_return_train_score(self, regression_data)`

### File: `test_data.py`
- **Class `TestTypeDetector`**:
  - *Doc:* T-001: 変数型自動判定のテスト。
  - `def test_detect_returns_detection_result(self, sample_df)`
  - `def test_detect_numeric_normal(self, sample_df)`
  - `def test_detect_numeric_log(self, sample_df)`
  - `def test_detect_binary(self, sample_df)`
  - `def test_detect_category_low(self, sample_df)`
  - `def test_detect_category_high(self, sample_df)`
  - `def test_detect_smiles(self, sample_df)`
  - `def test_detect_constant(self, sample_df)`
  - `def test_smiles_in_smiles_columns(self, sample_df)`
  - `def test_summary_table_shape(self, sample_df)`
  - `def test_get_numeric_columns(self, sample_df)`
  - `def test_datetime_detection(self)`
  - `def test_periodic_col_detection(self)`
  - `def test_null_rate_calculation(self)`
- **Class `TestLoader`**:
  - *Doc:* T-002: データローダーのテスト。
  - `def test_load_csv(self, csv_file)`
  - `def test_load_excel(self, tmp_path)`
  - `def test_load_parquet(self, tmp_path)`
  - `def test_load_json(self, tmp_path)`
  - `def test_load_sqlite(self, tmp_path)`
  - `def test_load_nonexistent_file(self, tmp_path)`
  - `def test_load_unsupported_ext(self, tmp_path)`
  - `def test_load_from_bytes_csv(self)`
  - `def test_save_csv(self, tmp_path)`
  - `def test_get_supported_extensions(self)`
  - `def test_load_csv_encodings(self, tmp_path)`
  - `def test_load_sdf(self, tmp_path)`
  - `def test_load_mol(self, tmp_path)`
  - `def test_load_from_bytes_exhaustive(self)`
  - `def test_load_csv_encodings(self, tmp_path)`
  - `def test_save_all_formats_detailed(self, tmp_path)`
- **Class `TestPreprocessor`**:
  - *Doc:* T-003: 前処理パイプライン構築のテスト。
  - `def test_build_returns_column_transformer(self, sample_df)`
  - `def test_fit_transform_shape(self, sample_df)`
  - `def test_build_excludes_target_col(self, sample_df)`
  - `def test_log_transformer(self)`
  - `def test_sincos_transformer(self)`
  - `def test_transformer_property_raises_before_build(self)`
  - `def test_build_full_pipeline(self, sample_df)`
  - `def test_categorical_encoders_all(self, sample_df)`
  - `def test_categorical_encoders_fallback(self, sample_df)`
  - `def test_custom_transformers_exhaustive(self)`
  - `def test_all_imputers_and_scalers(self, sample_df)`
  - `def test_pipeline_with_missing_indicator(self, sample_df)`
- **Function `sample_df`()**
  - *Doc:* 型判定テスト用のサンプルDataFrame。
- **Function `csv_file`(tmp_path)**
  - *Doc:* テスト用CSVファイルを作成する。

### File: `test_data_cleaner.py`
- **Class `TestDropColumns`**:
  - *Doc:* T-CLEAN-001: 列除外のテスト。
  - `def test_basic_drop(self, sample_df)`
  - `def test_drop_single_column(self, sample_df)`
  - `def test_drop_nonexistent_column_raises(self, sample_df)`
  - `def test_drop_empty_list_raises(self, sample_df)`
  - `def test_drop_partial_nonexistent(self, sample_df)`
  - `def test_action_details(self, sample_df)`
  - `def test_action_timestamp_format(self, sample_df)`
- **Class `TestDropRowsWithMissing`**:
  - *Doc:* T-CLEAN-002: 欠損行削除のテスト。
  - `def test_basic_drop_any_missing(self, sample_df)`
  - `def test_threshold_50_percent(self, all_missing_df)`
  - `def test_threshold_100_removes_none(self, sample_df)`
  - `def test_invalid_threshold_raises(self, sample_df)`
  - `def test_subset_columns(self, sample_df)`
  - `def test_subset_with_missing(self, all_missing_df)`
  - `def test_empty_subset(self, sample_df)`
  - `def test_reset_index(self, sample_df)`
- **Class `TestRemoveConstantColumns`**:
  - *Doc:* T-CLEAN-003: 定数列除去のテスト。
  - `def test_basic_remove(self, sample_df)`
  - `def test_no_constant_columns(self)`
  - `def test_all_constant(self)`
  - `def test_all_nan_is_constant(self, all_missing_df)`
  - `def test_details_has_list(self, sample_df)`
- **Class `TestClipOutliers`**:
  - *Doc:* T-CLEAN-004: 外れ値クリッピングのテスト。
  - `def test_basic_clip(self, outlier_df)`
  - `def test_no_outliers(self)`
  - `def test_iqr_zero_skipped(self, outlier_df)`
  - `def test_column_subset(self, outlier_df)`
  - `def test_invalid_multiplier_raises(self, outlier_df)`
  - `def test_larger_multiplier_clips_less(self, outlier_df)`
  - `def test_values_within_bounds(self, outlier_df)`
- **Class `TestRemoveDuplicates`**:
  - *Doc:* T-CLEAN-005: 重複行除去のテスト。
  - `def test_basic_remove(self, dup_df)`
  - `def test_no_duplicates(self, sample_df)`
  - `def test_keep_last(self, dup_df)`
  - `def test_subset(self, dup_df)`
  - `def test_reset_index(self, dup_df)`
- **Class `TestPreviewFunctions`**:
  - *Doc:* T-CLEAN-006: プレビュー関数のテスト。
  - `def test_preview_missing_basic(self, sample_df)`
  - `def test_preview_missing_with_threshold(self, all_missing_df)`
  - `def test_preview_missing_matches_actual(self, sample_df)`
  - `def test_preview_outlier_basic(self, outlier_df)`
  - `def test_preview_outlier_matches_actual(self, outlier_df)`
- **Class `TestGetCleaningSummary`**:
  - *Doc:* T-CLEAN-007: クリーニングサマリーのテスト。
  - `def test_basic_summary(self, sample_df)`
  - `def test_all_missing_summary(self, all_missing_df)`
  - `def test_dup_summary(self, dup_df)`
- **Class `TestEdgeCases`**:
  - *Doc:* エッジケースのテスト。
  - `def test_empty_df_constant_cols(self, empty_df)`
  - `def test_empty_df_duplicates(self, empty_df)`
  - `def test_single_row_df(self)`
  - `def test_cleaning_action_properties(self)`
- **Function `sample_df`()**
  - *Doc:* 標準テスト用DataFrame。
- **Function `outlier_df`()**
  - *Doc:* 外れ値を含むDataFrame。
- **Function `dup_df`()**
  - *Doc:* 重複行を含むDataFrame。
- **Function `empty_df`()**
  - *Doc:* 空のDataFrame。
- **Function `all_missing_df`()**
  - *Doc:* 全欠損列を含むDataFrame。

### File: `test_data_cleaner_extra.py`
- **Class `TestCleaningAction`**:
  - `def test_rows_removed(self)`
  - `def test_cols_removed(self)`
- **Class `TestDropColumns`**:
  - `def test_basic(self)`
  - `def test_multiple(self)`
  - `def test_empty_raises(self)`
  - `def test_nonexistent_raises(self)`
- **Class `TestDropRowsWithMissing`**:
  - `def test_basic(self)`
  - `def test_threshold_zero(self)`
  - `def test_invalid_threshold(self)`
  - `def test_with_subset(self)`
  - `def test_no_check_cols(self)`
- **Class `TestRemoveConstantColumns`**:
  - `def test_with_constants(self)`
  - `def test_no_constants(self)`
- **Class `TestClipOutliers`**:
  - `def test_basic(self)`
  - `def test_with_columns(self)`
  - `def test_invalid_multiplier(self)`
- **Class `TestRemoveDuplicates`**:
  - `def test_basic(self)`
  - `def test_keep_last(self)`
  - `def test_subset(self)`
- **Class `TestPreviewFunctions`**:
  - `def test_preview_missing(self)`
  - `def test_preview_missing_zero(self)`
  - `def test_preview_outlier(self)`
  - `def test_get_cleaning_summary(self)`

### File: `test_descriptastorus_adapter.py`
- **Class `TestDescriptaStorusAdapter`**:
  - *Doc:* DescriptaStorus アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_init_descriptor_type(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_compute_basic(self)`

### File: `test_dim_reduction.py`
- **Class `TestDimReducerPCA`**:
  - *Doc:* T-DR-001: DimReducer PCAモードのテスト。
  - `def test_output_shape_2d(self, sample_df)`
  - `def test_output_shape_3d(self, sample_df)`
  - `def test_explained_variance_ratio(self, sample_df)`
  - `def test_cumulative_variance_le1(self, sample_df)`
  - `def test_transform_on_new_data(self, sample_df)`
  - `def test_scale_false(self, sample_df)`
  - `def test_feature_names_out(self, sample_df)`
  - `def test_numpy_input(self)`
  - `def test_incremental_pca_trigger(self, sample_df)`
  - `def test_reconstruction_error(self, sample_df)`
  - `def test_loadings(self, sample_df)`
- **Class `TestDimReducerTSNE`**:
  - *Doc:* T-DR-002: DimReducer t-SNEモードのテスト。
  - `def test_output_shape(self, sample_df)`
  - `def test_transform_returns_cached_embedding(self, sample_df)`
  - `def test_explained_variance_none_for_tsne(self, sample_df)`
- **Class `TestDimReducerUMAP`**:
  - *Doc:* T-DR-003: DimReducer UMAPモードのテスト。
  - `def test_umap_raises_without_lib(self)`
- **Function `sample_df`()**

### File: `test_dim_reduction_extra.py`
- **Class `TestDimReductionConfig`**:
  - `def test_defaults(self)`
- **Class `TestDimReducerPCA`**:
  - `def test_pca_fit_transform(self)`
  - `def test_pca_explained_variance(self)`
  - `def test_pca_loadings(self)`
  - `def test_pca_reconstruction_error(self)`
  - `def test_pca_feature_names(self)`
  - `def test_get_feature_names_out(self)`
  - `def test_pca_no_scale(self)`
  - `def test_pca_whiten(self)`
  - `def test_pca_ndarray_input(self)`
- **Class `TestDimReducerTSNE`**:
  - `def test_tsne_fit_transform(self)`
- **Class `TestDimReducerUnknown`**:
  - `def test_unknown_method(self)`
- **Class `TestConvenienceFunctions`**:
  - `def test_run_pca(self)`
  - `def test_run_pca_with_target(self)`
  - `def test_run_tsne(self)`

### File: `test_dim_reduction_final.py`
- **Function `test_pca_reconstruction_error`()**
- **Function `test_dim_reduction_extra_params`()**

### File: `test_domainml_analysis.py`
- **Function `test_metrics_satisfaction_score`()**
- **Function `test_constrained_cv`()**
- **Function `test_uncertainty_estimator`()**

### File: `test_domainml_engine.py`
- **Function `test_lazy_evaluator`()**
- **Function `test_monotonicity_engine`()**

### File: `test_domainml_kernel_opt.py`
- **Function `test_kernel_monotonicity_increasing`()**
- **Function `test_kernel_monotonicity_decreasing`()**
- **Function `test_kernel_invalid_inputs`()**

### File: `test_domainml_laplacian.py`
- **Function `test_sparse_laplacian_builder`()**
- **Function `test_manifold_validity_estimator`()**

### File: `test_e2e_smiles_to_ml.py`
- **Class `TestSmilesDescriptorTransform`**:
  - *Doc:* F-01: SmilesDescriptorTransformer の変換動作を検証する。
  - `def test_rdkit_descriptors_computed(self)`
  - `def test_no_all_nan_columns(self)`
  - `def test_row_count_preserved(self)`
- **Class `TestE2ERegressionSmilesToML`**:
  - *Doc:* F-02: サンプル回帰データのE2E完走を検証する。
  - `def regression_result(self)`
  - `def test_e2e_completes_without_error(self, regression_result)`
  - `def test_best_model_selected(self, regression_result)`
  - `def test_best_score_is_finite(self, regression_result)`
  - `def test_best_pipeline_predict(self, regression_result)`
  - `def test_model_scores_populated(self, regression_result)`
  - `def test_elapsed_is_positive(self, regression_result)`
- **Class `TestE2EClassificationSmilesToML`**:
  - *Doc:* F-03: サンプル分類データのE2E完走を検証する。
  - `def classification_result(self)`
  - `def test_e2e_classification_completes(self, classification_result)`
  - `def test_classification_task_detected(self, classification_result)`
  - `def test_classification_predict(self, classification_result)`
- **Class `TestAutoTaskDetection`**:
  - *Doc:* F-04: task='auto' で回帰/分類が自動判定されること。
  - `def test_auto_detects_regression(self)`
  - `def test_auto_detects_classification(self)`
- **Class `TestAutoMLWithSmilesColDirect`**:
  - *Doc:* F-05: smiles_col をAutoMLEngineに直接渡した場合の動作検証。
  - `def test_smiles_col_in_engine_run(self)`
- **Class `TestDescriptorNaNHandling`**:
  - *Doc:* F-06: 記述子にNaNが含まれていても前処理で補完されること。
  - `def test_nan_imputed_in_pipeline(self)`

### File: `test_eda.py`
- **Class `TestComputeColumnStats`**:
  - *Doc:* T-EDA-001: 列統計計算のテスト。
  - `def test_returns_list_length(self, mixed_df)`
  - `def test_numeric_col_has_mean(self, mixed_df)`
  - `def test_categorical_col_has_top_values(self, mixed_df)`
  - `def test_null_rate_correctness(self, mixed_df)`
  - `def test_dtype_field(self, mixed_df)`
- **Class `TestSummarizeDataframe`**:
  - *Doc:* T-EDA-002: DataFrame全体サマリーのテスト。
  - `def test_required_keys(self, mixed_df)`
  - `def test_shape_correctness(self, mixed_df)`
  - `def test_n_numeric_count(self, mixed_df)`
  - `def test_null_rate_range(self, mixed_df)`
  - `def test_memory_mb_positive(self, mixed_df)`
- **Class `TestComputeCorrelation`**:
  - *Doc:* T-EDA-003: 相関計算のテスト。
  - `def test_pearson_matrix_shape(self, mixed_df)`
  - `def test_diagonal_is_one(self, mixed_df)`
  - `def test_spearman_method(self, mixed_df)`
  - `def test_target_col_filter(self, mixed_df)`
  - `def test_few_columns_raises(self)`
- **Class `TestDetectOutliers`**:
  - *Doc:* T-EDA-004: 外れ値検出のテスト。
  - `def test_iqr_detects_extreme_values(self, outlier_df)`
  - `def test_zscore_method(self, outlier_df)`
  - `def test_modified_zscore_method(self, outlier_df)`
  - `def test_outlier_rate_range(self, outlier_df)`
  - `def test_unknown_method_raises(self, outlier_df)`
  - `def test_col_filter(self, outlier_df)`
- **Class `TestComputeDistribution`**:
  - *Doc:* T-EDA-005: 分布計算のテスト。
  - `def test_numeric_returns_histogram(self, mixed_df)`
  - `def test_categorical_returns_categories(self, mixed_df)`
  - `def test_sum_of_counts_equals_non_null(self, mixed_df)`
- **Class `TestAnalyzeTarget`**:
  - *Doc:* T-EDA-006: 目的変数分析のテスト。
  - `def test_regression_keys(self, mixed_df)`
  - `def test_classification_keys(self, mixed_df)`
  - `def test_auto_detection_regression(self, mixed_df)`
  - `def test_missing_col_raises(self, mixed_df)`
  - `def test_null_rate(self, mixed_df)`
- **Function `mixed_df`()**
  - *Doc:* 数値・カテゴリ・欠損を含む混合DataFrame。
- **Function `outlier_df`()**
  - *Doc:* 外れ値を含むDataFrame。

### File: `test_eda_extra.py`
- **Class `TestComputeColumnStats`**:
  - `def test_numeric_stats(self)`
  - `def test_categorical_stats(self)`
  - `def test_null_stats(self)`
  - `def test_single_value(self)`
- **Class `TestSummarizeDataframe`**:
  - `def test_basic(self)`
  - `def test_with_duplicates(self)`
- **Class `TestComputeCorrelation`**:
  - `def test_pearson(self)`
  - `def test_spearman(self)`
  - `def test_with_target(self)`
  - `def test_too_few_columns(self)`
- **Class `TestDetectOutliers`**:
  - `def test_iqr(self)`
  - `def test_zscore(self)`
  - `def test_modified_zscore(self)`
  - `def test_specific_cols(self)`
  - `def test_unknown_method(self)`
- **Class `TestComputeDistribution`**:
  - `def test_numeric(self)`
  - `def test_categorical(self)`
- **Class `TestAnalyzeTarget`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_auto_float(self)`
  - `def test_auto_categorical(self)`
  - `def test_missing_col(self)`

### File: `test_factory.py`
- **Class `TestGetModel`**:
  - `def test_regression_models(self, key)`
  - `def test_classification_models(self, key)`
  - `def test_override_params(self)`
  - `def test_unknown_key_raises(self)`
  - `def test_unknown_task_raises(self)`
- **Class `TestListModels`**:
  - `def test_regression_list_not_empty(self)`
  - `def test_classification_list_not_empty(self)`
  - `def test_filter_by_tag(self)`
  - `def test_filter_ensemble(self)`
  - `def test_model_entry_has_required_fields(self)`
  - `def test_available_only_filter(self)`
- **Class `TestGetDefaultAutoMLModels`**:
  - `def test_regression_defaults(self)`
  - `def test_classification_defaults(self)`
  - `def test_all_defaults_are_instantiable(self)`
- **Class `TestGetModelRegistry`**:
  - `def test_regression_registry(self)`
  - `def test_classification_registry(self)`
  - `def test_registry_entries_have_name(self)`
- **Class `TestOptionalModels`**:
  - `def test_xgb_if_available(self)`
  - `def test_lgbm_if_available(self)`
  - `def test_catboost_if_available(self)`
  - `def test_lineartree_if_available(self)`
  - `def test_rgf_if_available(self)`

### File: `test_factory_comprehensive.py`
- **Class `TestGetModel`**:
  - `def test_ridge(self)`
  - `def test_rf(self)`
  - `def test_rf_classifier(self)`
  - `def test_logistic(self)`
  - `def test_svr(self)`
  - `def test_knn(self)`
  - `def test_dt(self)`
  - `def test_gbm(self)`
  - `def test_hgbm(self)`
  - `def test_mlp(self)`
  - `def test_gp(self)`
  - `def test_pls(self)`
  - `def test_override_params(self)`
  - `def test_unknown_key(self)`
  - `def test_unknown_task(self)`
- **Class `TestListModels`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_with_tags(self)`
  - `def test_available_only(self)`
- **Class `TestDefaultAutoml`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
- **Class `TestGetModelRegistry`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
- **Class `TestGetRegistry`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_unknown(self)`

### File: `test_factory_extra.py`
- **Class `TestGetRegistry`**:
  - `def test_regression(self)`
  - `def test_classification(self)`
  - `def test_unknown_raises(self)`
- **Class `TestGetModel`**:
  - `def test_linear_regression(self)`
  - `def test_ridge(self)`
  - `def test_lasso(self)`
  - `def test_rf(self)`
  - `def test_svr(self)`
  - `def test_knn(self)`
  - `def test_classifier(self)`
  - `def test_unknown_key(self)`
  - `def test_regression_default(self)`
  - `def test_classification_default(self)`
- **Class `TestListModels`**:
  - `def test_regression_list(self)`
  - `def test_classification_list(self)`
  - `def test_available_only(self)`
  - `def test_filter_by_tags(self)`
  - `def test_filter_ensemble(self)`
- **Class `TestDefaultAutoMLModels`**:
  - `def test_regression_defaults(self)`
  - `def test_classification_defaults(self)`

### File: `test_feature_engineer.py`
- **Class `TestInteractionTransformer`**:
  - *Doc:* T-FE-001: 交互作用項Transformerのテスト。
  - `def test_output_shape_2col(self, small_numeric_df)`
  - `def test_output_shape_3col(self, small_numeric_df)`
  - `def test_feature_names_out(self, small_numeric_df)`
  - `def test_numpy_input(self)`
  - `def test_fit_transform_equivalence(self, small_numeric_df)`
- **Class `TestDatetimeFeatureExtractor`**:
  - *Doc:* T-FE-002: 時系列特徴量抽出のテスト。
  - `def test_output_cols_include_cyclic(self, datetime_series)`
  - `def test_no_cyclic(self, datetime_series)`
  - `def test_feature_names_out(self, datetime_series)`
  - `def test_hour_range(self, datetime_series)`
  - `def test_is_weekend_flag(self)`
- **Class `TestLagRollingTransformer`**:
  - *Doc:* T-FE-003: ラグ・ローリングTransformerのテスト。
  - `def test_output_shape(self, small_numeric_df)`
  - `def test_no_future_leak_in_lag(self, small_numeric_df)`
  - `def test_feature_names_count(self, small_numeric_df)`
  - `def test_numpy_input(self)`
  - `def test_fill_nan_with_zero(self)`
- **Class `TestFeatureEngineeringConfig`**:
  - *Doc:* T-FE-004: 設定クラスとパイプライン構築のテスト。
  - `def test_default_config(self)`
  - `def test_build_pipeline_empty_when_no_interactions(self)`
  - `def test_build_pipeline_has_interaction_step(self)`
- **Class `TestGroupAggTransformer`**:
  - *Doc:* T-FE-005: グループ集約Transformerのテスト。
  - `def test_output_shape(self, group_df)`
  - `def test_feature_names(self, group_df)`
  - `def test_group_mean_value(self, group_df)`
  - `def test_missing_group_col_raises(self)`
  - `def test_fillna_zero_for_new_group(self, group_df)`
- **Function `group_df`()**
  - *Doc:* グループ集約テスト用DataFrameフィクスチャ。
- **Function `small_numeric_df`()**
- **Function `datetime_series`()**

### File: `test_feature_engineer_ext.py`
- **Class `TestInteractionTransformer`**:
  - `def test_basic_transform(self)`
  - `def test_with_bias(self)`
  - `def test_numpy_input(self)`
  - `def test_feature_names_out(self)`
- **Class `TestGroupAggTransformer`**:
  - `def test_basic_aggregation(self)`
  - `def test_multiple_agg_funcs(self)`
  - `def test_missing_group_col_raises(self)`
  - `def test_feature_names_out(self)`
- **Class `TestDatetimeFeatureExtractor`**:
  - `def test_basic_extraction(self)`
  - `def test_cyclic_features(self)`
  - `def test_is_weekend(self)`
  - `def test_numpy_input(self)`
  - `def test_feature_names_out(self)`
- **Class `TestLagRollingTransformer`**:
  - `def test_basic_lags(self)`
  - `def test_rolling_mean(self)`
  - `def test_multiple_columns(self)`
  - `def test_numpy_input(self)`
  - `def test_feature_names_out(self)`
- **Class `TestBuildFeatureEngineeringPipeline`**:
  - `def test_empty_config(self)`
  - `def test_interactions_only(self)`
  - `def test_group_agg(self)`
  - `def test_datetime_features(self)`
  - `def test_lag_rolling(self)`
  - `def test_all_combined(self)`
  - `def test_multiple_group_agg(self)`
  - `def test_empty_group_agg_skipped(self)`

### File: `test_feature_engineer_extra.py`
- **Class `TestInteractionTransformer`**:
  - `def test_fit_transform_ndarray(self)`
  - `def test_fit_transform_dataframe(self)`
  - `def test_feature_names_out(self)`
  - `def test_not_fitted(self)`
- **Class `TestGroupAggTransformer`**:
  - `def test_basic(self)`
  - `def test_feature_names(self)`
  - `def test_missing_group_col(self)`
- **Class `TestDatetimeFeatureExtractor`**:
  - `def test_basic(self)`
  - `def test_no_cyclic(self)`
  - `def test_specific_components(self)`
  - `def test_ndarray_input(self)`
- **Class `TestLagRollingTransformer`**:
  - `def test_basic_ndarray(self)`
  - `def test_basic_dataframe(self)`
  - `def test_feature_names(self)`
- **Class `TestFeatureEngineeringConfig`**:
  - `def test_defaults(self)`
  - `def test_build_empty_pipeline(self)`
  - `def test_build_with_interactions(self)`
  - `def test_build_with_datetime(self)`
  - `def test_build_with_lag_rolling(self)`
  - `def test_build_with_group_agg(self)`

### File: `test_feature_generator_extra.py`
- **Class `TestFeatureGenConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestFeatureGeneratorNone`**:
  - `def test_passthrough_ndarray(self)`
  - `def test_passthrough_dataframe(self)`
  - `def test_is_passthrough(self)`
  - `def test_feature_names_out_none(self)`
  - `def test_n_output_features_none(self)`
- **Class `TestFeatureGeneratorPoly`**:
  - `def test_polynomial(self)`
  - `def test_polynomial_with_bias(self)`
  - `def test_is_not_passthrough(self)`
  - `def test_n_output_features_poly(self)`
- **Class `TestFeatureGeneratorInteraction`**:
  - `def test_interaction_only(self)`
  - `def test_feature_names_out_interaction(self)`

### File: `test_feature_integrity.py`
- **Class `TestFeatureIntegrity`**:
  - *Doc:* フロントエンドに必須アダプターが含まれることを検証する保護テスト群。
  - `def test_frontend_dir_exists(self)`
  - `def test_required_adapter_referenced(self, adapter)`
  - `def test_smiles_feature_ui_present(self)`

### File: `test_feature_selector_comprehensive.py`
- **Class `TestFeatureSelectorConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestFeatureSelectorNone`**:
  - `def test_passthrough(self, reg_data)`
  - `def test_feature_names_passthrough(self)`
- **Class `TestFeatureSelectorLasso`**:
  - `def test_lasso(self, reg_data)`
- **Class `TestFeatureSelectorRidge`**:
  - `def test_ridge(self, reg_data)`
- **Class `TestFeatureSelectorRF`**:
  - `def test_rfr(self, reg_data)`
  - `def test_rfc(self, cls_data)`
- **Class `TestFeatureSelectorKBest`**:
  - `def test_select_kbest(self, reg_data)`
  - `def test_select_kbest_classification(self, cls_data)`
- **Class `TestFeatureSelectorPercentile`**:
  - `def test_select_percentile(self, reg_data)`
  - `def test_feature_names_out(self, reg_data)`
- **Class `TestFeatureSelectorUnknown`**:
  - `def test_unknown_method_fallback(self, reg_data)`
- **Function `reg_data`()**
- **Function `cls_data`()**

### File: `test_feature_selector_extra.py`
- **Class `TestFeatureSelectorNone`**:
  - `def test_none_passthrough(self)`
  - `def test_none_feature_names(self)`
- **Class `TestLasso`**:
  - `def test_lasso_regression(self)`
- **Class `TestRidge`**:
  - `def test_ridge_regression(self)`
- **Class `TestRandomForest`**:
  - `def test_rfr(self)`
  - `def test_rfc(self)`
- **Class `TestXGBoost`**:
  - `def test_xgb_regression(self)`
  - `def test_xgb_classification(self)`
- **Class `TestSelectPercentile`**:
  - `def test_percentile_regression(self)`
  - `def test_percentile_classification(self)`
  - `def test_percentile_mutual_info(self)`
- **Class `TestSelectKBest`**:
  - `def test_kbest_regression(self)`
  - `def test_kbest_unknown_score_func(self)`
- **Class `TestSelectFromModel`**:
  - `def test_sfm_with_estimator_key(self)`
  - `def test_sfm_without_estimator_key(self)`
  - `def test_sfm_invalid_estimator_key(self)`
- **Class `TestUnknownMethod`**:
  - `def test_unknown_method_uses_rf(self)`
- **Class `TestNdarrayInput`**:
  - `def test_ndarray_input(self)`
- **Class `TestMaxFeatures`**:
  - `def test_max_features_limits(self)`
- **Class `TestReliefF`**:
  - `def test_relieff_or_fallback(self)`
- **Class `TestBoruta`**:
  - `def test_boruta_or_fallback(self)`
- **Class `TestGenetic`**:
  - `def test_genetic_or_fallback(self)`
- **Class `TestGroupLasso`**:
  - `def test_group_lasso_or_fallback(self)`

### File: `test_final_push.py`
- **Function `test_loader_remaining_branches`(tmp_path)**
- **Function `test_cv_manager_remaining_branches`()**
- **Function `test_tuner_remaining_branches`()**
- **Function `test_factory_list_models_detailed`()**
- **Function `test_optional_import_safe_exhaustive`()**

### File: `test_forward_inverse_combinations.py`
- **Class `TestForwardModelCVCombinations`**:
  - *Doc:* 順解析: モデル × CV手法のすべての組み合わせで実行可能か
  - `def reg_df(self)`
  - `def test_model_cv_combination(self, reg_df, model_key, cv_key)`
  - `def test_classification_models(self, model_key)`
- **Class `TestForwardModelPreprocessCombinations`**:
  - *Doc:* 順解析: モデル × 前処理設定の組み合わせ
  - `def df_with_outliers(self)`
  - `def test_model_with_outlier_data(self, df_with_outliers, model_key)`
- **Class `TestInverseMethodTargetCombinations`**:
  - *Doc:* 逆解析: 5手法 × 3目標モードの全組み合わせ
  - `def _get_method_params(self, method)`
  - `def test_method_target_combination_2d(self, method, target_mode)`
  - `def test_dirichlet_target_modes(self, target_mode)`
- **Class `TestInverseMethodConstraintCombinations`**:
  - *Doc:* 逆解析: 手法 × 制約タイプの組み合わせ
  - `def test_random_with_fixed_vars(self)`
  - `def test_grid_with_inactive_vars(self)`
  - `def test_bayesian_narrow_bounds(self)`
  - `def test_method_with_3_vars_1_fixed(self, method)`
  - `def test_dirichlet_with_tight_bounds(self)`
- **Class `TestForwardToInverseE2E`**:
  - *Doc:* 順解析の学習結果を逆解析の予測関数として使うE2Eテスト
  - `def trained_model(self)`
  - `def test_e2e_various_methods(self, trained_model, method)`
  - `def test_e2e_multiple_models_then_inverse(self)`
- **Class `TestEdgeCasesDetailed`**:
  - `def test_all_but_one_fixed(self)`
  - `def test_missing_constraints_defaults(self)`
  - `def test_progress_callback_called(self)`
- **Class `TestHighDimensionalInverse`**:
  - `def _make_high_dim_fn(self, n_dims)`
  - `def test_10d_optimization(self, method)`
- **Class `TestBayesianAcquisitionCombinations`**:
  - `def test_acq_objective_combination(self, acq_func, objective)`
  - `def test_kernel_types(self)`
  - `def test_batch_strategies(self)`
  - `def test_ptr_target_range(self)`
  - `def test_multi_objective_parego(self)`
  - `def test_multi_objective_max_direction(self)`
- **Class `TestGAParameterCombinations`**:
  - `def test_ga_params(self, pop_size, mutation_rate, crossover_rate)`
  - `def test_sbx_crossover_direct(self)`
  - `def test_sbx_crossover_identical_parents(self)`
- **Class `TestDirichletParameterCombinations`**:
  - `def test_concentration_total_sum(self, concentration, total_sum)`
- **Class `TestConstraintCombinations`**:
  - `def _make_df(self, n)`
  - `def test_range_plus_sum(self)`
  - `def test_range_plus_inequality_plus_atleast(self)`
  - `def test_all_constraint_types(self)`
  - `def test_constraint_is_satisfied_single_row(self)`
  - `def test_constraint_describe(self)`
  - `def test_custom_constraint_error_returns_false(self)`
  - `def test_inequality_operators(self)`
  - `def test_atleast_alias(self)`
- **Class `TestScoreCalculation`**:
  - `def test_maximize_score(self)`
  - `def test_minimize_score(self)`
  - `def test_range_score_center(self)`
  - `def test_range_score_gaussian_shape(self)`
- **Class `TestFullPipelineE2E`**:
  - `def test_full_pipeline_with_constraints(self)`
  - `def test_build_full_df(self)`
  - `def test_build_full_df_missing_col_fills_zero(self)`
  - `def test_build_result_df(self)`
- **Class `TestCVManagerCombinations`**:
  - *Doc:* CVManagerの全手法組み合わせテスト
  - `def reg_data(self)`
  - `def test_get_cv_all_types(self, cv_key)`
  - `def test_run_cv_with_ridge(self, reg_data, cv_key)`
  - `def test_list_cv_methods_regression(self)`
  - `def test_list_cv_methods_classification(self)`
  - `def test_list_cv_methods_groups_filter(self)`
  - `def test_walk_forward_split(self)`
  - `def test_walk_forward_with_gap(self)`
  - `def test_walk_forward_too_small_raises(self)`
  - `def test_walk_forward_get_n_splits(self)`
  - `def test_cv_config_with_unknown_key_uses_get_cv_class(self)`
  - `def test_cv_invalid_key_raises(self)`
- **Class `TestAutoMLAdvanced`**:
  - `def test_auto_task_detection_regression(self)`
  - `def test_auto_task_detection_classification(self)`
  - `def test_model_details_populated(self)`
  - `def test_elapsed_time(self)`
  - `def test_progress_callback(self)`
  - `def test_oof_predictions(self)`
  - `def test_processed_X(self)`
  - `def test_too_few_rows_raises(self)`
  - `def test_missing_target_col_raises(self)`
  - `def test_run_multi_feature_sets(self)`
- **Class `TestBOAdditionalCoverage`**:
  - `def test_get_gp_info_not_fitted(self)`
  - `def test_get_gp_info_multi_objective(self)`
  - `def test_predict_multi_objective(self)`
  - `def test_maximin_select(self)`
  - `def test_maximin_select_n_exceeds(self)`
  - `def test_unknown_acquisition_raises(self)`
  - `def test_ptr_missing_bounds_raises(self)`
  - `def test_not_fitted_suggest_raises(self)`
  - `def test_not_fitted_predict_raises(self)`
  - `def test_suggest_with_dataframe(self)`
- **Class `TestFactoryComprehensive`**:
  - `def simple_data(self)`
  - `def test_fit_predict_regression(self, simple_data, key)`
  - `def cls_data(self)`
  - `def test_fit_predict_classification(self, cls_data, key)`
  - `def test_ransac_regression(self, simple_data)`
  - `def test_list_models_count(self)`
  - `def test_default_automl_models(self)`

### File: `test_group_contrib.py`
- **Class `TestGroupContribAdapter`**:
  - `def _adapter(self)`
  - `def test_is_available(self)`
  - `def test_name_and_description(self)`
  - `def test_descriptor_names(self)`
  - `def test_descriptor_metadata(self)`
  - `def test_compute_methane(self)`
  - `def test_compute_ethanol(self)`
  - `def test_compute_acetic_acid(self)`
  - `def test_compute_benzene(self)`
  - `def test_compute_aspirin(self)`
  - `def test_boiling_point_positive(self)`
  - `def test_critical_temp_above_boiling(self)`
  - `def test_critical_volume_positive(self)`
  - `def test_larger_molecule_higher_vc(self)`
  - `def test_invalid_smiles(self)`
  - `def test_empty_list(self)`
  - `def test_single_atom(self)`
  - `def test_batch_computation(self)`
- **Class `TestJobackInternals`**:
  - `def test_count_groups_ethanol(self)`
  - `def test_count_groups_acetone(self)`
  - `def test_estimate_properties_returns_dict(self)`
  - `def test_cp298_positive(self)`
  - `def test_halogen_present(self)`

### File: `test_integration.py`
- **Class `TestDataPipeline`**:
  - *Doc:* T-INT-001: データロード→型判定の統合テスト。
  - `def test_load_csv_from_bytes(self, reg_df)`
  - `def test_type_detection(self, reg_df)`
  - `def test_column_transformer_builds(self, reg_df)`
  - `def test_preprocess_produces_numpy(self, reg_df)`
  - `def test_no_data_leakage(self, reg_df)`
- **Class `TestEDAPipeline`**:
  - *Doc:* T-INT-002: EDA→次元削減の統合テスト。
  - `def test_eda_summary(self, reg_df)`
  - `def test_column_stats_returns_list(self, reg_df)`
  - `def test_pca_on_numeric(self, reg_df)`
  - `def test_outlier_detection(self, reg_df)`
- **Class `TestModelTrainingPipeline`**:
  - *Doc:* T-INT-003: 前処理→モデル学習→評価の統合テスト。
  - `def test_regression_end_to_end(self, reg_df)`
  - `def test_classification_end_to_end(self, cls_df)`
  - `def test_train_test_column_consistency(self, reg_df)`
- **Function `reg_df`()**
  - *Doc:* 回帰タスク用サンプルDataFrame。
- **Function `cls_df`()**
  - *Doc:* 分類タスク用サンプルDataFrame。

### File: `test_interpret.py`
- **Class `TestSRIDecomposer`**:
  - *Doc:* T-008: SRI分解のテスト。
  - `def test_decompose_returns_sriresult(self, dummy_shap_result)`
  - `def test_synergy_matrix_shape(self, dummy_shap_result)`
  - `def test_redundancy_matrix_shape(self, dummy_shap_result)`
  - `def test_independence_vec_shape(self, dummy_shap_result)`
  - `def test_synergy_matrix_symmetric(self, dummy_shap_result)`
  - `def test_redundancy_matrix_symmetric(self, dummy_shap_result)`
  - `def test_independence_vec_nonnegative(self, dummy_shap_result)`
  - `def test_total_sri_tuple(self, dummy_shap_result)`
  - `def test_feature_names_preserved(self, dummy_shap_result)`
  - `def test_summary_df_has_required_columns(self, dummy_shap_result)`
  - `def test_pairwise_df_has_required_columns(self, dummy_shap_result)`
  - `def test_pairwise_df_row_count(self, dummy_shap_result)`
  - `def test_multiclass_decompose_uses_class0(self, multiclass_shap_result)`
  - `def test_center_option_no_error(self, dummy_shap_result)`
- **Class `TestSelectFeaturesByIndependence`**:
  - *Doc:* T-009: Independence基準の特徴量選択のテスト。
  - `def test_select_top_n(self, dummy_shap_result)`
  - `def test_select_with_threshold(self, dummy_shap_result)`
  - `def test_select_all_if_no_args(self, dummy_shap_result)`
  - `def test_select_returns_independence_sorted(self, dummy_shap_result)`
  - `def test_sri_plot_methods(self, dummy_shap_result)`
- **Class `TestShapExplainer`**:
  - *Doc:* T-010: SHAP解釈のテスト。
  - `def test_explain_tree_model(self)`
  - `def test_explain_linear_model(self)`
  - `def test_get_feature_importance_df(self, dummy_shap_result)`
  - `def test_plot_methods_no_error(self, dummy_shap_result, tmp_path)`
  - `def test_explain_multiclass_plots(self, multiclass_shap_result)`
- **Function `dummy_shap_result`()**
  - *Doc:* テスト用のダミーShapResultを生成する。
- **Function `multiclass_shap_result`()**
  - *Doc:* マルチクラス用のダミーShapResultを生成する。

### File: `test_inverse_optimizer.py`
- **Class `TestRandomOptimizer`**:
  - `def test_basic_random(self)`
  - `def test_minimize_mode(self)`
- **Class `TestGridOptimizer`**:
  - `def test_basic_grid(self)`
- **Class `TestBayesianOptimizer`**:
  - `def test_basic_bayesian(self)`
- **Class `TestGAOptimizer`**:
  - `def test_basic_ga(self)`
- **Class `TestDirichletOptimizer`**:
  - *Doc:* ディリクレ分布α更新型の組成系最適化テスト。
  - `def test_basic_dirichlet_composition(self)`
  - `def test_dirichlet_total_sum_100(self)`
  - `def test_dirichlet_alpha_convergence(self)`
- **Class `TestEdgeCases`**:
  - `def test_no_search_cols_raises(self)`
  - `def test_range_mode(self)`
  - `def test_unknown_method_raises(self)`

### File: `test_leakage_detector.py`
- **Class `TestHatMatrix`**:
  - *Doc:* F-ld-001: ハット行列の数学的正確性
  - `def test_symmetry(self)`
  - `def test_idempotent(self)`
  - `def test_trace_equals_p(self)`
  - `def test_diagonal_between_0_and_1(self)`
  - `def test_singular_case(self)`
- **Class `TestRBFGram`**:
  - *Doc:* F-ld-002: RBFグラム行列の正確性
  - `def test_symmetry(self)`
  - `def test_diagonal_is_one(self)`
  - `def test_values_between_0_and_1(self)`
  - `def test_custom_gamma(self)`
  - `def test_identical_samples_have_high_similarity(self)`
- **Class `TestRFProximity`**:
  - *Doc:* F-ld-003: RF Proximity の正確性
  - `def test_symmetry(self)`
  - `def test_diagonal_is_one(self)`
  - `def test_values_between_0_and_1(self)`
  - `def test_unsupervised_mode(self)`
  - `def test_grouped_data_high_within_group(self)`
- **Class `TestGroupConsistencyScore`**:
  - *Doc:* F-ld-004: グループ一貫性スコアの正確性
  - `def test_perfect_groups(self)`
  - `def test_no_groups(self)`
  - `def test_score_range(self)`
- **Class `TestEstimateGroups`**:
  - *Doc:* F-ld-005: グループ推定のテスト
  - `def test_clear_groups_detected(self)`
  - `def test_returns_correct_shape(self)`
  - `def test_too_few_samples(self)`
- **Class `TestDetectLeakage`**:
  - *Doc:* F-ld-006: detect_leakage メインAPIのテスト
  - `def test_returns_leakage_report(self)`
  - `def test_low_risk_for_independent_data(self)`
  - `def test_high_risk_for_grouped_data(self)`
  - `def test_group_labels_assigned_for_high_risk(self)`
  - `def test_cv_recommendation(self)`
  - `def test_accepts_dataframe(self)`
  - `def test_auto_method_selection(self)`
  - `def test_rf_method(self)`
  - `def test_suspicious_pairs_sorted(self)`
  - `def test_handles_nan(self)`
  - `def test_details_contain_metadata(self)`

### File: `test_leakage_detector_extra.py`
- **Class `TestComputeHatMatrix`**:
  - `def test_basic(self)`
  - `def test_rank_deficient(self)`
- **Class `TestComputeRBFGram`**:
  - `def test_basic(self)`
  - `def test_custom_gamma(self)`
- **Class `TestComputeRFProximity`**:
  - `def test_with_y(self)`
  - `def test_without_y(self)`
- **Class `TestFindSuspiciousPairs`**:
  - `def test_basic(self)`
  - `def test_no_pairs(self)`
- **Class `TestGroupConsistency`**:
  - `def test_basic(self)`
  - `def test_small_n(self)`
- **Class `TestEstimateGroups`**:
  - `def test_basic(self)`
  - `def test_tiny_data(self)`
- **Class `TestDetectLeakage`**:
  - `def test_hat_method(self)`
  - `def test_rbf_method(self)`
  - `def test_rf_method(self)`
  - `def test_auto_method(self)`
  - `def test_with_dataframe(self)`
  - `def test_with_nan(self)`
  - `def test_unknown_method(self)`
- **Class `TestCheckFeatureLeakage`**:
  - `def test_no_leakage(self)`
  - `def test_high_correlation_leakage(self)`
  - `def test_name_similarity(self)`
  - `def test_missing_target(self)`
  - `def test_classification_separation(self)`

### File: `test_linear_tree_extra.py`
- **Class `TestUtilities`**:
  - `def test_to_numpy_array(self)`
  - `def test_to_numpy_dataframe(self)`
  - `def test_fit_linear_success(self)`
  - `def test_fit_linear_empty(self)`
  - `def test_predict_linear(self)`
  - `def test_mse(self)`
  - `def test_mse_empty(self)`
  - `def test_gini(self)`
  - `def test_gini_pure(self)`
- **Class `TestNode`**:
  - `def test_is_leaf(self)`
  - `def test_is_not_leaf(self)`
- **Class `TestLinearTreeRegressor`**:
  - `def test_fit_predict(self)`
  - `def test_with_dataframe(self)`
  - `def test_custom_estimator(self)`
  - `def test_n_leaves(self)`
  - `def test_max_features_sqrt(self)`
- **Class `TestLinearTreeClassifier`**:
  - `def test_fit_predict(self)`
  - `def test_predict_proba(self)`
  - `def test_multiclass(self)`
- **Class `TestLinearForestRegressor`**:
  - `def test_fit_predict(self)`
  - `def test_no_bootstrap(self)`
- **Class `TestLinearForestClassifier`**:
  - `def test_fit_predict(self)`
  - `def test_predict_proba(self)`
- **Class `TestLinearBoostRegressor`**:
  - `def test_fit_predict(self)`
  - `def test_subsample(self)`
- **Class `TestLinearBoostClassifier`**:
  - `def test_fit_predict_binary(self)`
  - `def test_predict_proba_binary(self)`

### File: `test_linear_tree_rgf_monotonic.py`
- **Class `TestLinearTreeRegressor`**:
  - `def setup_method(self)`
  - `def test_fit_predict_shape(self)`
  - `def test_fit_does_not_mutate_base_estimator(self)`
  - `def test_n_leaves_positive(self)`
  - `def test_residual_improvement(self)`
  - `def test_different_base_estimators(self)`
  - `def test_clone_compatibility(self)`
  - `def test_min_samples_enforcement(self)`
- **Class `TestLinearTreeClassifier`**:
  - `def setup_method(self)`
  - `def test_predict_classes_in_range(self)`
  - `def test_predict_proba_sums_to_one(self)`
  - `def test_multiclass(self)`
  - `def test_fit_not_mutate_base(self)`
- **Class `TestLinearForestRegressor`**:
  - `def setup_method(self)`
  - `def test_basic(self)`
  - `def test_averaging(self)`
- **Class `TestLinearBoostRegressor`**:
  - `def setup_method(self)`
  - `def test_basic(self)`
  - `def test_mse_improves_with_rounds(self)`
- **Class `TestLinearBoostClassifier`**:
  - `def setup_method(self)`
  - `def test_binary(self)`
  - `def test_multiclass(self)`
- **Class `TestRGFRegressor`**:
  - `def setup_method(self)`
  - `def test_basic(self)`
  - `def test_leaf_indicator_dim(self)`
  - `def test_weights_shape(self)`
  - `def test_regularization_effect(self)`
  - `def test_mse_improves(self)`
- **Class `TestRGFClassifier`**:
  - `def setup_method(self)`
  - `def test_binary(self)`
  - `def test_multiclass(self)`
  - `def test_proba_range(self)`
- **Class `TestMonotonicKernelWrapper`**:
  - `def test_svr_increasing(self)`
  - `def test_no_constraint_passthrough(self)`
  - `def test_predict_shape(self)`
  - `def test_violation_stored(self)`
  - `def test_kernel_ridge_decreasing(self)`
  - `def test_is_soft_monotonic_candidate(self)`
- **Class `TestPipelineIntegration`**:
  - `def test_apply_monotonic_constraints_native(self)`
  - `def test_apply_monotonic_constraints_soft(self)`
  - `def test_apply_no_constraint_returns_same(self)`

### File: `test_loader_extra.py`
- **Class `TestLoadFile`**:
  - `def test_csv(self, tmp_path)`
  - `def test_tsv(self, tmp_path)`
  - `def test_json(self, tmp_path)`
  - `def test_parquet(self, tmp_path)`
  - `def test_file_not_found(self)`
  - `def test_unsupported_extension(self, tmp_path)`
- **Class `TestLoadFromBytes`**:
  - `def test_csv_bytes(self)`
  - `def test_json_bytes(self)`
  - `def test_unsupported_bytes(self)`
- **Class `TestSaveDataframe`**:
  - `def test_save_csv(self, tmp_path)`
  - `def test_save_tsv(self, tmp_path)`
  - `def test_save_json(self, tmp_path)`
  - `def test_save_parquet(self, tmp_path)`
  - `def test_save_unsupported(self, tmp_path)`
- **Class `TestSupportedExtensions`**:
  - `def test_returns_list(self)`
  - `def test_matches_internal(self)`

### File: `test_mlops.py`
- **Function `mock_mlflow`()**
- **Function `mlflow_manager`()**
  - *Doc:* MLflowManager のインスタンスを返すフィクスチャ
- **Function `test_mlflow_manager_init`(mock_mlflow, mlflow_manager)**
- **Function `test_start_end_run`(mock_mlflow, mlflow_manager)**
- **Function `test_log_params_metrics`(mock_mlflow, mlflow_manager)**
- **Function `test_mlrun_context`(mock_mlflow, mlflow_manager)**
- **Function `test_run_context_fail`(mock_mlflow, mlflow_manager)**
  - *Doc:* コンテキスト内部で例外が起きたとき fail_run が呼ばれること。(T-008-05)
- **Function `test_save_and_load_model`(mock_mlflow, mlflow_manager, tmp_path)**
  - *Doc:* モデルの保存と読み込みができること。(T-008-06)
- **Function `test_get_experiment_runs`(mock_mlflow, mlflow_manager)**
  - *Doc:* 実験結果の検索が呼ばれること。(T-008-07)
- **Function `test_log_figure`(mock_mlflow, mlflow_manager)**
  - *Doc:* 図の記録が呼ばれること。(T-008-08)
- **Function `test_best_run_none`(mock_mlflow, mlflow_manager)**
  - *Doc:* ランが無い場合にNoneを返すこと。(T-008-09)
- **Function `test_save_model_registry_fail`(mock_mlflow, mlflow_manager, tmp_path)**
  - *Doc:* Model Registry 登録失敗時に警告が出るが続行されること。(T-008-11)
- **Function `test_log_artifact_no_mlflow`(tmp_path)**
  - *Doc:* MLflow 未インストール時に log_artifact が何もしないこと。(T-008-12)
- **Function `test_run_context_with_tags`(mock_mlflow, mlflow_manager)**
  - *Doc:* タグ付きランのコンテキスト。(T-008-10)

### File: `test_mlops_extended.py`
- **Class `TestMLflowManagerGracefulDegradation`**:
  - *Doc:* mlflowが未インストールの場合のGraceful Degradation
  - `def test_init_without_mlflow(self)`
  - `def test_start_run_returns_none(self)`
  - `def test_end_run_noop(self)`
  - `def test_log_params_noop(self)`
  - `def test_log_metrics_noop(self)`
  - `def test_log_artifact_noop(self)`
  - `def test_log_figure_noop(self)`
  - `def test_get_experiment_runs_returns_empty(self)`
  - `def test_get_best_run_returns_none(self)`
  - `def test_fail_run_noop(self)`
- **Class `TestModelSaveLoad`**:
  - `def test_save_and_load_model(self)`
  - `def test_load_model_not_found(self)`
  - `def test_save_creates_directory(self)`
- **Class `TestMLRunContext`**:
  - `def test_context_without_mlflow(self)`
  - `def test_context_exception_calls_fail_run(self)`
- **Class `TestCosmoAdapterMock`**:
  - `def test_name_and_description(self)`
  - `def test_descriptor_names(self)`
  - `def test_descriptor_metadata(self)`
  - `def test_compute_without_cosmi_files_returns_nan(self)`

### File: `test_models.py`
- **Class `TestModelFactory`**:
  - *Doc:* T-004: モデルファクトリーのテスト。
  - `def test_get_regression_model(self, regression_data)`
  - `def test_get_classification_model(self, classification_data)`
  - `def test_unknown_model_key_raises(self)`
  - `def test_unknown_task_raises(self)`
  - `def test_list_models_regression(self)`
  - `def test_list_models_classification(self)`
  - `def test_list_models_with_tags(self)`
  - `def test_get_default_automl_models(self)`
  - `def test_override_params(self, regression_data)`
  - `def test_sklearn_linear_models(self, regression_data)`
- **Class `TestCVManager`**:
  - *Doc:* T-005: クロスバリデーションのテスト。
  - `def test_get_kfold(self)`
  - `def test_get_stratified_kfold(self)`
  - `def test_get_timeseries_split(self)`
  - `def test_get_loo(self)`
  - `def test_unknown_cv_key_raises(self)`
  - `def test_walk_forward_split(self)`
  - `def test_list_cv_methods(self)`
  - `def test_run_cross_validation(self, regression_data)`
  - `def test_walk_forward_no_leak(self)`
- **Class `TestTuner`**:
  - *Doc:* T-006: ハイパーパラメータ最適化のテスト。
  - `def test_grid_search(self, regression_data)`
  - `def test_random_search(self, regression_data)`
  - `def test_unknown_method_raises(self, regression_data)`
  - `def test_best_score_is_float(self, regression_data)`
  - `def test_optuna_search(self, regression_data)`
  - `def test_factory_errors(self)`
  - `def test_halving_search(self, regression_data)`
  - `def test_tuner_bayes_fallback_v2(self, regression_data)`
- **Class `TestAutoMLEngine`**:
  - *Doc:* T-007: AutoMLエンジンのテスト。
  - `def test_automl_engine_regression(self)`
  - `def test_automl_engine_classification(self)`
  - `def test_run_returns_model_scores(self, regression_df)`
  - `def test_run_best_pipeline_can_predict(self, regression_df)`
  - `def test_automl_too_few_records(self)`
  - `def test_automl_invalid_target_col(self)`
- **Function `regression_data`()**
- **Function `classification_data`()**
- **Function `regression_df`()**
  - *Doc:* AutoMLテスト用DataFrameフィクスチャ。
- **Function `generate_numeric_data`(n_samples, n_features, n_classes, task)**
  - *Doc:* テスト用のシンプルな数値データセットを生成する関数。

### File: `test_mol2vec_adapter.py`
- **Class `TestMol2VecAdapter`**:
  - *Doc:* Mol2Vec アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_init_params(self)`
  - `def test_compute_returns_correct_shape(self)`

### File: `test_molfeat_adapter.py`
- **Class `TestMolfeatAdapter`**:
  - *Doc:* Molfeat アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_init_calculator_type(self)`
  - `def test_compute_ecfp(self)`

### File: `test_monotonic_kernel.py`
- **Class `TestComputeMonotonicViolation`**:
  - `def test_perfect_increasing(self)`
  - `def test_perfect_decreasing(self)`
  - `def test_violation_increasing(self)`
  - `def test_violation_decreasing(self)`
  - `def test_constant_no_violation(self)`
- **Class `TestMonotonicKernelWrapper`**:
  - `def test_fit_predict_no_constraints(self, monotonic_regression_data)`
  - `def test_fit_predict_with_constraints(self, monotonic_regression_data)`
  - `def test_default_base_estimator(self, monotonic_regression_data)`
  - `def test_kernel_ridge(self, monotonic_regression_data)`
  - `def test_score_method(self, monotonic_regression_data)`
  - `def test_get_params(self)`
  - `def test_set_params(self)`
  - `def test_clone_compatibility(self)`
  - `def test_n_features_in(self, monotonic_regression_data)`
  - `def test_violation_recorded(self, monotonic_regression_data)`
- **Class `TestMonotonicKernelClassifierWrapper`**:
  - `def test_fit_predict_no_constraints(self, classification_data)`
  - `def test_predict_proba(self, classification_data)`
  - `def test_with_constraints(self, classification_data)`
  - `def test_default_base_estimator(self, classification_data)`
- **Class `TestWrapFactory`**:
  - `def test_svr_is_candidate(self)`
  - `def test_kernel_ridge_is_candidate(self)`
  - `def test_svc_is_candidate(self)`
  - `def test_ridge_not_candidate(self)`
  - `def test_wrap_regressor(self)`
  - `def test_wrap_classifier(self)`
  - `def test_no_constraint_returns_original(self)`
  - `def test_wrap_empty_constraints_returns_original(self)`
- **Class `TestBuildGridX`**:
  - `def test_grid_shape(self)`
  - `def test_grid_varies_only_target_feature(self)`
- **Function `monotonic_regression_data`()**
  - *Doc:* 正相関が明確な回帰データ（x→y増加）
- **Function `classification_data`()**

### File: `test_monotonic_kernel_comprehensive.py`
- **Class `TestComputeMonotonicViolation`**:
  - `def test_monotonic_increasing(self)`
  - `def test_monotonic_decreasing(self)`
  - `def test_violation_increasing(self)`
  - `def test_violation_decreasing(self)`
- **Class `TestBuildGridX`**:
  - `def test_basic(self)`
  - `def test_zero_std(self)`
- **Class `TestFitWithWeight`**:
  - `def test_with_weight(self)`
  - `def test_without_weight(self)`
- **Class `TestIsSoftMonotonicCandidate`**:
  - `def test_svr(self)`
  - `def test_kernel_ridge(self)`
  - `def test_ridge_no(self)`
  - `def test_svc(self)`
- **Class `TestMonotonicKernelWrapper`**:
  - `def reg_data(self)`
  - `def test_no_constraints(self, reg_data)`
  - `def test_with_constraints(self, reg_data)`
  - `def test_default_estimator(self, reg_data)`
  - `def test_score(self, reg_data)`
  - `def test_get_set_params(self)`
  - `def test_clone(self)`
- **Class `TestMonotonicKernelClassifierWrapper`**:
  - `def cls_data(self)`
  - `def test_no_constraints(self, cls_data)`
  - `def test_with_constraints(self, cls_data)`
  - `def test_default_estimator(self, cls_data)`
  - `def test_score(self, cls_data)`
- **Class `TestWrapWithSoftMonotonic`**:
  - `def test_no_constraints_returns_original(self)`
  - `def test_regressor(self)`
  - `def test_classifier(self)`

### File: `test_monotonic_kernel_extra.py`
- **Class `TestToNumpy`**:
  - `def test_ndarray(self)`
  - `def test_dataframe(self)`
  - `def test_list(self)`
- **Class `TestComputeViolation`**:
  - `def test_monotonic_increasing_no_violation(self)`
  - `def test_monotonic_decreasing_no_violation(self)`
  - `def test_increasing_with_violation(self)`
  - `def test_decreasing_with_violation(self)`
- **Class `TestBuildGridX`**:
  - `def test_basic(self)`
  - `def test_zero_sigma(self)`
- **Class `TestBuildMonotonicAugmentedData`**:
  - `def test_no_violation(self)`
  - `def test_with_violation(self)`
- **Class `TestMonotonicKernelWrapper`**:
  - `def test_no_constraints(self)`
  - `def test_with_constraints(self)`
  - `def test_default_estimator(self)`
  - `def test_score(self)`
  - `def test_get_params(self)`
  - `def test_set_params(self)`
  - `def test_with_dataframe(self)`
- **Class `TestMonotonicKernelClassifierWrapper`**:
  - `def test_no_constraints(self)`
  - `def test_with_constraints(self)`
  - `def test_predict_proba(self)`
  - `def test_default_estimator(self)`
  - `def test_score(self)`
  - `def test_get_set_params(self)`
- **Class `TestFitWithWeight`**:
  - `def test_without_weight(self)`
  - `def test_with_weight(self)`
- **Class `TestSoftMonotonicCandidate`**:
  - `def test_svr(self)`
  - `def test_svc(self)`
  - `def test_kernel_ridge(self)`
  - `def test_ridge(self)`
  - `def test_logistic_regression(self)`
- **Class `TestWrapWithSoftMonotonic`**:
  - `def test_no_constraints(self)`
  - `def test_wrap_regressor(self)`
  - `def test_wrap_classifier(self)`
  - `def test_custom_params(self)`

### File: `test_monotonic_wrapper.py`
- **Class `TestMonotonicConstraintRegressorAPI`**:
  - *Doc:* F-001: sklearn API 完全互換テスト。
  - `def test_instantiation_default(self)`
  - `def test_get_params_deep(self)`
  - `def test_set_params(self)`
  - `def test_set_params_nested(self)`
  - `def test_clone_compatibility(self)`
  - `def test_feature_names_stored(self, regression_data)`
  - `def test_predict_shape(self, regression_data)`
- **Class `TestMonotonicConstraintRegressorMonotonicity`**:
  - *Doc:* F-002: 各モデルでペナルティ拡張により単調性が概ね改善されること。
  - `def test_monotonic_increase_x1(self, regression_data, base_cls, base_kwargs)`
  - `def test_monotonic_decrease_x3(self, regression_data, base_cls, base_kwargs)`
  - `def test_no_constraint_passthrough(self, regression_data)`
- **Class `TestMonotonicConstraintClassifierAPI`**:
  - *Doc:* F-003: 分類版 sklearn API テスト。
  - `def test_fit_predict(self, classification_data)`
  - `def test_predict_proba_shape(self, classification_data)`
  - `def test_classes_stored(self, classification_data)`
  - `def test_clone(self)`
- **Class `TestModelMonotonicStrategy`**:
  - *Doc:* F-005: モデル種別判定。
  - `def test_rf_is_native(self)`
  - `def test_svr_is_penalty(self)`
  - `def test_ridge_is_penalty(self)`
  - `def test_histgbm_is_native(self)`
  - `def test_xgboost_is_native(self)`
- **Class `TestWrapMonotonic`**:
  - *Doc:* F-006: ファクトリー関数テスト。
  - `def test_wraps_regressor(self)`
  - `def test_wraps_classifier(self)`
  - `def test_no_constraint_returns_original(self)`
  - `def test_auto_detect_wraps(self)`
- **Class `TestApplyMonotonicConstraintsRouting`**:
  - *Doc:* F-007: pipeline_builder の2段階ルーティング。
  - `def _make_col_meta(self, names, mono_vals, strength)`
  - `def test_rfr_native_monotonic(self)`
  - `def test_rfc_native_monotonic(self)`
  - `def test_svr_gets_penalty_wrapper(self)`
  - `def test_no_constraint_returns_original(self)`
  - `def test_histgbm_native(self)`
  - `def test_xgb_native(self)`
  - `def test_strength_propagated_penalty_model(self)`
  - `def test_native_auto_detect_fallback(self)`
- **Class `TestColumnMetaIntegrationMonotonic`**:
  - *Doc:* F-008: ColumnMeta → ラッパー → E2E テスト。
  - `def test_column_meta_with_constraint_strength(self, regression_data)`
  - `def test_column_meta_auto_detect(self, regression_data)`
  - `def test_end_to_end_fit_predict(self, regression_data)`
- **Class `TestAutoDetectMonotonic`**:
  - *Doc:* F-009: Spearman相関による方向自動検出。
  - `def test_resolve_positive_correlation(self, regression_data)`
  - `def test_resolve_negative_correlation(self, regression_data)`
  - `def test_auto_detect_in_wrapper(self, regression_data)`
- **Class `TestConstraintStrength`**:
  - *Doc:* F-010: weak / strong プリセットの動作テスト。
  - `def test_weak_uses_low_penalty(self)`
  - `def test_strong_uses_high_penalty(self)`
  - `def test_none_uses_individual_params(self)`
  - `def test_strong_fit_works(self, regression_data)`
- **Class `TestExtrapolationMonotonicity`**:
  - *Doc:* F-011: デフォルト sigma_factor=3.0 での外挿範囲単調性。
  - `def test_sigma_factor_default_is_3(self)`
  - `def test_extrapolation_fit(self, regression_data)`
- **Function `regression_data`()**
  - *Doc:* 単調性が明確なデータセット (80サンプル, 3特徴量)。
- **Function `classification_data`()**
  - *Doc:* 単調性を持つ2値分類データセット。

### File: `test_mordred_adapter.py`
- **Class `TestMordredAdapterProperties`**:
  - `def test_name(self)`
  - `def test_description(self)`
  - `def test_is_available(self)`
  - `def test_selected_descriptors_not_empty(self)`
  - `def test_default_selected_only_true(self)`
- **Class `TestDescriptorNames`**:
  - `def test_get_descriptor_names_selected(self)`
  - `def test_get_descriptor_names_all(self)`
  - `def test_get_descriptors_metadata(self)`
- **Class `TestMordredCompute`**:
  - `def _skip_if_unavailable(self)`
  - `def test_basic_compute(self)`
  - `def test_invalid_smiles(self)`
  - `def test_empty_smiles(self)`
  - `def test_all_invalid(self)`
  - `def test_selected_only_limits_columns(self)`
  - `def test_full_descriptors(self)`
  - `def test_no_inf_values(self)`
  - `def test_metadata_in_result(self)`
  - `def test_smiles_list_preserved(self)`

### File: `test_nicegui_components.py`
- **Class `TestAutoDetectColumns`**:
  - *Doc:* _auto_detect_columns のロジックテスト
  - `def _get_func(self)`
  - `def test_detects_smiles_column_by_name(self)`
  - `def test_detects_smiles_column_case_insensitive(self)`
  - `def test_target_col_is_last_column(self)`
  - `def test_classification_task_detection(self)`
  - `def test_regression_task_detection(self)`
  - `def test_no_smiles_column(self)`
  - `def test_none_df(self)`
- **Class `TestToggleModel`**:
  - *Doc:* _toggle_model のステート操作テスト
  - `def _get_func(self)`
  - `def test_add_model(self)`
  - `def test_remove_model(self)`
  - `def test_add_duplicate_model(self)`
  - `def test_remove_nonexistent_model(self)`
  - `def test_empty_list(self)`
- **Class `TestOnTargetChange`**:
  - *Doc:* _on_target_change のタスク自動判定テスト
  - `def _get_func(self)`
  - `def test_regression_detection(self)`
  - `def test_classification_detection(self)`
  - `def test_column_not_in_df(self)`
- **Class `TestSampleSmiles`**:
  - *Doc:* SAMPLE_SMILES 定数のテスト
  - `def test_sample_smiles_not_empty(self)`
  - `def test_sample_smiles_are_strings(self)`
  - `def test_sample_smiles_unique(self)`
- **Class `TestRunEngineSync`**:
  - *Doc:* _run_engine_sync のテスト
  - `def test_function_exists(self)`
  - `def test_analysis_running_flag_exists(self)`
  - `def test_run_engine_sync_calls_engine(self, MockEngine)`
  - `def test_progress_callback_sends_to_queue(self, MockEngine)`
- **Class `TestAllEngines`**:
  - *Doc:* _ALL_ENGINES 定数のテスト
  - `def test_engines_list_not_empty(self)`
  - `def test_engine_tuple_structure(self)`
  - `def test_engine_count(self)`
- **Class `TestEngineInfo`**:
  - *Doc:* _ENGINE_INFO 定数のテスト (manual_url フィールド含む)
  - `def _get_engine_info(self)`
  - `def test_engine_info_not_empty(self)`
  - `def test_engine_info_count(self)`
  - `def test_all_engines_have_manual_url(self)`
  - `def test_all_manual_urls_are_non_empty(self)`
  - `def test_all_manual_urls_start_with_http(self)`
  - `def test_required_fields_present(self)`
  - `def test_engine_cls_names_unique(self)`
  - `def test_rdkit_adapter_url(self)`
  - `def test_xtb_adapter_url(self)`
- **Class `TestAutoParamsUiImport`**:
  - *Doc:* auto_params_ui のインポートテスト
  - `def test_render_param_editor_importable(self)`
  - `def test_render_model_param_editor_importable(self)`
  - `def test_render_adapter_param_editor_importable(self)`

### File: `test_optim_comprehensive.py`
- **Class `TestRangeConstraint`**:
  - `def test_in_range(self)`
  - `def test_below(self)`
  - `def test_above(self)`
  - `def test_mask(self)`
  - `def test_describe(self)`
  - `def test_lo_only(self)`
  - `def test_hi_only(self)`
- **Class `TestSumConstraint`**:
  - `def test_exact(self)`
  - `def test_not_exact(self)`
  - `def test_mask(self)`
  - `def test_describe(self)`
- **Class `TestInequalityConstraint`**:
  - `def test_le(self)`
  - `def test_ge(self)`
  - `def test_lt(self)`
  - `def test_gt(self)`
  - `def test_mask(self)`
  - `def test_describe(self)`
- **Class `TestAtLeastNConstraint`**:
  - `def test_satisfied(self)`
  - `def test_not_satisfied(self)`
  - `def test_mask(self)`
- **Class `TestCustomConstraint`**:
  - `def test_eval(self)`
  - `def test_mask(self)`
  - `def test_describe(self)`
- **Class `TestApplyConstraints`**:
  - `def test_multiple(self)`
  - `def test_no_constraints(self)`
- **Class `TestVariable`**:
  - `def test_continuous(self)`
  - `def test_discrete(self)`
  - `def test_categorical(self)`
  - `def test_validation_no_lo(self)`
  - `def test_validation_lo_gt_hi(self)`
  - `def test_validation_discrete_no_step(self)`
  - `def test_validation_categorical_no_categories(self)`
- **Class `TestSearchSpace`**:
  - `def simple_space(self)`
  - `def test_dim(self, simple_space)`
  - `def test_names(self, simple_space)`
  - `def test_grid(self, simple_space)`
  - `def test_random(self, simple_space)`
  - `def test_lhs(self, simple_space)`
  - `def test_auto(self, simple_space)`
  - `def test_grid_downsample(self, simple_space)`
  - `def test_random_lhs(self, simple_space)`
  - `def test_invalid_method(self, simple_space)`
  - `def test_no_variables(self)`
  - `def test_estimate_grid_size(self, simple_space)`
  - `def test_auto_recommend(self, simple_space)`
  - `def test_from_dataframe(self)`
  - `def test_add_variable(self)`
  - `def test_with_categorical(self)`

### File: `test_optional_import_extra.py`
- **Class `TestSafeImport`**:
  - `def test_import_existing_module(self)`
  - `def test_import_with_alias(self)`
  - `def test_import_nonexistent(self)`
  - `def test_import_nonexistent_with_alias(self)`
  - `def test_import_submodule(self)`
- **Class `TestIsAvailable`**:
  - `def test_available(self)`
  - `def test_not_available(self)`
  - `def test_unknown_key(self)`
- **Class `TestRequire`**:
  - `def test_require_available(self)`
  - `def test_require_unavailable(self)`
  - `def test_require_with_feature(self)`
- **Class `TestGetReport`**:
  - `def test_returns_dict(self)`
  - `def test_includes_probed(self)`
- **Class `TestProbeAll`**:
  - `def test_probe_returns_dict(self)`
  - `def test_probe_includes_known_libs(self)`

### File: `test_padel_adapter.py`
- **Class `TestPaDELAdapter`**:
  - *Doc:* PaDEL-Descriptor アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_init_params(self)`
  - `def test_compute_basic(self)`

### File: `test_param_schema.py`
- **Function `test_sklearn_estimators`()**
  - *Doc:* TEST 1: sklearn estimators — 全モデルイントロスペクション
- **Function `test_chem_adapters`()**
  - *Doc:* TEST 2: ChemAdapters
- **Function `test_feature_selection_preprocessing`()**
  - *Doc:* TEST 3: Feature Selection & Preprocessing
- **Function `test_apply_params`()**
  - *Doc:* TEST 4: apply_params — 型変換とバリデーション
- **Function `test_to_dict_json`()**
  - *Doc:* TEST 5: ParamSpec.to_dict (JSON serialization)
- **Function `test_new_model_auto_detection`()**
  - *Doc:* TEST 6: 新モデル追加（GPR）→ 自動検出テスト（UIコード変更不要の証明）
- **Function `test_factory_integration`()**
  - *Doc:* TEST 7: factory.pyレジストリとの統合

### File: `test_param_schema_comprehensive.py`
- **Class `TestParamSpec`**:
  - `def test_defaults(self)`
  - `def test_to_dict(self)`
- **Class `TestIntrospectParams`**:
  - `def test_random_forest(self)`
  - `def test_ridge(self)`
  - `def test_svr(self)`
  - `def test_logistic(self)`
  - `def test_advanced_group(self)`
  - `def test_instance(self)`
  - `def test_skip_params(self)`
  - `def test_extra_descriptions(self)`
- **Class `TestApplyParams`**:
  - `def test_basic(self)`
  - `def test_skip_default(self)`
  - `def test_missing_key(self)`
- **Class `TestConvertValue`**:
  - `def test_bool_true(self)`
  - `def test_bool_false(self)`
  - `def test_int(self)`
  - `def test_float(self)`
  - `def test_str(self)`
  - `def test_select(self)`
  - `def test_multiselect(self)`
  - `def test_text_number(self)`
  - `def test_text_none(self)`
  - `def test_text_bool(self)`
  - `def test_nullable(self)`
- **Class `TestGroupFilters`**:
  - `def test_basic(self)`
- **Class `TestDocstringExtraction`**:
  - `def test_sklearn_class(self)`

### File: `test_param_schema_extra.py`
- **Class `SimpleModel`**:
  - *Doc:* A simple model.
  - `def __init__(self, n_estimators, learning_rate, verbose)`
  - `def get_params(self, deep)`
- **Class `ModelWithOptionalHints`**:
  - *Doc:* Model with Optional-type params.
  - `def __init__(self, alpha, solver, max_features)`
- **Class `EnumModel`**:
  - `def __init__(self, mode)`
- **Class `LiteralModel`**:
  - `def __init__(self, method)`
- **Class `ListParamModel`**:
  - `def __init__(self, layers)`
- **Class `DictParamModel`**:
  - `def __init__(self, options)`
- **Class `NumpydocModel`**:
  - *Doc:* A model with numpydoc style docstring.
  - `def __init__(self, n_estimators, max_depth)`
- **Class `RstDocModel`**:
  - *Doc:* Model with rST-style docstring.
  - `def __init__(self, alpha, beta)`
- **Class `NoDocModel`**:
  - `def __init__(self, x)`
- **Class `VarArgModel`**:
  - `def __init__(self)`
- **Class `PrivateParamModel`**:
  - `def __init__(self, _internal, public)`
- **Class `TestIntrospectParams`**:
  - `def test_basic_model(self)`
  - `def test_basic_types(self)`
  - `def test_advanced_classification(self)`
  - `def test_optional_types(self)`
  - `def test_union_type(self)`
  - `def test_literal_type(self)`
  - `def test_list_type(self)`
  - `def test_dict_default(self)`
  - `def test_skip_params(self)`
  - `def test_extra_descriptions(self)`
  - `def test_instance_defaults(self)`
  - `def test_vararg_model(self)`
  - `def test_private_params_excluded(self)`
- **Class `TestIntrospectAdapter`**:
  - `def test_adapter_instance(self)`
  - `def test_adapter_class(self)`
- **Class `TestDocstringParsing`**:
  - `def test_google_style(self)`
  - `def test_numpydoc_style(self)`
  - `def test_rst_style(self)`
  - `def test_no_doc(self)`
- **Class `TestParamSpecToDict`**:
  - `def test_to_dict(self)`
- **Class `TestApplyParams`**:
  - `def test_basic_apply(self)`
  - `def test_no_change(self)`
  - `def test_type_conversion(self)`
  - `def test_float_conversion(self)`
  - `def test_bool_conversion(self)`
  - `def test_missing_param(self)`
- **Class `TestConvertValue`**:
  - `def test_none_nullable(self)`
  - `def test_none_not_nullable(self)`
  - `def test_empty_string_nullable(self)`
  - `def test_bool_string_conversion(self)`
  - `def test_multiselect_string(self)`
  - `def test_multiselect_list(self)`
  - `def test_text_numeric_float(self)`
  - `def test_text_numeric_int(self)`
  - `def test_text_none_string(self)`
  - `def test_text_bool_string(self)`
  - `def test_text_plain_string(self)`
  - `def test_select(self)`
- **Class `TestGroupFiltering`**:
  - `def test_basic_specs(self)`
  - `def test_advanced_specs(self)`

### File: `test_pipeline.py`
- **Class `TestColumnSelectorWrapper`**:
  - `def test_t001_all_mode(self, simple_numeric_df)`
  - `def test_t002_include_by_column_names(self, simple_numeric_df)`
  - `def test_t003_include_by_range(self, simple_numeric_df)`
  - `def test_t004_exclude_mode(self, simple_numeric_df)`
  - `def test_t005_column_meta_monotonic(self, simple_numeric_df)`
  - `def test_t005b_invalid_mode_raises(self, simple_numeric_df)`
  - `def test_t005c_include_nonexistent_column_warning(self, simple_numeric_df)`
- **Class `TestColPreprocessor`**:
  - `def test_t006_numeric_standard_scaler(self, simple_numeric_df)`
  - `def test_t007_all_scalers(self, simple_numeric_df, scaler)`
  - `def test_t008_mixed_data(self, mixed_df)`
  - `def test_t009_override_types(self, mixed_df)`
  - `def test_t010_knn_imputer(self, simple_numeric_df)`
  - `def test_t011_get_feature_names_out(self, mixed_df)`
- **Class `TestFeatureGenerator`**:
  - `def test_t012_none_passthrough(self, regression_data)`
  - `def test_t013_polynomial(self, regression_data)`
  - `def test_t014_interaction_only(self, regression_data)`
  - `def test_t015_feature_names_out(self, regression_data)`
- **Class `TestFeatureSelector`**:
  - `def test_t016_none_passthrough(self, regression_data)`
  - `def test_t017_lasso_regression(self, regression_data)`
  - `def test_t018_ridge(self, regression_data)`
  - `def test_t019_rfr_regression(self, regression_data)`
  - `def test_t020_rfc_classification(self, classification_data)`
  - `def test_t021_select_percentile(self, regression_data)`
  - `def test_t022_select_kbest(self, regression_data)`
  - `def test_t023_get_feature_names_out(self, regression_data)`
- **Class `TestBuildPipeline`**:
  - `def test_t024_regression_default(self, regression_data)`
  - `def test_t025_classification_default(self, classification_data)`
  - `def test_t026_with_feature_gen_and_selection(self, regression_data)`
  - `def test_t027_exclude_columns(self, regression_data)`
  - `def test_t028_xgb_pipeline(self, regression_data)`
- **Class `TestMonotonicAndGroups`**:
  - `def test_t029_apply_monotonic_non_support_no_error(self)`
  - `def test_t030_apply_monotonic_xgb(self)`
  - `def test_t031_extract_group_array(self)`
  - `def test_t031b_all_none_groups_returns_none(self)`
- **Function `simple_numeric_df`()**
  - *Doc:* 数値のみの小さな DataFrame。
- **Function `mixed_df`()**
  - *Doc:* 数値・カテゴリ・バイナリ混在 DataFrame。
- **Function `regression_data`()**
  - *Doc:* sklearn の make_regression データセット。
- **Function `classification_data`()**
  - *Doc:* sklearn の make_classification データセット。

### File: `test_pipeline_builder_extra.py`
- **Class `TestPipelineConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestBuildPipeline`**:
  - `def test_basic_regression_pipeline(self)`
  - `def test_classification_pipeline(self)`
  - `def test_with_estimator_params(self)`
- **Class `TestApplyMonotonicConstraints`**:
  - `def test_no_constraints(self)`
  - `def test_with_constraints_on_non_monotonic_model(self)`
- **Class `TestExtractGroupArray`**:
  - `def test_with_groups(self)`
  - `def test_no_groups(self)`
  - `def test_partial_groups(self)`

### File: `test_pipeline_comprehensive.py`
- **Class `TestColumnMeta`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestColumnSelectorWrapper`**:
  - `def sample_df(self)`
  - `def test_mode_all(self, sample_df)`
  - `def test_mode_include(self, sample_df)`
  - `def test_mode_exclude(self, sample_df)`
  - `def test_invalid_mode_raises(self, sample_df)`
  - `def test_empty_selection_raises(self, sample_df)`
  - `def test_non_dataframe_raises(self)`
  - `def test_get_feature_names_out(self, sample_df)`
  - `def test_selected_columns_property(self, sample_df)`
  - `def test_clone_compatibility(self)`
  - `def test_col_range(self)`
  - `def test_get_column_meta_default(self)`
  - `def test_get_monotonic_constraints(self, sample_df)`
  - `def test_get_groups_array(self, sample_df)`
  - `def test_missing_column_in_transform(self, sample_df)`
- **Class `TestExtractGroupArray`**:
  - `def test_basic(self)`
  - `def test_all_none(self)`
  - `def test_mixed(self)`
- **Class `TestPipelineConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestBuildPipeline`**:
  - `def test_minimal_pipeline(self)`
  - `def test_pipeline_with_monotonic(self)`
  - `def test_classification_pipeline(self)`
- **Class `TestApplyMonotonicConstraints`**:
  - `def test_no_constraints(self)`
  - `def test_with_hist_gb(self)`

### File: `test_pipeline_grid.py`
- **Class `TestCountCombinations`**:
  - `def test_tg001_single_each(self)`
  - `def test_tg001_two_estimators(self)`
  - `def test_tg001_two_scalers_two_estimators(self)`
  - `def test_tg001_full_grid(self)`
  - `def test_tg001_empty_list_defaults(self)`
- **Class `TestSingleCombination`**:
  - `def test_tg002_single(self, reg_df)`
  - `def test_tg002_name_format(self, reg_df)`
- **Class `TestMultipleImputerEstimator`**:
  - `def test_tg003_two_imputers_two_estimators(self)`
  - `def test_tg003_all_names_unique(self)`
  - `def test_tg003_config_fields(self)`
- **Class `TestFullGridAxes`**:
  - `def test_tg004_all_scalers(self)`
  - `def test_tg004_all_encoders(self)`
- **Class `TestFeatureGenGrid`**:
  - `def test_tg005_none_and_polynomial(self)`
  - `def test_tg005_polynomial_multi_degree(self)`
- **Class `TestFeatureSelGrid`**:
  - `def test_tg006_none_and_rfr(self)`
  - `def test_tg006_sel_methods_all_appear(self)`
- **Class `TestMaxCombinations`**:
  - `def test_tg007_limit(self)`
  - `def test_tg007_no_limit(self)`
- **Class `TestEndToEnd`**:
  - `def test_tg008_fit_predict_regression(self, reg_df)`
  - `def test_tg008_fit_predict_with_selection(self, reg_df)`
- **Class `TestEstimatorParamsList`**:
  - `def test_tg009_params_applied(self, reg_df)`
- **Class `TestClassificationGrid`**:
  - `def test_tg010_classification_fit_predict(self, clf_df)`
  - `def test_tg010_feature_sel_classification(self, clf_df)`
- **Class `TestCatImputersGrid`**:
  - `def test_tg011_cat_imputers_count(self)`
  - `def test_tg011_cat_imputers_config_applied(self, reg_df)`
  - `def test_tg011_fit_predict(self, reg_df)`
- **Class `TestBinaryImputersGrid`**:
  - `def test_tg012_binary_imputers_count(self)`
  - `def test_tg012_binary_imputers_config_applied(self)`
- **Class `TestExcludeColumns`**:
  - `def test_tg013_exclude_columns_mode(self)`
  - `def test_tg013_no_exclude(self)`
  - `def test_tg013_full_grid_with_exclude(self)`
- **Function `reg_df`()**
- **Function `clf_df`()**

### File: `test_pipeline_grid_comprehensive.py`
- **Class `TestPipelineGridConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestCountCombinations`**:
  - `def test_single(self)`
  - `def test_multiple(self)`
- **Class `TestEnsureNonempty`**:
  - `def test_nonempty(self)`
  - `def test_empty(self)`
- **Class `TestMakeLabel`**:
  - `def test_basic(self)`
- **Class `TestBuildGenCombinations`**:
  - `def test_none_only(self)`
  - `def test_polynomial(self)`
- **Class `TestGeneratePipelineGrid`**:
  - `def test_single(self)`
  - `def test_multiple_estimators(self)`
  - `def test_max_combinations(self)`
  - `def test_empty_estimators(self)`
  - `def test_with_feature_gen(self)`

### File: `test_pipeline_grid_extra.py`
- **Class `TestPipelineGridConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestHelpers`**:
  - `def test_ensure_nonempty_with_list(self)`
  - `def test_ensure_nonempty_empty(self)`
  - `def test_make_label(self)`
  - `def test_build_gen_combinations_none(self)`
  - `def test_build_gen_combinations_poly(self)`
- **Class `TestCountCombinations`**:
  - `def test_default_count(self)`
  - `def test_multi_estimator_count(self)`
  - `def test_multi_scaler_count(self)`
- **Class `TestGeneratePipelineGrid`**:
  - `def test_basic(self)`
  - `def test_multiple_estimators(self)`
  - `def test_max_combinations(self)`
  - `def test_empty_estimators_error(self)`
  - `def test_with_estimator_params(self)`

### File: `test_pipeline_modules.py`
- **Class `TestColPreprocessor`**:
  - `def test_default_config_fit_transform(self, mixed_df)`
  - `def test_feature_names_out(self, mixed_df)`
  - `def test_column_transformer_property(self, mixed_df)`
  - `def test_column_transformer_raises_before_fit(self)`
  - `def test_transform_raises_before_fit(self, mixed_df)`
  - `def test_numeric_scaler_variants(self, mixed_df)`
  - `def test_numeric_imputer_variants(self)`
  - `def test_override_types(self, mixed_df)`
  - `def test_encoder_onehot(self, mixed_df)`
  - `def test_encoder_ordinal(self, mixed_df)`
  - `def test_ndarray_input(self)`
  - `def test_empty_df_raises(self)`
- **Class `TestColumnSelectorWrapper`**:
  - `def test_all_mode(self, numeric_df)`
  - `def test_include_mode(self, numeric_df)`
  - `def test_exclude_mode(self, numeric_df)`
  - `def test_include_col_range(self, numeric_df)`
  - `def test_invalid_mode_raises(self, numeric_df)`
  - `def test_ndarray_raises(self)`
  - `def test_get_feature_names_out(self, numeric_df)`
  - `def test_selected_columns_property(self, numeric_df)`
  - `def test_column_meta(self)`
  - `def test_monotonic_constraints(self)`
  - `def test_groups_array(self)`
  - `def test_missing_column_warning(self, numeric_df)`
  - `def test_zero_selection_raises(self, numeric_df)`
- **Class `TestFeatureGenerator`**:
  - `def test_none_passthrough(self, numeric_df)`
  - `def test_polynomial(self, numeric_df)`
  - `def test_interaction_only(self, numeric_df)`
  - `def test_get_feature_names_out(self, numeric_df)`
  - `def test_is_passthrough(self)`
  - `def test_n_output_features(self, numeric_df)`
  - `def test_include_bias(self, numeric_df)`
  - `def test_ndarray_input(self)`
- **Class `TestFeatureSelector`**:
  - `def test_none_passthrough(self, numeric_df, regression_target)`
  - `def test_lasso_selection(self, numeric_df, regression_target)`
  - `def test_rfr_selection(self, numeric_df, regression_target)`
  - `def test_select_kbest(self, numeric_df, regression_target)`
  - `def test_select_percentile(self, numeric_df, regression_target)`
  - `def test_get_feature_names_out(self, numeric_df, regression_target)`
  - `def test_support_mask(self, numeric_df, regression_target)`
  - `def test_unknown_method_fallback(self, numeric_df, regression_target)`
  - `def test_classification_task(self)`
  - `def test_ndarray_input(self, regression_target)`
  - `def test_ridge_selection(self, numeric_df, regression_target)`
- **Function `mixed_df`()**
  - *Doc:* 数値・カテゴリ・バイナリ混合DataFrame。
- **Function `numeric_df`()**
  - *Doc:* 数値のみのDataFrame。
- **Function `regression_target`(numeric_df)**

### File: `test_preprocessor_comprehensive.py`
- **Class `TestLogTransformer`**:
  - `def test_transform(self)`
  - `def test_inverse(self)`
  - `def test_fit_returns_self(self)`
- **Class `TestSinCosTransformer`**:
  - `def test_transform(self)`
  - `def test_feature_names(self)`
  - `def test_period_360(self)`
- **Class `TestPreprocessConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestPreprocessor`**:
  - `def test_build(self, detection_result)`
  - `def test_transformer_property(self, detection_result)`
  - `def test_transformer_before_build(self)`
  - `def test_fit_transform(self, detection_result, mixed_df)`
  - `def test_custom_scaler(self, detection_result)`
  - `def test_knn_imputer(self, detection_result)`
- **Class `TestBuildFullPipeline`**:
  - `def test_basic(self, detection_result, mixed_df)`
- **Function `mixed_df`()**
- **Function `detection_result`(mixed_df)**

### File: `test_preprocessor_extra.py`
- **Class `TestLogTransformer`**:
  - `def test_transform(self)`
  - `def test_inverse(self)`
  - `def test_offset_zero(self)`
- **Class `TestSinCosTransformer`**:
  - `def test_default_period(self)`
  - `def test_degree_period(self)`
  - `def test_feature_names(self)`
- **Class `TestPreprocessConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestPreprocessor`**:
  - `def _make_detection_result(self)`
  - `def test_build(self)`
  - `def test_build_with_target(self)`
  - `def test_transformer_property_error(self)`
  - `def test_transformer_property(self)`
  - `def test_different_scalers(self)`
  - `def test_knn_imputer(self)`
- **Class `TestBuildFullPipeline`**:
  - `def test_basic(self)`

### File: `test_preset_manager.py`
- **Class `TestSavePreset`**:
  - `def test_save_creates_file(self, tmp_preset_dir, sample_state)`
  - `def test_save_content(self, tmp_preset_dir, sample_state)`
  - `def test_save_empty_name_raises(self, tmp_preset_dir, sample_state)`
  - `def test_save_excludes_non_pipeline_keys(self, tmp_preset_dir, sample_state)`
  - `def test_save_special_chars_in_name(self, tmp_preset_dir, sample_state)`
- **Class `TestLoadPreset`**:
  - `def test_load_restores_state(self, tmp_preset_dir, sample_state)`
  - `def test_load_nonexistent_raises(self, tmp_preset_dir)`
  - `def test_load_does_not_overwrite_non_pipeline_keys(self, tmp_preset_dir, sample_state)`
- **Class `TestListPresets`**:
  - `def test_list_empty(self, tmp_preset_dir)`
  - `def test_list_multiple(self, tmp_preset_dir, sample_state)`
- **Class `TestDeletePreset`**:
  - `def test_delete_existing(self, tmp_preset_dir, sample_state)`
  - `def test_delete_nonexistent(self, tmp_preset_dir)`
- **Class `TestExportStateSummary`**:
  - `def test_export(self, sample_state)`
  - `def test_export_empty(self)`
- **Class `TestMakeSerializable`**:
  - `def test_numpy_types(self)`
  - `def test_nested_dict(self)`
  - `def test_plain_types(self)`
- **Class `TestRecordAnalysis`**:
  - `def test_record_creates_file(self, tmp_path, sample_state)`
- **Class `TestListHistory`**:
  - `def test_list_empty(self, tmp_path)`
  - `def test_list_after_record(self, tmp_path, sample_state)`
- **Class `TestExportImportConfigYaml`**:
  - `def test_export_yaml(self, sample_state)`
  - `def test_import_yaml(self, sample_state)`
  - `def test_roundtrip(self, sample_state)`
- **Function `tmp_preset_dir`(tmp_path)**
  - *Doc:* テスト用のプリセットディレクトリ。
- **Function `sample_state`()**
  - *Doc:* テスト用のstate辞書。

### File: `test_protonation.py`
- **Class `_FakeConfig`**:
  - *Doc:* MoleculeChargeConfig の最小モック。
  - `def __init__(self, mode, ph)`
- **Class `TestNeutralize`**:
  - *Doc:* RDKit MolStandardize による中性化。
  - `def test_sodium_carboxylate(self)`
  - `def test_already_neutral(self)`
  - `def test_invalid_smiles(self)`
  - `def test_empty_string(self)`
  - `def test_ammonium(self)`
- **Class `TestMaxDeprotonate`**:
  - `def test_carboxylic_acid(self)`
  - `def test_invalid_smiles(self)`
  - `def test_benzene(self)`
- **Class `TestMaxProtonate`**:
  - `def test_amine(self)`
  - `def test_invalid_smiles(self)`
- **Class `TestApplyProtonation`**:
  - `def test_as_is(self)`
  - `def test_neutral_mode(self)`
  - `def test_max_acid_mode(self)`
  - `def test_max_base_mode(self)`
  - `def test_unknown_mode(self)`
  - `def test_none_input(self)`
  - `def test_empty_string(self)`
  - `def test_non_string_input(self)`
  - `def test_auto_ph_without_unipka(self)`
- **Class `TestApplyProtonationBatch`**:
  - `def test_batch(self)`
  - `def test_empty_list(self)`
  - `def test_batch_as_is(self)`

### File: `test_psmiles_adapter.py`
- **Class `TestPSmilesAdapter`**:
  - `def _adapter(self)`
  - `def test_is_available(self)`
  - `def test_name(self)`
  - `def test_description(self)`
  - `def test_is_psmiles_true(self)`
  - `def test_is_psmiles_false(self)`
  - `def test_is_psmiles_non_string(self)`
  - `def test_fallback_simple_psmiles(self)`
  - `def test_fallback_isotope_psmiles(self)`
  - `def test_fallback_star_only(self)`
  - `def test_compute_psmiles_basic(self)`
  - `def test_compute_has_expected_columns(self)`
  - `def test_compute_morgan_fp_present(self)`
  - `def test_compute_empty_list(self)`
  - `def test_compute_invalid_input(self)`
  - `def test_compute_mixed_valid_invalid(self)`
  - `def test_compute_regular_smiles_treated_as_psmiles(self)`
  - `def test_adapter_name_in_result(self)`
  - `def test_metadata_has_psmiles_lib(self)`

### File: `test_random_projection.py`
- **Class `TestJLRandomProjectionPassthrough`**:
  - *Doc:* 低次元入力では射影が適用されずpassthroughとなる。
  - `def test_passthrough_flag(self, low_dim_X)`
  - `def test_passthrough_identity(self, low_dim_X)`
- **Class `TestJLRandomProjectionActive`**:
  - *Doc:* 高次元入力では射影が適用され次元削減される。
  - `def test_projection_active(self, high_dim_X)`
  - `def test_dimension_reduction(self, high_dim_X)`
  - `def test_deterministic_with_seed(self, high_dim_X)`
- **Class `TestSummary`**:
  - `def test_summary_passthrough(self, low_dim_X)`
  - `def test_summary_active(self, high_dim_X)`
- **Class `TestFeatureNamesOut`**:
  - `def test_passthrough_names(self, low_dim_X)`
  - `def test_active_names(self, high_dim_X)`
- **Class `TestMethodAuto`**:
  - `def test_auto_selects_sparse_for_highdim(self)`
  - `def test_auto_selects_gaussian_for_lowdim(self)`
- **Class `TestMethodFixed`**:
  - `def test_sparse_fixed(self, high_dim_X)`
  - `def test_gaussian_fixed(self, high_dim_X)`
- **Class `TestShouldApply`**:
  - `def test_should_not_apply_low_dim(self)`
  - `def test_should_apply_high_dim(self)`
- **Class `TestBuildPipelineWithJLRP`**:
  - `def test_jl_rp_step_present(self, regression_df)`
  - `def test_pipeline_fits(self, regression_df)`
- **Class `TestBuildPipelineWithoutJLRP`**:
  - `def test_no_jl_rp_step_by_default(self, regression_df)`
- **Class `TestPreprocessConfigRP`**:
  - `def test_default_disabled(self)`
  - `def test_custom_config(self)`
- **Class `TestRunMultiFeatureSets`**:
  - *Doc:* AutoMLEngine.run_multi_feature_sets のテスト。
  - `def test_multi_sets_basic(self, regression_df)`
- **Class `TestNotFittedError`**:
  - `def test_transform_before_fit(self)`
- **Class `TestSparseInput`**:
  - `def test_sparse_matrix_passthrough(self)`
  - `def test_sparse_matrix_projection(self)`
- **Function `low_dim_X`()**
  - *Doc:* JL条件を満たさない低次元データ (50サンプル×5特徴量)。
- **Function `high_dim_X`()**
  - *Doc:* JL条件を満たす高次元データ (100サンプル×2000特徴量)。
- **Function `regression_df`()**
  - *Doc:* 20列の数値特徴量を持つ回帰データ。

### File: `test_recommender.py`
- **Class `TestRecommenderDatabase`**:
  - `def test_all_recommendations_not_empty(self)`
  - `def test_at_least_14_targets(self)`
  - `def test_each_target_has_8_descriptors(self)`
  - `def test_target_names_unique(self)`
  - `def test_all_descriptors_have_required_fields(self)`
  - `def test_target_recommendations_have_required_fields(self)`
  - `def test_known_libraries_used(self)`
- **Class `TestGetTargetRecommendationByName`**:
  - `def test_exact_match(self)`
  - `def test_partial_match(self)`
  - `def test_english_match(self)`
  - `def test_no_match_returns_none(self)`
  - `def test_case_insensitive(self)`
- **Class `TestGetTargetNames`**:
  - `def test_returns_list_of_strings(self)`
  - `def test_count_matches_recommendations(self)`
- **Class `TestGetTargetCategories`**:
  - `def test_returns_unique_categories(self)`
  - `def test_known_categories_present(self)`
- **Class `TestGetTargetsByCategory`**:
  - `def test_returns_correct_category(self)`
  - `def test_empty_for_unknown_category(self)`
- **Class `TestGetAllDescriptorCategories`**:
  - `def test_returns_unique_list(self)`
  - `def test_known_descriptor_categories(self)`
- **Class `TestGetDescriptorsByCategory`**:
  - `def test_returns_descriptors(self)`
  - `def test_no_duplicates(self)`
  - `def test_empty_for_unknown(self)`
- **Class `TestSpecificTargets`**:
  - `def test_refractive_index_has_molmr(self)`
  - `def test_tg_has_backbone_flexibility(self)`
  - `def test_toxicity_has_logp(self)`
  - `def test_viscosity_has_molwt(self)`

### File: `test_recommender_comprehensive.py`
- **Class `TestDescriptorInfo`**:
  - `def test_fields(self)`
- **Class `TestTargetRecommendations`**:
  - `def test_fields(self)`
- **Class `TestGetAllRecommendations`**:
  - `def test_returns_list(self)`
  - `def test_each_has_descriptors(self)`
- **Class `TestGetByName`**:
  - `def test_exact(self)`
  - `def test_partial(self)`
  - `def test_not_found(self)`
- **Class `TestGetTargetNames`**:
  - `def test_returns_list(self)`
- **Class `TestGetTargetCategories`**:
  - `def test_unique(self)`
  - `def test_known_categories(self)`
- **Class `TestGetTargetsByCategory`**:
  - `def test_filter(self)`
  - `def test_empty(self)`
- **Class `TestGetDescriptorCategories`**:
  - `def test_returns(self)`
  - `def test_known(self)`
- **Class `TestGetDescriptorsByCategory`**:
  - `def test_filter(self)`
  - `def test_unique_names(self)`

### File: `test_rgf_comprehensive.py`
- **Class `TestUtilities`**:
  - `def test_to_numpy_from_list(self)`
  - `def test_to_numpy_from_df(self)`
  - `def test_sigmoid_basic(self)`
  - `def test_sigmoid_clipping(self)`
  - `def test_softmax_basic(self)`
- **Class `TestRGFRegressor`**:
  - `def data_reg(self)`
  - `def test_fit_predict(self, data_reg)`
  - `def test_n_features_in(self, data_reg)`
  - `def test_clone_compatibility(self, data_reg)`
  - `def test_cross_val_score(self, data_reg)`
  - `def test_pipeline_compatibility(self, data_reg)`
  - `def test_subsample(self, data_reg)`
  - `def test_l1_regularization(self, data_reg)`
  - `def test_small_dataset(self)`
- **Class `TestRGFClassifier`**:
  - `def data_binary(self)`
  - `def data_multiclass(self)`
  - `def test_binary_fit_predict(self, data_binary)`
  - `def test_binary_predict_proba(self, data_binary)`
  - `def test_multiclass_fit_predict(self, data_multiclass)`
  - `def test_multiclass_predict_proba(self, data_multiclass)`
  - `def test_clone_compatibility(self)`
  - `def test_binary_subsample(self, data_binary)`
- **Class `TestRGFCore`**:
  - `def test_init_forest_state(self)`
  - `def test_predict_from_weights_empty(self)`

### File: `test_rgf_extra.py`
- **Class `TestRGFUtilities`**:
  - `def test_to_numpy_array(self)`
  - `def test_to_numpy_dataframe(self)`
  - `def test_sigmoid(self)`
  - `def test_softmax(self)`
- **Class `TestRGFRegressor`**:
  - `def test_fit_predict(self)`
  - `def test_with_l1(self)`
  - `def test_subsample(self)`
  - `def test_with_dataframe(self)`
- **Class `TestRGFClassifier`**:
  - `def test_binary_fit_predict(self)`
  - `def test_binary_predict_proba(self)`
  - `def test_multiclass(self)`
  - `def test_multiclass_proba(self)`
  - `def test_with_l1_subsample(self)`

### File: `test_search_space_extra.py`
- **Class `TestVariable`**:
  - `def test_continuous(self)`
  - `def test_discrete(self)`
  - `def test_categorical(self)`
  - `def test_grid_continuous(self)`
  - `def test_grid_discrete(self)`
  - `def test_grid_categorical(self)`
  - `def test_validation_no_bounds(self)`
  - `def test_validation_lo_gt_hi(self)`
  - `def test_validation_discrete_no_step(self)`
  - `def test_validation_categorical_empty(self)`
  - `def test_validation_categorical_none(self)`
- **Class `TestSearchSpaceBasic`**:
  - `def test_add_variables(self)`
  - `def test_init_with_list(self)`
  - `def test_estimate_grid_size(self)`
  - `def test_estimate_empty(self)`
  - `def test_auto_recommend_grid(self)`
  - `def test_auto_recommend_random(self)`
- **Class `TestGenerateCandidates`**:
  - `def test_grid(self)`
  - `def test_random(self)`
  - `def test_lhs(self)`
  - `def test_random_lhs(self)`
  - `def test_grid_downsample(self)`
  - `def test_auto(self)`
  - `def test_invalid_method(self)`
  - `def test_empty_variables(self)`
  - `def test_lhs_with_discrete(self)`
  - `def test_lhs_with_categorical(self)`
- **Class `TestFromDataFrame`**:
  - `def test_numeric_columns(self)`
  - `def test_integer_columns(self)`
  - `def test_specific_columns(self)`
  - `def test_margin(self)`
  - `def test_categorical_column(self)`

### File: `test_search_space_generator.py`
- **Class `TestGenerateGridSpace`**:
  - `def test_rf_grid(self)`
  - `def test_ridge_grid(self)`
  - `def test_svr_grid(self)`
  - `def test_empty_specs(self)`
- **Class `TestGenerateOptunaSpace`**:
  - `def test_rf_optuna(self)`
  - `def test_ridge_optuna(self)`
  - `def test_empty_specs(self)`
- **Class `TestGenerateSearchSpaces`**:
  - `def test_rf_all(self)`
  - `def test_include_advanced(self)`
- **Class `TestFromEstimator`**:
  - `def test_various_estimators(self, cls_name)`
  - `def test_custom_estimator(self)`
- **Class `TestKnownPresets`**:
  - `def test_known_params_have_presets(self, param_name)`
  - `def test_learning_rate_log_scale(self)`
  - `def test_l1_ratio_bounded(self)`
- **Class `TestAutoInference`**:
  - `def test_int_auto(self)`
  - `def test_float_auto(self)`
  - `def test_bool_auto(self)`
  - `def test_select_auto(self)`
  - `def test_str_returns_none(self)`
- **Class `TestSearchParamSpecMethods`**:
  - `def test_to_grid_entry_enabled(self)`
  - `def test_to_grid_entry_disabled(self)`
  - `def test_to_grid_entry_empty(self)`
  - `def test_to_optuna_entry_int(self)`
  - `def test_to_optuna_entry_float_log(self)`
  - `def test_to_optuna_entry_categorical(self)`
  - `def test_to_dict(self)`
- **Class `TestParseValueList`**:
  - `def test_int_list(self)`
  - `def test_float_list(self)`
  - `def test_string_list(self)`
  - `def test_bool_list(self)`
  - `def test_none_in_list(self)`
  - `def test_mixed_types(self)`
  - `def test_empty(self)`
  - `def test_single_value(self)`
- **Class `TestRangeInference`**:
  - `def test_int_range_default_100(self)`
  - `def test_int_range_default_1(self)`
  - `def test_int_range_default_5(self)`
  - `def test_float_range_default_01(self)`
  - `def test_float_range_default_0(self)`
  - `def test_float_range_default_10(self)`
- **Class `TestEstimatorConfig`**:
  - `def test_to_dict(self)`
  - `def test_default_empty(self)`
- **Class `TestDocstringIntegration`**:
  - *Doc:* docstringからパラメータ説明が正しく取得されること
  - `def test_rf_descriptions(self)`
  - `def test_rf_description_no_type_info(self)`
  - `def test_ridge_descriptions(self)`
  - `def test_ridge_alpha_description_quality(self)`
  - `def test_gbm_learning_rate_description(self)`
  - `def test_custom_estimator_descriptions(self)`
  - `def test_no_docstring_estimator(self)`
  - `def test_rst_style_docstring(self)`
- **Class `TestEndToEnd`**:
  - *Doc:* estimator → ParamSpec → SearchParamSpec → tuner形式の変換テスト
  - `def test_rf_full_pipeline(self)`
  - `def test_custom_estimator_full_pipeline(self)`
  - `def test_estimator_config_round_trip(self)`

### File: `test_shap_explainer_extra.py`
- **Class `TestShapConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestShapResult`**:
  - `def _make_result(self, n, d, multiclass)`
  - `def test_basic_properties(self)`
  - `def test_multiclass(self)`
  - `def test_feature_importance(self)`
  - `def test_top_features(self)`
- **Class `TestShapExplainer`**:
  - `def test_auto_select_tree(self)`
  - `def test_explain_tree_model(self)`
  - `def test_explain_linear_model(self)`

### File: `test_skfp_adapter.py`
- **Class `TestSkfpAdapter`**:
  - *Doc:* scikit-fingerprints アダプタのテスト
  - `def test_name(self)`
  - `def test_is_available_returns_bool(self)`
  - `def test_description_not_empty(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_compute_returns_descriptor_result(self)`
  - `def test_compute_with_invalid_smiles(self)`
  - `def test_custom_fp_types(self)`
  - `def test_custom_fp_configs(self)`

### File: `test_smiles_pipeline.py`
- **Class `TestSmilesAutoMLPipeline`**:
  - *Doc:* SMILES列展開後のAutoMLパイプラインのend-to-endテスト。
  - `def test_fit_predict_with_smiles_col(self)`
  - `def test_fit_predict_no_smiles_col(self)`
  - `def test_fit_predict_selected_descriptors(self)`
  - `def test_invalid_descriptor_names(self)`
- **Function `test_smiles_pipeline_pre_expanded_fit`()**
  - *Doc:* 事前計算済みのデータでfitし、生データ（SMILESのみ）でpredictできることを検証する。

### File: `test_smiles_transformer.py`
- **Class `TestSmilesDescriptorTransformer`**:
  - `def test_init(self)`
  - `def test_compute_psmiles(self, mock_compute, mock_avail, mock_is_psmiles)`
  - `def test_compute_plugins_success(self, mock_get_plugins, mock_compute_all)`
  - `def test_compute_fallback_adapters(self, mock_md_compute, mock_md_avail, mock_rd_compute, mock_rd_avail, mock_compute_all)`
  - `def test_compute_plugins_exception(self, mock_md_avail, mock_rd_avail, mock_compute_all)`
  - `def test_identify_count_columns(self)`
  - `def test_compute_molar_volumes(self)`
  - `def test_apply_count_normalization(self)`
  - `def test_fit_transform(self, mock_compute)`
  - `def test_streamlit_cache_reuse(self, mock_compute)`
- **Class `TestPrecalculateFunctions`**:
  - `def test_progressive_precalculate_psmiles(self, mock_compute, mock_avail, mock_is_psmiles)`
  - `def test_progressive_empty(self)`
  - `def test_progressive_normal(self, mock_rd_meta, mock_rd_compute, mock_rd_avail, mock_rec)`
  - `def test_precalc_empty(self)`
  - `def test_precalculate_all(self, mock_rd_meta, mock_rd_compute, mock_rd_avail, mock_get_rec)`
- **Function `smiles_df`()**

### File: `test_sri.py`
- **Class `_MockShapResult`**:
  - *Doc:* SRI分解テスト用のモックShapResult
- **Class `TestSRIDecomposer`**:
  - `def test_basic_decomposition(self)`
  - `def test_symmetry(self)`
  - `def test_diagonal_zero(self)`
  - `def test_independence_nonnegative(self)`
  - `def test_total_sri_tuple(self)`
  - `def test_center_false(self)`
  - `def test_correlated_features_high_redundancy(self)`
  - `def test_single_feature(self)`
  - `def test_3d_shap_multiclass(self)`
  - `def test_invalid_1d_raises(self)`
- **Class `TestSRIResult`**:
  - `def sri_result(self)`
  - `def test_summary_df(self, sri_result)`
  - `def test_summary_df_normalized(self, sri_result)`
  - `def test_pairwise_df(self, sri_result)`
- **Class `TestSelectFeatures`**:
  - `def sri_result(self)`
  - `def test_top_n(self, sri_result)`
  - `def test_threshold(self, sri_result)`
  - `def test_no_filter(self, sri_result)`

### File: `test_sri_extra.py`
- **Class `TestSRIDecomposer`**:
  - `def test_basic_decompose(self)`
  - `def test_no_center(self)`
  - `def test_multiclass(self)`
  - `def test_invalid_shape(self)`
- **Class `TestSRIResult`**:
  - `def _make_result(self)`
  - `def test_summary_df(self)`
  - `def test_pairwise_df(self)`
- **Class `TestSelectFeatures`**:
  - `def _make_result(self)`
  - `def test_top_n(self)`
  - `def test_threshold(self)`
  - `def test_all(self)`

### File: `test_tuner.py`
- **Class `TestTunerConfig`**:
  - `def test_default_values(self)`
  - `def test_custom_values(self)`
- **Class `TestGridSearch`**:
  - `def test_grid_search_basic(self, regression_data)`
  - `def test_grid_search_refit(self, regression_data)`
- **Class `TestRandomSearch`**:
  - `def test_random_search_basic(self, regression_data)`
  - `def test_random_search_reproducible(self, regression_data)`
- **Class `TestHalvingSearch`**:
  - `def test_halving_grid(self, regression_data)`
  - `def test_halving_random(self, regression_data)`
- **Class `TestOptuna`**:
  - `def test_optuna_or_fallback(self, regression_data)`
- **Class `TestBayesSearch`**:
  - `def test_bayes_or_fallback(self, regression_data)`
- **Class `TestErrorHandling`**:
  - `def test_unknown_method_raises(self, regression_data)`
  - `def test_empty_param_grid(self, regression_data)`
- **Class `TestCVResults`**:
  - `def test_cv_results_is_dataframe(self, regression_data)`
- **Function `regression_data`()**
- **Function `classification_data`()**

### File: `test_tuner_comprehensive.py`
- **Class `TestTunerConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestConvertOptunaGrid`**:
  - `def test_float_grid(self)`
  - `def test_float_log_grid(self)`
  - `def test_int_grid(self)`
  - `def test_categorical_grid(self)`
  - `def test_passthrough_list(self)`
- **Class `TestTuneGrid`**:
  - `def reg_data(self)`
  - `def test_grid_search(self, reg_data)`
  - `def test_random_search(self, reg_data)`
- **Class `TestTuneUnknown`**:
  - `def test_unknown_method(self)`
- **Class `TestTuneHalving`**:
  - `def reg_data(self)`
  - `def test_halving_grid(self, reg_data)`
  - `def test_halving_random(self, reg_data)`
- **Class `TestTuneOptuna`**:
  - `def reg_data(self)`
  - `def test_optuna_or_fallback(self, reg_data)`
- **Class `TestTuneBayes`**:
  - `def reg_data(self)`
  - `def test_bayes_or_fallback(self, reg_data)`

### File: `test_tuner_extra.py`
- **Class `TestTunerConfig`**:
  - `def test_defaults(self)`
  - `def test_custom(self)`
- **Class `TestConvertOptunaGrid`**:
  - `def test_float_param(self)`
  - `def test_float_log(self)`
  - `def test_int_param(self)`
  - `def test_categorical_param(self)`
  - `def test_unknown_type(self)`
  - `def test_list_passthrough(self)`
  - `def test_mixed_grid(self)`
  - `def test_empty_grid(self)`
- **Class `TestTuneGrid`**:
  - `def test_grid_search(self)`
- **Class `TestTuneRandom`**:
  - `def test_random_search(self)`
- **Class `TestTuneHalving`**:
  - `def test_halving_grid(self)`
  - `def test_halving_random(self)`
- **Class `TestTuneOptuna`**:
  - `def test_optuna_or_fallback(self)`
- **Class `TestTuneBayes`**:
  - `def test_bayes_or_fallback(self)`
- **Class `TestTuneUnknown`**:
  - `def test_unknown_method_raises(self)`

### File: `test_tuner_pipeline.py`
- **Class `TestDetectPipelineStepName`**:
  - `def test_pipeline_with_estimator_step(self)`
  - `def test_pipeline_with_model_step(self)`
  - `def test_bare_estimator(self)`
  - `def test_single_step_pipeline(self)`
- **Class `TestPrefixParamGrid`**:
  - `def test_basic_prefix(self)`
  - `def test_already_prefixed(self)`
  - `def test_nested_key_preserved(self)`
  - `def test_empty_grid(self)`
- **Class `TestStripPrefix`**:
  - `def test_basic_strip(self)`
  - `def test_no_prefix(self)`
  - `def test_mixed(self)`
- **Class `TestTuneWithPipeline`**:
  - `def data(self)`
  - `def test_grid_with_pipeline(self, data)`
  - `def test_grid_with_bare_estimator(self, data)`
  - `def test_random_with_pipeline(self, data)`
  - `def test_multiple_params_pipeline(self, data)`

### File: `test_type_detector_comprehensive.py`
- **Class `TestColumnInfo`**:
  - `def test_is_numeric(self)`
  - `def test_is_not_numeric(self)`
  - `def test_is_categorical(self)`
  - `def test_is_not_categorical(self)`
- **Class `TestTypeDetector`**:
  - `def detector(self)`
  - `def test_constant_column(self, detector)`
  - `def test_binary_numeric(self, detector)`
  - `def test_binary_string(self, detector)`
  - `def test_category_low(self, detector)`
  - `def test_category_high(self, detector)`
  - `def test_numeric_normal(self, detector)`
  - `def test_numeric_log_skewed(self, detector)`
  - `def test_datetime_column(self, detector)`
  - `def test_datetime_string(self, detector)`
  - `def test_text_column(self, detector)`
  - `def test_smiles_by_name(self, detector)`
  - `def test_periodic_column(self)`
  - `def test_null_handling(self, detector)`
- **Class `TestDetectionResult`**:
  - `def test_summary_table(self)`
  - `def test_numeric_columns(self)`
  - `def test_categorical_columns(self)`
  - `def test_ignored_columns(self)`
  - `def test_get_columns_by_type(self)`

### File: `test_type_detector_extra.py`
- **Class `TestColumnInfo`**:
  - `def test_is_numeric(self)`
  - `def test_is_categorical(self)`
  - `def test_numeric_log(self)`
  - `def test_categorical_high(self)`
- **Class `TestTypeDetector`**:
  - `def test_numeric_normal(self)`
  - `def test_numeric_log(self)`
  - `def test_binary_numeric(self)`
  - `def test_binary_string(self)`
  - `def test_category_low(self)`
  - `def test_category_high(self)`
  - `def test_constant(self)`
  - `def test_datetime(self)`
  - `def test_text_long(self)`
  - `def test_smiles_by_name(self)`
  - `def test_periodic_user_specified(self)`
  - `def test_null_rate(self)`
- **Class `TestDetectionResult`**:
  - `def _make_result(self)`
  - `def test_numeric_columns(self)`
  - `def test_categorical_columns(self)`
  - `def test_binary_columns(self)`
  - `def test_constant_columns(self)`
  - `def test_summary_table(self)`
  - `def test_ignored_columns(self)`
  - `def test_get_columns_by_type(self)`

### File: `test_uma_adapter.py`
- **Class `TestUMAAdapterProperties`**:
  - *Doc:* F-uma-001: UMAAdapter の基本プロパティテスト
  - `def test_name(self)`
  - `def test_description_not_empty(self)`
  - `def test_default_model_name(self)`
  - `def test_default_device(self)`
  - `def test_custom_model_name(self)`
  - `def test_custom_device(self)`
- **Class `TestUMAAdapterAvailability`**:
  - *Doc:* F-uma-002: is_available のテスト
  - `def test_is_available_with_fairchem(self)`
  - `def test_is_available_without_fairchem(self)`
- **Class `TestUMAAdapterDescriptors`**:
  - *Doc:* F-uma-003: 記述子メタデータのテスト
  - `def test_descriptor_names_count(self)`
  - `def test_descriptor_names_prefix(self)`
  - `def test_descriptor_metadata_consistency(self)`
  - `def test_descriptor_metadata_fields(self)`
  - `def test_expected_descriptors_present(self)`
- **Class `TestSmilesToAseAtoms`**:
  - *Doc:* F-uma-004: SMILES → ASE Atoms 変換テスト
  - `def test_valid_smiles(self)`
  - `def test_invalid_smiles(self)`
  - `def test_atom_count_ethanol(self)`
  - `def test_pbc_is_false(self)`
  - `def test_charge_info(self)`
  - `def test_charged_molecule(self)`
- **Class `TestUMAAdapterCompute`**:
  - *Doc:* F-uma-005: compute メソッドのテスト（モック使用）
  - `def _make_adapter_with_mock(self)`
  - `def test_compute_returns_descriptor_result(self)`
  - `def test_compute_energy_values(self)`
  - `def test_compute_failed_smiles(self)`
  - `def test_compute_multiple_smiles(self)`
  - `def test_compute_mixed_success_failure(self)`
  - `def test_compute_success_rate(self)`
  - `def test_compute_metadata(self)`
- **Class `TestUMAInitIntegration`**:
  - *Doc:* F-uma-006: __init__.py からの安全importテスト
  - `def test_import_from_init(self)`
  - `def test_unavailable_fallback(self)`

### File: `test_utils_and_base.py`
- **Class `TestConfig`**:
  - `def test_random_state_exists(self)`
  - `def test_project_root_is_dir(self)`
  - `def test_app_config_defaults(self)`
  - `def test_app_config_custom(self)`
  - `def test_default_config_instance(self)`
  - `def test_threshold_constants(self)`
  - `def test_automl_constants(self)`
- **Class `TestOptionalImport`**:
  - `def test_safe_import_existing_module(self)`
  - `def test_safe_import_missing_module(self)`
  - `def test_safe_import_with_alias(self)`
  - `def test_is_available_missing(self)`
  - `def test_require_available(self)`
  - `def test_require_unavailable(self)`
  - `def test_get_availability_report(self)`
  - `def test_probe_all_optional_libraries(self)`
- **Class `TestDescriptorResult`**:
  - `def test_success_rate_all_ok(self)`
  - `def test_success_rate_some_failed(self)`
  - `def test_success_rate_empty(self)`
  - `def test_n_descriptors(self)`
- **Class `TestDescriptorMetadata`**:
  - `def test_creation(self)`
- **Class `TestBaseChemAdapter`**:
  - `def test_concrete_subclass(self)`
  - `def test_require_available_raises(self)`
  - `def test_repr(self)`
  - `def test_get_descriptor_names_default(self)`
- **Class `TestBenchmarkDatasets`**:
  - `def test_list_benchmark_datasets(self)`
  - `def test_list_has_required_fields(self)`
  - `def test_load_benchmark_invalid(self)`
  - `def test_benchmark_urls_exist(self)`
- **Class `TestMolAITokenizer`**:
  - `def test_tokenize_simple(self)`
  - `def test_tokenize_br_cl(self)`
  - `def test_smiles_to_onehot_shape(self)`
- **Class `TestMolAIAdapter`**:
  - `def test_name_and_description(self)`
  - `def test_get_descriptor_names(self)`
  - `def test_get_descriptors_metadata(self)`
  - `def test_compute_if_torch_available(self)`

### File: `verify_automl_smiles_fix.py`
- **Function `test_automl_smiles_only`()**
