# 📥 AI モデル重みの手動ダウンロード手順

本ドキュメントでは、エンタープライズ環境や研究機関のネットワーク制約等により、ChemAI内の「自動ダウンロード機能」が利用できない場合に、深層学習等のモデル重みを手動でダウンロードして配置する手順を解説します。

## 事前準備

1. [Hugging Face アカウント](https://huggingface.co/join) の作成
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens) でトークンを発行（Read 権限で十分です）

## ダウンロード手順

### 方法 A: ブラウザからの個別ファイルダウンロード（推奨・小規模モデル向け）

1. 対象のリポジトリページにアクセスします。
   例: *MolAI 化学構造エンコーダ (v1)* の場合: `https://huggingface.co/jckkvs/molai-chem-v1`
2. 上部タブから **「Files and versions」** を開きます。
3. 必要なファイル（`pytorch_model.bin`, `config.json`, `tokenizer.json` 等）を右クリックして「リンク先を保存」を選択します。
4. ChemAI のローカルフォルダ構成を以下のように調整し、該当モデルのディレクトリへ配置してください：
   ```text
   chemai2/
   └── models/
       └── molai-chem-v1/                 # リポジトリ名に基づくディレクトリ
           ├── pytorch_model.bin
           ├── config.json
           └── tokenizer.json
   ```

### 方法 B: git-lfs を利用（大規模・複数モデル向け）

`git` および `git-lfs` がインストールされている環境では、リポジトリ全体を高速にクローンすることが可能です。

```bash
# git-lfs を初期化
git lfs install

# 認証情報の保持を設定（初回アクセス時にトークン入力を求められます）
git config --global credential.helper store

# モデルを models ディレクトリにクローン
mkdir -p models
git lfs clone https://huggingface.co/jckkvs/molai-chem-v1 models/molai-chem-v1
```

## 🔐 プロキシ環境下での対応

プロキシを経由する必要がある場合、以下の方法で環境変数を設定してダウンロードを行ってください。

### ブラウザ経由の場合
お使いのブラウザや OS のプロキシ設定に従って通常のダウンロードを行ってください。

### git-lfs または CLI を利用する場合

```bash
# --- Windows (PowerShell) の場合 ---
$env:HTTP_PROXY="http://proxy.example.com:8080"
$env:HTTPS_PROXY="http://proxy.example.com:8080"

# --- Linux / macOS (bash/zsh) の場合 ---
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"

# 設定後、再度 git lfs clone を実行
git lfs clone https://huggingface.co/jckkvs/molai-chem-v1 models/molai-chem-v1
```

## ✅ 動作確認

配置が完了したら、正常にロードできるかを以下の Python スクリプトでテスト可能です：

```python
# python にて確認
from backend.chem.molai_adapter import MolAIAdapter
adapter = MolAIAdapter(model_path="models/molai-chem-v1")
print("MolAI Loaded:", adapter.is_available())  # True が出力されれば成功
```

> **Note**: アプリケーションは起動時に `models/` ディレクトリをスキャンし、利用可能な重みを自動認識します。配置直後の場合はアプリを再起動してください。
