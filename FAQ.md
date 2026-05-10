# ❓ FAQ — よくある質問とトラブルシューティング

---

## インストール関連

### Q: `pip install -r requirements.txt` が途中で失敗する

**A:** requirements.txt に含まれるオプショナルパッケージ（catboost, lightgbm 等）のビルドに失敗する場合があります。

```powershell
# 必須パッケージのみ先にインストール
pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly nicegui joblib pyarrow openpyxl pyyaml xgboost shap

# オプショナルは個別に（失敗してもOK）
pip install lightgbm
pip install catboost
pip install optuna
```

> 💡 アプリは必須パッケージだけでも起動します。未インストールのエンジンはUIでグレーアウト表示されます。

---

### Q: RDKit がインストールできない

**A:** pip からは `rdkit-pypi` としてインストールできます。

```powershell
pip install rdkit-pypi
```

Conda を使う場合:

```powershell
conda install -c conda-forge rdkit
```

> 💡 Conda の方がバイナリ依存の問題が少なく安定です。

---

### Q: Mordred がインストールできない（Python 3.12+）

**A:** Mordred は Python 3.12 以降で互換性問題が発生する場合があります。

```powershell
# Mordred フォーク版を試す
pip install mordredcommunity
```

Mordred なしでもアプリは動作します（Mordred記述子は利用不可になります）。

---

### Q: PyTorch のインストールが遅い / 失敗する

**A:** CPU版を明示的に指定してください:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

GPU版が不要な場合（記述子計算のみ）、CPU版で十分です。

---

### Q: `fairchem-core` (UMA) がインストールできない

**A:** fairchem-core は PyTorch に依存します。先に PyTorch をインストールしてから実行してください。
一部の環境ではビルドに失敗する場合があります。UMA なしでもアプリは正常に動作します。

---

## xTB 関連

### Q: xTB が「PATH未設定」と表示される

**A:** 以下の方法で PATH を設定してください:

```powershell
# 方法1: バッチファイルを実行（恒久的に設定）
.\tools\add_xtb_to_path.bat

# 方法2: PowerShellで一時的に設定
$env:PATH = "$PWD\tools\xtb-6.7.1\bin;$env:PATH"
xtb --version  # 確認
```

新しいターミナルを開くと method 1 の設定が反映されます。

---

### Q: xTB の計算が遅い / タイムアウトする

**A:** xTB は分子あたり数秒〜数十秒かかります。大きなデータセット（100分子以上）ではかなり時間がかかるため:

- まずは RDKit + GroupContrib のみで試す
- xTB は少量のデータ（10-50分子）で試行
- `backend/chem/xtb_adapter.py` のタイムアウト設定を調整

---

## アプリ起動関連

### Q: `python frontend_nicegui/main.py` でエラーが出る

**A:** よくある原因:

1. **仮想環境が有効化されていない**
   ```powershell
   .\.venv\Scripts\Activate.ps1   # PowerShell
   .\.venv\Scripts\activate.bat   # コマンドプロンプト
   ```

2. **ポートが既に使用中**
   ```powershell
   # 8085番ポートを使用中のプロセスを確認
   netstat -ano | findstr :8085
   # PIDを確認してプロセスを終了
   taskkill /PID <PID> /F
   ```

3. **NiceGUI がインストールされていない**
   ```powershell
   pip install nicegui
   ```

---

### Q: ブラウザが自動で開かない

**A:** 手動で http://localhost:8085 にアクセスしてください。
ファイアウォールがブロックしている場合は `localhost` を許可してください。

---

## データ関連

### Q: CSVの文字化け

**A:** UTF-8 エンコードの CSV を使用してください。Excel で保存する場合は「CSV UTF-8」を選択。
Shift-JIS の CSV は自動判定で読み込まれますが、一部文字化けする場合があります。

---

### Q: SMILES列が認識されない

**A:** データ読み込み後、「列の役割」タブで SMILES 列を手動指定してください。
自動検出は列名に `smiles`, `SMILES`, `smi` 等が含まれる場合のみ動作します。

---

## 解析関連

### Q: 記述子が1つも計算されない

**A:** 以下を確認してください:

1. SMILES列が正しく指定されているか
2. SMILES文字列が有効か（RDKit で parse できるか）
3. 少なくとも1つのエンジンが有効か（エンジン詳細タブで確認）

---

### Q: AutoML が非常に遅い

**A:** 以下の設定で高速化できます:

- 記述子数を減らす（上位30件のみ選択）
- Optuna の試行回数を減らす
- 交差検証の分割数を 3 に設定
- 学習器を Random Forest + LightGBM のみに絞る

---

### Q: SHAP 分析でメモリ不足になる

**A:** データセットが大きい場合（1000行以上）、SHAP のサンプル数を制限してください。
TreeExplainer は通常問題ありませんが、KernelExplainer は計算量が大きいです。

---

## その他

### Q: 開発に参加したい

**A:** [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md) を参照してください。

### Q: テストの実行方法

```powershell
python -m pytest tests/ -v --tb=short
```

### Q: ライセンスは？

**A:** [LICENSE](LICENSE) ファイルを参照してください。
