"""
scripts/audit_ui_features.py

UIの必須機能が main.py から正しく呼ばれているかをチェックする監査スクリプト。
新しいタブ追加・リファクタリングの前後に必ず実行すること。

使い方: python scripts/audit_ui_features.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
MAIN_PY = ROOT / "frontend_nicegui" / "main.py"
DATA_TAB_PY = ROOT / "frontend_nicegui" / "components" / "data_tab.py"

# ─────────────────────────────────────────────────────────────────────────────
# 必須機能チェックリスト
# 各エントリー: (説明, ファイルパス, 検索キーワード)
# ─────────────────────────────────────────────────────────────────────────────

CHECKLIST = [
    # ── main.py で呼ばれるべき機能 ──────────────────────────────────────────
    ("データ設定タブ",       MAIN_PY, "render_data_tab"),
    ("EDAパネル（外側）",   MAIN_PY, "render_eda_panel"),
    ("リーク検出",           MAIN_PY, "render_leakage_check_panel"),
    ("CV設定",               MAIN_PY, "render_cv_config"),
    ("パイプライン設定",     MAIN_PY, "render_pipeline_config"),
    ("結果確認タブ",         MAIN_PY, "render_results_tab"),
    ("逆解析タブ",           MAIN_PY, "render_inverse_panel"),
    ("実験計画タブ",         MAIN_PY, "render_doe_tab"),

    # ── data_tab.py 内のサブタブ ─────────────────────────────────────────────
    ("データ読込タブ",       DATA_TAB_PY, "_render_data_load"),
    ("列の役割タブ",         DATA_TAB_PY, "_render_column_roles"),
    ("SMILES特徴量タブ",     DATA_TAB_PY, "_render_smiles_features"),
    ("EDA内側タブ",          DATA_TAB_PY, "_render_eda"),

    # ── カスタムプラグイン（descriptor_plugins_ui.py） ────────────────────────
    ("カスタムプラグインUI", ROOT / "frontend_nicegui" / "components" / "descriptor_plugins_ui.py",
     "_render_custom_plugins"),
    ("CV設定UI定義",         ROOT / "frontend_nicegui" / "components" / "cv_config_ui.py",
     "render_cv_config"),
    ("DoEバックエンド",      ROOT / "backend" / "doe" / "__init__.py",
     "DoEOptimizer"),
]


# ─────────────────────────────────────────────────────────────────────────────

def check():
    errors = []
    warnings = []
    ok_count = 0

    for desc, filepath, keyword in CHECKLIST:
        if not filepath.exists():
            errors.append(f"[FILE NOT FOUND] {filepath}")
            continue
        text = filepath.read_text(encoding="utf-8")
        if keyword in text:
            print(f"  [OK]  {desc:<25} ({keyword})")
            ok_count += 1
        else:
            errors.append(f"  [NG]  {desc:<25} ({keyword}) -- NOT FOUND in {filepath.name}")

    print()
    print(f"結果: {ok_count}/{len(CHECKLIST)} 機能が確認されました")

    if errors:
        print()
        print("── エラー（機能が消えている可能性） ──────────────────")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print()
        print("全機能が確認されました。安全にリリースできます。")
        sys.exit(0)


if __name__ == "__main__":
    print("=" * 60)
    print("ChemAI UI 機能監査スクリプト")
    print("=" * 60)
    check()
