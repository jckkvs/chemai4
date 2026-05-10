"""
変数中心単調性制約設定パネル
・変数ごとの個別設定
・特徴量セット（RDKit, MolAI等）の一括設定
・モデル非依存の状態管理
"""

from typing import List, Dict
import re
from nicegui import ui
import pandas as pd

@ui.refreshable
def render_monotonicity_config(state: dict):
    """
    単調性制約設定UIをレンダリング。
    
    Args:
        state: 状態管理辞書
    """
    ui.label("📐 単調性制約設定（変数中心）").classes("text-h6 mb-1")
    ui.markdown("モデル選択とは独立して設定します。設定は変数に紐づき、対応モデル実行時に自動適用されます。").classes("text-caption text-grey-6 mb-4")

    # 特徴量の取得
    df = state.get("df_encoded")
    if df is None:
        df = state.get("df")
    
    if df is None or df.empty:
        ui.label("ℹ️ データを読み込むと、特徴量ごとの単調性設定が可能になります。").classes("text-grey-6")
        return

    feature_names = list(df.columns)
    
    # Feature group の自動分類
    feature_groups = {"その他": []}
    prefix_map = {
        "RDKit": r"^(rdkit_|molwt|logp|tpsa|num_)",
        "Mordred": r"^(mordred_|ABC|ATS|BCUT)",
        "MolAI": r"^(molai_|latent_|pca_)",
        "PhysChem": r"^(mw|logp|tpsa|hba|hbd|rot_)"
    }
    
    for feat in feature_names:
        matched = False
        for group, pattern in prefix_map.items():
            if re.match(pattern, feat, re.IGNORECASE):
                feature_groups.setdefault(group, []).append(feat)
                matched = True
                break
        if not matched:
            feature_groups["その他"].append(feat)

    constraints = state.setdefault("feature_constraints", {})

    with ui.row().classes("w-full gap-4"):
        
        # ── 左側：セット一括操作パネル ──
        with ui.card().classes("w-1/4 p-4"):
            ui.label("⚡ 特徴量セット一括設定").classes("text-subtitle1 font-bold mb-2")
            ui.markdown("特定のグループに属する変数すべてに対して、一度に制約を適用します。").classes("text-caption text-grey-6 mb-2")
            
            if not feature_groups:
                ui.label("ℹ️ グループ情報がありません").classes("text-caption text-grey-5")
            else:
                # グループ選択
                group_select = ui.select(
                    options=list(feature_groups.keys()),
                    label="対象グループ",
                    value=list(feature_groups.keys())[0] if feature_groups else None
                ).props("dense")
                
                # 制約タイプ選択
                type_select = ui.select(
                    options={"none": "指定なし", "increasing": "単調増加", "decreasing": "単調減少", "unknown": "単調性不明"},
                    value="none",
                    label="適用する制約"
                ).props("dense")
                
                # 適用ボタン
                ui.button("🚀 グループに適用", on_click=lambda: _apply_to_group(
                    state, group_select.value, type_select.value, feature_groups, feature_names
                ), color="primary").props("dense unelevated").classes("w-full mt-2")
                
            # リセットボタン
            ui.separator().classes("my-2")
            ui.button("🗑️ 全設定リセット", on_click=lambda: _reset_all(state, feature_names), color="negative").props("dense flat").classes("w-full")

        # ── 右側：変数個別設定テーブル ──
        with ui.card().classes("w-3/4 p-4"):
            ui.label("📋 変数別詳細設定").classes("text-subtitle1 font-bold mb-2")
            
            # テーブル行データ生成
            rows = []
            for feat in feature_names:
                cfg = constraints.get(feat, {})
                rows.append({
                    "feature": feat,
                    "direction": cfg.get("direction", "none"),
                })
            
            # テーブル描画
            columns = [
                {"name": "feature", "label": "変数名", "field": "feature", "sortable": True},
                {"name": "direction", "label": "制約", "field": "direction"},
            ]
            
            table = ui.table(columns=columns, rows=rows, row_key="feature").props("dense flat bordered virtual-scroll").classes("h-[400px]")
            
            # セル編集コントロールの埋め込み
            table.add_slot("body-cell-direction", '''
                <q-td :props="props">
                    <q-select
                        dense options-dense borderless emit-value map-options
                        v-model="props.row.direction"
                        :options="[
                            {label: '🚫 なし', value: 'none'},
                            {label: '⬆️ 増加', value: 'increasing'},
                            {label: '⬇️ 減少', value: 'decreasing'},
                            {label: '🔄 不明(単調)', value: 'unknown'}
                        ]"
                        @update:model-value="() => $parent.$emit('update_direction', props.row)"
                    />
                </q-td>
            ''')
            table.on("update_direction", lambda e: _update_constraint(state, e.args["feature"], e.args["direction"]))

    # 設定状況サマリー
    active = [f for f, c in constraints.items() if c.get("direction") != "none"]
    ui.label(f"✅ 設定済み変数: {len(active)}件").classes("text-caption text-positive mt-2")


def _update_constraint(state: dict, feature: str, direction: str):
    """変数単位の制約を更新"""
    state.setdefault("feature_constraints", {})
    state["feature_constraints"][feature] = {
        "direction": direction,
        "strength": 1.0, # デフォルト
        "sigma": 3.0     # デフォルト
    }
    render_monotonicity_config.refresh(state) # 状態変更時にUI更新

def _apply_to_group(state: dict, group_name: str, direction: str, feature_groups: Dict[str, List[str]]):
    """選択されたグループに属する変数すべてに制約を適用"""
    if not group_name: return
    
    targets = feature_groups.get(group_name, [])
    state.setdefault("feature_constraints", {})
    for feat in targets:
        state["feature_constraints"][feat] = {
            "direction": direction,
            "strength": 1.0,
            "sigma": 3.0
        }
    
    ui.notify(f"✅ グループ「{group_name}」の {len(targets)} 変数に「{direction}」を適用しました", type="positive", timeout=3000)
    render_monotonicity_config.refresh(state)

def _reset_all(state: dict):
    """全設定をクリア"""
    state.setdefault("feature_constraints", {}).clear()
    ui.notify("🗑️ 全設定をリセットしました", type="info", timeout=2000)
    render_monotonicity_config.refresh(state)
