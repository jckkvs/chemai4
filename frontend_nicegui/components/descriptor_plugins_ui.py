"""
frontend_nicegui/components/descriptor_plugins_ui.py

SMILES記述子パネル — NiceGUI版

設計思想:
  - ユーザーは「選択」するだけ。計算は自動（バックグラウンド）
  - 記述子テーブルに必要な情報をすべて表示（意味・相関・カーディナリティ・ソース）
  - 目的変数に応じた推薦記述子をデフォルト適用
  - ボタン優先度: 塗り=必須操作、outline=オプション、flat=上級者用
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

from frontend_nicegui.components.descriptor_selector_dialog import (
    open_descriptor_detail_dialog,
    render_selected_descriptors_panel,
    render_descriptor_sets_panel,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# エンジン定義（全エンジンをフラットに定義）
# ═══════════════════════════════════════════════════════════
_ENGINE_INFO: list[dict[str, Any]] = [
    {"cls": "RDKitAdapter", "label": "RDKit 基本記述子", "category": "物理化学",
     "dims": "~200", "speed": "⚡高速", "desc": "MW, LogP, TPSA, HBA/HBD等",
     "manual_url": "https://www.rdkit.org/docs/GettingStartedInPython.html"},
    {"cls": "GroupContribAdapter", "label": "基団寄与法", "category": "物理化学",
     "dims": "~15", "speed": "⚡高速", "desc": "Crippen LogP, MR分解",
     "manual_url": "https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation"},
    {"cls": "SkfpAdapter", "label": "scikit-fingerprints", "category": "フィンガープリント",
     "dims": "~2200", "speed": "⚡高速", "desc": "ECFP, MACCS, Avalon等30+種",
     "manual_url": "https://scikit-fingerprints.readthedocs.io/"},
    {"cls": "MolfeatAdapter", "label": "Molfeat", "category": "フィンガープリント",
     "dims": "可変", "speed": "⚡高速", "desc": "統合FPフレームワーク",
     "manual_url": "https://molfeat.datamol.io/"},
    {"cls": "MordredAdapter", "label": "Mordred", "category": "包括的QSPR",
     "dims": "~1800", "speed": "🟡中速", "desc": "2D/3D記述子を網羅的に計算",
     "manual_url": "https://mordred-descriptor.github.io/documentation/"},
    {"cls": "DescriptaStorusAdapter", "label": "DescriptaStorus", "category": "包括的QSPR",
     "dims": "~200", "speed": "⚡高速", "desc": "Merck開発の高速記述子",
     "manual_url": "https://github.com/bp-kelley/descriptastorus"},
    {"cls": "PaDELAdapter", "label": "PaDEL", "category": "包括的QSPR",
     "dims": "~1800", "speed": "🟡中速", "desc": "PaDEL-Descriptor互換",
     "manual_url": "http://www.yapcwsoft.com/dd/padeldescriptor/"},
    {"cls": "MolAIAdapter", "label": "MolAI (CNN+PCA)", "category": "深層学習",
     "dims": "指定可", "speed": "🟡中速", "desc": "CNN潜在ベクトル→PCA次元圧縮",
     "manual_url": "https://molfeat.datamol.io/featurizers/pretrained"},
    {"cls": "Mol2VecAdapter", "label": "Mol2Vec", "category": "深層学習",
     "dims": "300", "speed": "🟡中速", "desc": "Word2Vec分散表現",
     "manual_url": "https://github.com/samoturk/mol2vec"},
    {"cls": "ChempropAdapter", "label": "Chemprop (D-MPNN)", "category": "深層学習",
     "dims": "可変", "speed": "🔴低速", "desc": "Directed Message Passing GNN",
     "manual_url": "https://chemprop.readthedocs.io/"},
    {"cls": "XTBAdapter", "label": "xTB (GFN2-xTB)", "category": "量子化学",
     "dims": "~20", "speed": "🔴低速", "desc": "HOMO, LUMO, 双極子, 分極率",
     "manual_url": "https://xtb-docs.readthedocs.io/"},
    {"cls": "CosmoAdapter", "label": "COSMO-RS", "category": "量子化学",
     "dims": "~10", "speed": "🔴低速", "desc": "溶媒和自由エネルギー, σプロファイル",
     "manual_url": "https://www.scm.com/addon/cosmo-rs/"},
    {"cls": "UniPkaAdapter", "label": "UniPKa", "category": "量子化学",
     "dims": "~5", "speed": "🟡中速", "desc": "酸解離定数pKa予測",
     "manual_url": "https://github.com/mayrf/pkasolver"},
    {"cls": "UMAAdapter", "label": "UMA (Meta FAIR)", "category": "量子化学",
     "dims": "~7", "speed": "🔴低速", "desc": "DFTレベル分子物性",
     "manual_url": "https://ai.meta.com/research/publications/uma-universal-models-for-atoms/"},
]


# ═══════════════════════════════════════════════════════════
# アダプタ可用性チェック
# ═══════════════════════════════════════════════════════════
def _load_adapters() -> dict:
    """各アダプタをbackend.chemから読み込み、可用性を返す。"""
    adapters: dict = {}
    try:
        mod = importlib.import_module("backend.chem")
    except ImportError:
        return adapters
    for eng in _ENGINE_INFO:
        cls_name = eng["cls"]
        try:
            cls = getattr(mod, cls_name)
            adapters[cls_name] = cls()
        except Exception:
            adapters[cls_name] = None
    return adapters


def _is_available(adapters: dict, cls_name: str) -> bool:
    adp = adapters.get(cls_name)
    if adp is None:
        return False
    try:
        return adp.is_available()
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# メインUI
# ═══════════════════════════════════════════════════════════
def render_descriptor_plugins(state: dict[str, Any]) -> None:
    """SMILES記述子パネルの完全なUI。"""

    # アダプタ読み込み（キャッシュ: None はキャッシュミス扱い）
    if not isinstance(state.get("_chem_adapters"), dict):
        state["_chem_adapters"] = _load_adapters()
    adapters = state["_chem_adapters"]

    n_ok = sum(1 for eng in _ENGINE_INFO if _is_available(adapters, eng["cls"]))
    n_all = len(_ENGINE_INFO)
    has_precalc = state.get("precalc_done") and state.get("precalc_df") is not None
    n_desc = state["precalc_df"].shape[1] if has_precalc else 0

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width"):
        ui.icon("science").classes("text-cyan text-h5")
        ui.label("SMILES 記述子").classes("text-h6")
        ui.badge(f"{n_ok}/{n_all} エンジン利用可", color="cyan")
        if has_precalc:
            ui.badge(f"✅ {n_desc}個計算済み", color="green")
        else:
            ui.badge("⏳ 計算待ち", color="amber")

    # ── 数え上げ記述子の正規化設定 ──
    with ui.row().classes("items-center q-gutter-sm full-width q-mb-sm").style(
        "background: rgba(6,182,212,0.08); border-radius: 8px; padding: 6px 12px;"
    ):
        ui.icon("tune", color="cyan").classes("text-body1")
        ui.label("整数カウント系記述子の正規化:").classes("text-body2")

        # デフォルト値の設定
        if "count_normalization" not in state:
            state["count_normalization"] = "density"

        norm_toggle = ui.toggle(
            {"density": "密度（分子量で正規化）", "raw": "個数（そのまま）"},
            value=state.get("count_normalization", "density"),
        ).props("dense no-caps color=cyan size=sm").tooltip(
            "原子数・環数・官能基数などの整数カウント記述子を\n"
            "分子量で割って『密度』に変換するか、生の個数のまま使うかを選択します。\n"
            "密度モードでは分子サイズの影響を除き、\n"
            "異なるサイズの分子を公平に比較できます。"
        )

        def _on_norm_change(e):
            state["count_normalization"] = e.value
            mode_label = "密度（分子量で正規化）" if e.value == "density" else "個数（そのまま）"
            ui.notify(f"整数カウント系記述子: {mode_label}モードに切り替えました", type="info", timeout=2500)

        norm_toggle.on("update:model-value", _on_norm_change)

        ui.label(
            "fr_*, Num*, *Count 等の整数カウント記述子に適用"
        ).classes("text-caption text-grey")

    # ─────────────────────────────────────────────────────
    # セクション1: 計算状態UI + 手動計算トリガー
    # ─────────────────────────────────────────────────────

    # 手動計算用UIコンテナ (常時存在させるが進捗は非表示)
    progress_container = ui.column().classes("full-width q-mt-sm")
    progress_container.set_visibility(False)

    with progress_container:
        progress_bar = ui.linear_progress(
            value=0, show_value=False, color="cyan",
        ).props("rounded instant-feedback stripe").style("height: 8px;")
        progress_label = ui.label("準備中...").classes("text-caption text-cyan")
        progress_step = ui.label("").classes("text-caption text-grey")

    async def _manual_compute():
        if state.get("df") is None or not state.get("smiles_col"):
            ui.notify("SMILES列を含むデータを先に読み込んでください", type="warning")
            return
        compute_btn.disable()
        compute_btn.text = "計算中..."
        progress_container.set_visibility(True)
        progress_bar.value = 0
        progress_label.text = "分子構造の解析を開始します..."
        progress_step.text = ""

        import importlib
        import pandas as pd
        from nicegui import run

        smiles_list = state["df"][state["smiles_col"]].dropna().tolist()
        n_mols = len(smiles_list)
        target_name = state.get("target_col", "")

        # ── エンジン定義（ユーザー向け表示名） ──
        # Connection Lost 防止のため、エンジンごとに run.io_bound を分割して実行する。
        _MANUAL_ENGINE_STEPS = [
            {
                "label": "基本物理化学記述子（分子量・LogP・TPSA + フィンガープリント）",
                "step_msg": "RDKit で物理化学的特性とフィンガープリントを計算中…",
                "module": "backend.chem.rdkit_adapter",
                "cls": "RDKitAdapter",
                "kwargs": {},
            },
            {
                "label": "熱力学特性（Joback基団寄与法）",
                "step_msg": "沸点・融点・臨界温度などを推定中…",
                "module": "backend.chem.group_contrib_adapter",
                "cls": "GroupContribAdapter",
                "kwargs": {},
            },
            {
                "label": "包括的2D/3D記述子（Mordred全計算）",
                "step_msg": "Mordred で1800種以上の記述子を全計算中（少しお待ちください）…",
                "module": "backend.chem.mordred_adapter",
                "cls": "MordredAdapter",
                "kwargs": {},
            },
            {
                "label": "フィンガープリント（ECFP・MACCS等）",
                "step_msg": "分子フィンガープリントを生成中…",
                "module": "backend.chem.skfp_adapter",
                "cls": "SkfpAdapter",
                "kwargs": {},
            },
            {
                "label": "Merck高速記述子（DescriptaStorus）",
                "step_msg": "Merck 開発の高速記述子セットを計算中…",
                "module": "backend.chem.descriptastorus_adapter",
                "cls": "DescriptaStorusAdapter",
                "kwargs": {},
            },
            {
                "label": "統合フィンガープリント（Molfeat）",
                "step_msg": "Molfeat フィンガープリントを計算中…",
                "module": "backend.chem.molfeat_adapter",
                "cls": "MolfeatAdapter",
                "kwargs": {},
            },
            {
                "label": "分子埋め込み（Mol2Vec）",
                "step_msg": "Word2Vec ベースの分子ベクトルを生成中…",
                "module": "backend.chem.mol2vec_adapter",
                "cls": "Mol2VecAdapter",
                "kwargs": {},
            },
            {
                "label": "PaDEL記述子",
                "step_msg": "PaDEL 記述子を計算中…",
                "module": "backend.chem.padel_adapter",
                "cls": "PaDELAdapter",
                "kwargs": {},
            },
            {
                "label": "MolAI（CNN潜在ベクトル+PCA）",
                "step_msg": "深層学習モデルで分子の特徴を抽出中…",
                "module": "backend.chem.molai_adapter",
                "cls": "MolAIAdapter",
                "kwargs": {"n_components": 6},
            },
            {
                "label": "XTB（GFN2-xTB 量子化学計算）",
                "step_msg": "HOMO・LUMO・双極子モーメントなどを量子化学計算中…",
                "module": "backend.chem.xtb_adapter",
                "cls": "XTBAdapter",
                "kwargs": {},
            },
            {
                "label": "UniPKa（pKa/LogD予測）",
                "step_msg": "pKa・LogD・溶媒和エネルギーを予測中…",
                "module": "backend.chem.unipka_adapter",
                "cls": "UniPkaAdapter",
                "kwargs": {},
            },
            {
                "label": "COSMO-RS（溶媒和自由エネルギー）",
                "step_msg": "溶解度・分配係数を溶媒和モデルで計算中…",
                "module": "backend.chem.cosmo_adapter",
                "cls": "CosmoAdapter",
                "kwargs": {},
            },
            {
                "label": "UMA（Meta FAIR 量子化学）",
                "step_msg": "DFTレベルの高精度エネルギー記述子を計算中…",
                "module": "backend.chem.uma_adapter",
                "cls": "UMAAdapter",
                "kwargs": {},
            },
            {
                "label": "Chemprop（D-MPNN グラフニューラルネット）",
                "step_msg": "グラフニューラルネットで分子グラフを埋め込み中…",
                "module": "backend.chem.chemprop_adapter",
                "cls": "ChempropAdapter",
                "kwargs": {},
            },
        ]

        def _run_one_engine(module_path, class_name, smiles, kwargs):
            """1エンジン分の記述子を計算する（io_bound から呼ぶ）。"""
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                adapter = cls(**kwargs)
                if not adapter.is_available():
                    return None, 0
                result = adapter.compute(smiles)
                df = result.descriptors
                if df is None or df.empty:
                    return None, 0
                return df, df.shape[1]
            except Exception as exc:
                logger.debug(f"{class_name} スキップ: {exc}")
                return None, 0

        try:
            collected_dfs: list[pd.DataFrame] = []
            n_steps = len(_MANUAL_ENGINE_STEPS)
            n_ok = 0

            for i, step in enumerate(_MANUAL_ENGINE_STEPS):
                pct = i / n_steps
                progress_bar.value = pct
                progress_label.text = step["step_msg"]
                progress_step.text = f"エンジン {i + 1} / {n_steps}: {step['label']}"

                # ── ここで run.io_bound を await → イベントループに制御が戻り
                # ── WebSocketハートビートが維持されるため Connection Lost を防止
                df_eng, n_cols = await run.io_bound(
                    _run_one_engine,
                    step["module"], step["cls"], smiles_list, step["kwargs"],
                )
                if df_eng is not None and not df_eng.empty:
                    collected_dfs.append(df_eng.reset_index(drop=True))
                    n_ok += 1

            # 結合
            if collected_dfs:
                df_desc = pd.concat(collected_dfs, axis=1)
                df_desc = df_desc.loc[:, ~df_desc.columns.duplicated()]
                df_desc = df_desc.apply(pd.to_numeric, errors="coerce")
            else:
                df_desc = pd.DataFrame(index=range(n_mols))

            state["precalc_df"] = df_desc
            state["precalc_done"] = True
            n_desc = df_desc.shape[1]

            progress_bar.value = 1.0
            progress_label.text = f"✅ 計算完了！{n_desc}個の記述子が利用可能です"
            progress_step.text = "記述子を選択して機械学習に進みましょう"
            ui.notify(
                f"✅ {n_desc}個の記述子を計算しました",
                type="positive", timeout=5000,
            )

            # 再描画をトリガー
            refresh = state.get("_refresh_tabs")
            if refresh:
                refresh()

        except Exception as e:
            progress_label.text = "⚠️ 計算中にエラーが発生しました"
            progress_step.text = "「再計算」ボタンで再試行できます"
            ui.notify(
                "特徴量の計算中にエラーが発生しました",
                type="warning", timeout=5000,
            )
            logger.warning(f"[ManualCompute] エラー: {e}")
        finally:
            compute_btn.enable()
            compute_btn.text = "再計算" if state.get("precalc_done") else "特徴量を計算する"


    if not has_precalc:
        with ui.card().classes("full-width q-pa-md q-mb-sm").style(
            "border: 1px solid rgba(251,191,36,0.4); background: rgba(50,40,0,0.3); border-radius: 10px;"
        ):
            # ── ワークフローガイド ──
            with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                # ステップ1: 計算（未完了）
                ui.badge("1", color="amber").props("rounded")
                ui.label("記述子計算").classes("text-body2 text-amber text-bold")
                ui.icon("arrow_forward", color="grey").classes("text-caption")
                # ステップ2: 選択（灰色）
                ui.badge("2", color="grey").props("rounded outline")
                ui.label("記述子選択").classes("text-body2 text-grey")
                ui.icon("arrow_forward", color="grey").classes("text-caption")
                # ステップ3: ML（灰色）
                ui.badge("3", color="grey").props("rounded outline")
                ui.label("機械学習").classes("text-body2 text-grey")

            ui.label(
                "SMILES記述子はデータ読み込み時に自動計算されます。"
                "計算が完了すると記述子選択→機械学習に進めます。"
            ).classes("text-body2 text-amber")

    # ── 計算完了後: ワークフローガイド（ステップ2がアクティブ）──
    if has_precalc:
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            # ステップ1: 計算（完了）
            ui.badge("1", color="green").props("rounded")
            ui.label("記述子計算").classes("text-body2 text-green")
            ui.icon("check", color="green").classes("text-body2")
            ui.icon("arrow_forward", color="grey").classes("text-caption")
            # ステップ2: 選択（アクティブ）
            ui.badge("2", color="cyan").props("rounded")
            ui.label("記述子選択").classes("text-body2 text-cyan text-bold")
            ui.icon("arrow_forward", color="grey").classes("text-caption")
            # ステップ3: ML（灰色）
            ui.badge("3", color="grey").props("rounded outline")
            ui.label("機械学習").classes("text-body2 text-grey")
            
    # 再計算/手動計算ボタンの配置
    compute_btn_text = "再計算" if has_precalc else "特徴量を計算する"
    btn_color = "grey" if has_precalc else "amber"

    with ui.row().classes("q-mb-md"):
        compute_btn = ui.button(
            compute_btn_text, on_click=_manual_compute, color=btn_color
        ).props("outline size=sm no-caps")
        compute_btn.tooltip(
            "通常はデータ読み込み時に自動で計算されます。\n"
            "SMILES列を変更した後やエラーが発生した場合に押してください。"
        )

    # ─────────────────────────────────────────────────────
    # 計算完了後のみ表示されるセクション群
    # 「計算→選択→ML」の流れを守る
    # ─────────────────────────────────────────────────────
    if has_precalc:
        # ── デフォルト記述子セットの自動生成 ──
        _ensure_default_sets(state)

        # ── セット管理: コンパクトバー（sticky・6タブより上） ──
        _render_set_management_bar(state)

        # セクション2: 推薦記述子（目的変数ベース）+ 6タブ選択UI
        _render_target_recommendations(state, adapters)

        # セクション3: 計算済み記述子テーブル（メインコンテンツ）
        _render_descriptor_table(state)

        # セクション3.5: 選択済み記述子の一覧確認 + 個別ON/OFF
        render_selected_descriptors_panel(state)

        # ── CTA: 次の行動を明示 ──
        n_active = len(state.get("active_descriptors", []))
        n_selected = len(state.get("selected_descriptors", []))
        n_samples = len(state.get("df", [])) if state.get("df") is not None else 0

        # ⚠️ フォールバックで「全記述子」を使うのを禁止
        # selected_descriptors / active_descriptors が未設定の場合は
        # デフォルトセットから再取得する
        if n_active == 0 and n_selected == 0:
            # デフォルトセットから自動選択
            sets = state.get("descriptor_sets", {})
            for preferred in ["🤖 LLMによる推奨", "🔢 数えあげ系", "📈 相関Top-N", "📊 分散Top-N", "🎯 汎用QSPR", "🧠 MolAI+PCA"]:
                if preferred in sets and sets[preferred].get("descriptors"):
                    state["selected_descriptors"] = list(sets[preferred]["descriptors"])
                    state["current_set_name"] = preferred
                    n_selected = len(state["selected_descriptors"])
                    break

        use_count = n_active if n_active > 0 else n_selected

        # サンプル数 vs 記述子数チェック
        is_overfit_risk = n_samples > 0 and use_count > n_samples
        is_too_many = use_count > 500
        is_unset = use_count == 0

        with ui.card().classes("full-width q-pa-md q-mt-sm").style(
            "border: 2px solid rgba(0,212,255,0.5); border-radius: 12px;"
            "background: linear-gradient(135deg, rgba(0,40,80,0.6), rgba(0,20,60,0.4));"
        ):
            # 警告バナー
            if is_unset:
                with ui.row().classes("items-center q-gutter-xs q-mb-sm"):
                    ui.icon("warning", color="amber").classes("text-h6")
                    ui.label(
                        "記述子セットが選択されていません。上のセット管理バーからセットを選択してください。"
                    ).classes("text-amber text-caption")
            elif is_overfit_risk:
                with ui.row().classes("items-center q-gutter-xs q-mb-sm"):
                    ui.icon("warning", color="amber").classes("text-h6")
                    ui.label(
                        f"⚠️ 記述子数({use_count}) > サンプル数({n_samples})のため過学習リスクがあります。"
                        f"「🎯 汎用QSPR」または「📈 相関Top-N」セットの使用を推奨します。"
                    ).classes("text-amber text-caption")
            elif is_too_many:
                with ui.row().classes("items-center q-gutter-xs q-mb-sm"):
                    ui.icon("info", color="grey").classes("text-h6")
                    ui.label(
                        f"記述子が{use_count}個と多めです。計算時間が長くなる場合があります。"
                    ).classes("text-grey-6 text-caption")

            with ui.row().classes("items-center justify-between full-width"):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.icon(
                        "warning" if (is_overfit_risk or is_unset) else "rocket_launch",
                        color="amber" if (is_overfit_risk or is_unset) else "cyan",
                    ).classes("text-h4")
                    if is_unset:
                        ui.label("セットを選択してください").classes("text-body1 text-bold text-amber")
                    else:
                        ui.label(
                            f"{use_count}個の記述子で解析を実行"
                            + (f"（サンプル数: {n_samples}）" if n_samples > 0 else "")
                        ).classes("text-body1 text-bold")

                ui.button(
                    "解析を開始する",
                    on_click=lambda: (
                        state.get("_run_analysis") and
                        __import__("asyncio").ensure_future(state["_run_analysis"]())
                    ),
                ).props(
                    f"unelevated size=md no-caps color={'grey' if is_unset else 'cyan'}"
                ).classes(
                    "text-bold"
                ).style("font-size: 1.05rem;").set_enabled(not is_unset)


    # ─────────────────────────────────────────────────────
    # セクション4: エンジン詳細（上級者向け折りたたみ）
    # ─────────────────────────────────────────────────────
    with ui.expansion(
        f"⚙️ 計算エンジン詳細（{n_ok}/{n_all} 利用可能）", icon="tune",
    ).classes("full-width q-mt-sm"):
        _render_engine_details(adapters, state)

        # アダプタパラメータの動的UI（上級者向け）
        ui.separator().classes("q-my-sm")
        _render_adapter_params(adapters, state)

    # ─────────────────────────────────────────────────────
    # セクション5: MolAI PCA / 電荷設定 / カスタムプラグイン
    # ─────────────────────────────────────────────────────
    molai_key = "use_molai"
    if state.get(molai_key, False) and _is_available(adapters, "MolAIAdapter"):
        with ui.expansion("📊 MolAI PCA 設定", icon="settings").classes("full-width"):
            _render_molai_pca(state)

    _has_qchem = any(
        state.get(f"use_{e.replace('Adapter','').lower()}", False)
        for e in ["XTBAdapter", "CosmoAdapter", "UniPkaAdapter", "UMAAdapter"]
    )
    if _has_qchem:
        with ui.expansion("🔴 分子電荷・スピン設定（上級者）", icon="bolt").classes("full-width"):
            _render_charge_settings(state)

    with ui.expansion("📁 カスタムプラグイン管理", icon="folder_open").classes("full-width"):
        _render_custom_plugins(state)


# ═══════════════════════════════════════════════════════════
# デフォルト記述子セット自動生成
# ═══════════════════════════════════════════════════════════
def _ensure_default_sets(state: dict) -> None:
    """記述子計算完了後、初回のみデフォルトセットを自動生成する。

    セット:
      1. 🧠 MolAI+PCA — 深層学習ベース特徴量
      2. 🎯 汎用QSPR — 幅広い分子特性をカバーする厳選セット
      3. 📈 相関Top-N — 目的変数との|r|上位（多重共線性除外）

    全セットで n_descriptors < n_samples を保証。
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    if state.get("_default_sets_generated"):
        return  # 既に生成済み

    precalc_df = state.get("precalc_df")
    if precalc_df is None or precalc_df.empty:
        _logger.warning("_ensure_default_sets: precalc_df is None/empty, skipping")
        return

    _logger.info("_ensure_default_sets: 生成開始 (cols=%d, rows=%d)", precalc_df.shape[1], len(precalc_df))

    try:
        import numpy as np

        all_descs = list(precalc_df.columns)
        n_samples = len(precalc_df)

        # セット辞書の初期化
        if "descriptor_sets" not in state:
            state["descriptor_sets"] = {}
        sets = state["descriptor_sets"]

        # ─── セットA: 🧠 MolAI+PCA ───
        dl_prefixes = ("molai_", "cnn_pca_", "chemprop_", "mol2vec_", "uma_", "molfeat_")
        dl_descs = [d for d in all_descs if d.lower().startswith(dl_prefixes)]
        if dl_descs:
            max_dl = min(n_samples // 5, len(dl_descs), 50)
            sets["🧠 MolAI+PCA"] = {
                "engines": ["MolAI"],
                "active": True,
                "descriptors": dl_descs[:max(max_dl, 5)],
            }

        # ─── セットB: 🎯 汎用QSPR ───
        _universal_candidates = [
            # 物理化学
            "MolWt", "MolLogP", "TPSA", "MolMR",
            "HeavyAtomCount", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "FractionCSP3",
            # 電子状態
            "MaxPartialCharge", "MinPartialCharge",
            "MaxAbsPartialCharge", "MinAbsPartialCharge",
            # トポロジー
            "BertzCT", "Chi0v", "Chi1v", "Kappa1", "Kappa2",
            "HallKierAlpha", "BalabanJ",
            # 構造
            "RingCount", "NumAromaticRings",
            "NumSaturatedRings", "NumAliphaticRings",
            "NumAromaticHeterocycles",
            # 溶解度・分配
            "LabuteASA",
            # フラグメント
            "fr_Al_OH", "fr_Ar_OH", "fr_NH2", "fr_ether",
            # XTB（あれば）
            "HomoEnergy", "LumoEnergy", "HomoLumoGap",
            "DipoleMoment", "Polarizability",
            # 基団寄与法（あれば）
            "joback_Tc", "joback_Pc", "joback_Vc",
        ]
        available_set = set(all_descs)
        universal_descs = [d for d in _universal_candidates if d in available_set]

        # 足りない場合: 名前でソートして補充
        if len(universal_descs) < 10:
            remaining = [d for d in sorted(all_descs) if d not in universal_descs]
            universal_descs.extend(remaining[:30])

        max_univ = min(n_samples // 3, len(universal_descs))
        universal_descs = universal_descs[:max(max_univ, 5)]
        sets["🎯 汎用QSPR"] = {
            "engines": [],
            "active": True,
            "descriptors": universal_descs,
        }

        # ─── セットC: 📈 相関Top-N（多重共線性除外）───
        target_col = state.get("target_col")
        df = state.get("df")

        if target_col and df is not None and target_col in df.columns:
            try:
                import pandas as pd
                target_s = df[target_col]
                if pd.api.types.is_numeric_dtype(target_s):
                    aligned = target_s.iloc[:n_samples].reset_index(drop=True)
                    corr_abs = precalc_df.iloc[:len(aligned)].corrwith(
                        aligned, method="pearson"
                    ).abs().dropna()

                    sorted_by_corr = corr_abs.sort_values(ascending=False).index.tolist()

                    max_corr_n = min(n_samples // 5, 50)
                    selected_corr: list[str] = []
                    exclusion_reasons: dict[str, str] = {}
                    top_candidates = sorted_by_corr[:min(200, len(sorted_by_corr))]
                    
                    if len(top_candidates) > 1:
                        inter_corr = precalc_df[top_candidates].corr().abs()

                        for d in top_candidates:
                            if len(selected_corr) >= max_corr_n:
                                exclusion_reasons[d] = f"上限({max_corr_n}個)到達のため除外"
                                continue
                                
                            is_redundant = False
                            for s in selected_corr:
                                if s in inter_corr.columns and d in inter_corr.index:
                                    r_val = inter_corr.loc[d, s]
                                    if r_val > 0.9:
                                        is_redundant = True
                                        exclusion_reasons[d] = f"⚠️ 多重共線性: {s} と強い相関 (|r|={r_val:.2f})"
                                        break
                            if not is_redundant:
                                selected_corr.append(d)

                    if selected_corr:
                        sets["📈 相関Top-N"] = {
                            "engines": [],
                            "active": True,
                            "descriptors": selected_corr,
                            "exclusion_reasons": exclusion_reasons
                        }
            except Exception as e:
                _logger.warning("相関Top-Nセット生成エラー: %s", e)

        # 相関セットが作れなかった場合: 分散ベースにフォールバック
        if "📈 相関Top-N" not in sets:
            try:
                variances = precalc_df.var(numeric_only=True).dropna()
                sorted_var = variances.sort_values(ascending=False).index.tolist()
                max_var_n = min(n_samples // 5, 50)
                sets["📊 分散Top-N"] = {
                    "engines": [],
                    "active": True,
                    "descriptors": sorted_var[:max(max_var_n, 5)],
                }
            except Exception as e:
                _logger.warning("分散Top-Nセット生成エラー: %s", e)

        # ─── セットD: 🤖 LLMによる推奨 ───
        target_col = state.get("target_col")
        if target_col:
            try:
                from backend.chem.recommender import (
                    get_target_recommendation_by_name, get_all_target_recommendations,
                )
                rec = get_target_recommendation_by_name(target_col)
                if rec is None:
                    # 完全一致しない場合は全推奨から該当しそうなものを探す
                    all_recs = get_all_target_recommendations()
                    if all_recs:
                        rec = all_recs[0]  # デフォルトで最初のを使用

                if rec:
                    llm_descs = [d.name for d in rec.descriptors if d.name in available_set]
                    if llm_descs:
                        sets["🤖 LLMによる推奨"] = {
                            "engines": list(set(
                                ["" for d in rec.descriptors if d.library == "RDKit"] +
                                ["MolAI"] if any(d.library == "MolAI" for d in rec.descriptors) else []
                            )),
                            "active": True,
                            "descriptors": llm_descs[:max(min(n_samples // 3, len(llm_descs)), 3)],
                            "summary": rec.summary,
                            "category": rec.category,
                        }
            except Exception as e:
                _logger.warning("LLM推奨セット生成エラー: %s", e)

        # ─── セットE: 🔢 数えあげ系 ───
        _counting_patterns = ("num", "count", "n_", "heavyatom", "ringcount")
        counting_descs = [
            d for d in all_descs
            if any(p in d.lower() for p in _counting_patterns)
        ]
        # 代表的数えあげ記述子を優先
        _priority_counting = [
            "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "NumAromaticRings", "RingCount",
            "NumSaturatedRings", "NumAliphaticRings", "NumHeterocycles",
            "NumAromaticHeterocycles", "FractionCSP3", "BertzCT",
        ]
        priority_set = [d for d in _priority_counting if d in available_set]
        other_counting = [d for d in counting_descs if d not in priority_set]
        counting_descs = priority_set + other_counting

        if counting_descs:
            max_count = min(n_samples // 3, len(counting_descs))
            sets["🔢 数えあげ系"] = {
                "engines": [],
                "active": True,
                "descriptors": counting_descs[:max(max_count, 5)],
            }

        # デフォルトセット: MOLAI+PCA（全記述子は説明変数過多のため非推奨）
        if "デフォルト" in sets:
            del sets["デフォルト"]  # 旧バージョンの残骸を削除

        # 現在のセット名が未設定なら MOLAI+PCA → 汎用QSPR の優先順
        if "current_set_name" not in state or state.get("current_set_name") in ("デフォルト", None, ""):
            if "🧠 MolAI+PCA" in sets:
                state["current_set_name"] = "🧠 MolAI+PCA"
                state["selected_descriptors"] = list(sets["🧠 MolAI+PCA"]["descriptors"])
            elif "🎯 汎用QSPR" in sets:
                state["current_set_name"] = "🎯 汎用QSPR"
                state["selected_descriptors"] = list(sets["🎯 汎用QSPR"]["descriptors"])

        _logger.info(
            "_ensure_default_sets: 生成完了 — %d セット: %s",
            len(sets), list(sets.keys()),
        )
        state["_default_sets_generated"] = True

    except Exception as e:
        _logger.error("_ensure_default_sets 全体エラー: %s", e, exc_info=True)
        state["_default_sets_generated"] = True  # 無限リトライ防止


# ═══════════════════════════════════════════════════════════
# 記述子セット管理 — コンパクト sticky バー
# ═══════════════════════════════════════════════════════════
def _render_set_management_bar(state: dict) -> None:
    """
    記述子セット管理をコンパクトな1行バーとして表示。
    position: sticky で上部に固定し、スクロールしても常に見える。
    """
    import copy

    # セット管理の初期化
    if "descriptor_sets" not in state:
        state["descriptor_sets"] = {
            "デフォルト": {
                "engines": [],
                "active": True,
                "descriptors": None,
            }
        }
    if "current_set_name" not in state:
        state["current_set_name"] = "デフォルト"

    sets = state["descriptor_sets"]
    current = state["current_set_name"]
    precalc_df = state.get("precalc_df")
    total_available = precalc_df.shape[1] if precalc_df is not None else 0

    # 現在のセットの記述子数
    cur_descs = sets.get(current, {}).get("descriptors")
    n_cur = len(cur_descs) if cur_descs else total_available

    with ui.row().classes(
        "items-center full-width q-gutter-sm q-pa-xs q-mb-sm"
    ).style(
        "position: sticky; top: 0; z-index: 100;"
        "background: rgba(10,15,30,0.95); border-bottom: 1px solid rgba(0,212,255,0.3);"
        "border-radius: 8px; min-height: 40px; backdrop-filter: blur(10px);"
    ):
        # ── アイコン + ラベル ──
        ui.icon("layers", color="cyan").classes("text-body1")
        ui.label("セット:").classes("text-caption text-grey")

        # ── セット切替チップボタン ──
        set_names = list(sets.keys())
        for sn in set_names:
            is_active = (sn == current)
            s_descs = sets[sn].get("descriptors")
            s_count = len(s_descs) if s_descs else total_available

            def _switch(name=sn):
                state["current_set_name"] = name
                if sets[name].get("descriptors"):
                    state["active_descriptors"] = list(sets[name]["descriptors"])
                    state["selected_descriptors"] = list(sets[name]["descriptors"])
                ui.notify(f"🔄 「{name}」", type="info")
                if state.get("_refresh_tabs"):
                    state["_refresh_tabs"]()

            btn = ui.button(
                f"{sn} ({s_count})",
                on_click=_switch,
            ).props(
                f"{'unelevated' if is_active else 'outline'} dense size=sm no-caps "
                f"color={'cyan' if is_active else 'grey-6'}"
            ).classes("text-xs")

        # ── 記述子数バッジ ──
        ui.badge(
            f"{n_cur}/{total_available}", color="teal",
        ).props("outline").tooltip("選択中/利用可能な記述子数")

        # ── 保存ボタン ──
        def _save_current():
            active = state.get("active_descriptors", state.get("selected_descriptors", []))
            sets[current]["descriptors"] = list(active)
            ui.notify(f"💾 「{current}」に {len(active)} 記述子を保存", type="positive")
            if state.get("_refresh_tabs"):
                state["_refresh_tabs"]()

        ui.button(icon="save", on_click=_save_current).props(
            "flat round dense size=sm color=cyan"
        ).tooltip("現在の選択をセットに保存")

        # ── 新規セット ──
        def _add_set():
            idx = len(sets) + 1
            name = f"セット{idx}"
            while name in sets:
                idx += 1
                name = f"セット{idx}"
            active = state.get("active_descriptors", state.get("selected_descriptors", []))
            sets[name] = {
                "engines": [],
                "active": True,
                "descriptors": list(active) if active else None,
            }
            state["current_set_name"] = name
            ui.notify(f"➕ 「{name}」を作成", type="positive")
            if state.get("_refresh_tabs"):
                state["_refresh_tabs"]()

        ui.button(icon="add", on_click=_add_set).props(
            "flat round dense size=sm color=green"
        ).tooltip("現在の選択を新しいセットとして保存")

        # ── 複製ボタン ──
        def _dup_current():
            new_name = f"{current}_コピー"
            i = 2
            while new_name in sets:
                new_name = f"{current}_コピー{i}"
                i += 1
            sets[new_name] = copy.deepcopy(sets[current])
            state["current_set_name"] = new_name
            ui.notify(f"📋 「{new_name}」を作成", type="info")
            if state.get("_refresh_tabs"):
                state["_refresh_tabs"]()

        ui.button(icon="content_copy", on_click=_dup_current).props(
            "flat round dense size=sm color=grey"
        ).tooltip("現在のセットを複製")

        # ── 削除ボタン（デフォルト以外） ──
        if current != "デフォルト" and len(sets) > 1:
            def _del_current():
                del sets[current]
                state["current_set_name"] = "デフォルト"
                ui.notify(f"🗑️ 「{current}」を削除", type="info")
                if state.get("_refresh_tabs"):
                    state["_refresh_tabs"]()

            ui.button(icon="delete_outline", on_click=_del_current).props(
                "flat round dense size=sm color=red-4"
            ).tooltip("現在のセットを削除")

        # ── 全セット管理（ダイアログで詳細） ──
        def _open_full_mgmt():
            from frontend_nicegui.components.descriptor_selector_dialog import (
                render_descriptor_sets_panel,
            )
            with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
                "width: 90vw; max-width: 1200px; max-height: 85vh;"
            ):
                ui.label("📊 記述子セット管理（全セット表示）").classes("text-h6 q-mb-sm")
                render_descriptor_sets_panel(state)
                ui.separator().classes("q-my-sm")
                ui.button("閉じる", on_click=dlg.close).props("outline no-caps color=cyan")
            dlg.open()

        ui.button(icon="dashboard", on_click=_open_full_mgmt).props(
            "flat round dense size=sm color=amber"
        ).tooltip("全セットの比較・管理ダイアログ")

        # ── アクティブセット数（2以上の場合） ──
        active_sets = [n for n, info in sets.items() if info.get("active", True)]
        if len(active_sets) > 1:
            ui.badge(
                f"🔬 {len(active_sets)}セット比較", color="amber",
            ).props("outline dense").classes("text-xs")


# ═══════════════════════════════════════════════════════════
# 目的別プリセット定義（Streamlit版と統一）
# ═══════════════════════════════════════════════════════════
_PRESETS = {
    "🧪 基本物性（沸点・密度等）": {
        "engines": ["RDKitAdapter", "GroupContribAdapter"],
        "desc": "MW, LogP, TPSA等の主要物性15記述子を厳選",
        "descriptors": [
            # 分子サイズ
            "MolWt", "HeavyAtomCount", "LabuteASA",
            # 極性・溶解性
            "MolLogP", "TPSA", "MolMR",
            # 水素結合
            "NumHAcceptors", "NumHDonors",
            # トポロジー
            "NumRotatableBonds", "RingCount", "NumAromaticRings",
            "FractionCSP3",
            # 基団寄与法（物性推算）
            "joback_Tb", "joback_Tc", "joback_Hf",
        ],
    },
    "🔑 構造活性相関（FP中心）": {
        "engines": ["RDKitAdapter", "SkfpAdapter"],
        "desc": "ECFP + 主要物性20記述子で活性予測・QSAR",
        "descriptors": [
            # 基本物性（少数）
            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "RingCount", "FractionCSP3",
            # 電子状態
            "MaxPartialCharge", "MinPartialCharge",
            # トポロジー
            "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2",
            "Chi1n", "Chi1v",
            # EState
            "MaxAbsEStateIndex", "MinAbsEStateIndex",
            # QED
            "qed",
        ],
        "include_fp_prefix": ["Morgan_r2_"],  # Morganフィンガープリントも含む
    },
    "📐 網羅的記述子（特徴量選択前提）": {
        "engines": ["RDKitAdapter", "MordredAdapter"],
        "desc": "RDKit+Mordred全記述子→特徴量選択で自動絞り込み",
        # descriptors未指定 → エンジンの全記述子を使用
    },
    "🧠 深層学習表現": {
        "engines": ["MolAIAdapter", "Mol2VecAdapter"],
        "desc": "CNN潜在ベクトル+Word2Vec分散表現",
    },
    "⚛️ 量子化学込み": {
        "engines": ["RDKitAdapter", "XTBAdapter", "CosmoAdapter"],
        "desc": "HOMO/LUMO, 溶媒和エネルギー等を加えた高精度モデル",
        "descriptors": [
            # 基本物性
            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "RingCount",
            # 量子化学
            "HomoEnergy", "LumoEnergy", "HomoLumoGap",
            "DipoleMoment", "Polarizability",
            # xTB
            "xtb_total_energy", "xtb_homo", "xtb_lumo", "xtb_gap",
            "xtb_dipole", "xtb_polarizability",
        ],
    },
    "🚀 フルセット（全エンジン）": {
        "engines": [e["cls"] for e in _ENGINE_INFO],
        "desc": "利用可能な全エンジンを一括ON（時間がかかります）",
    },
}


# ═══════════════════════════════════════════════════════════
# 記述子選択 — 6タブUI
# ═══════════════════════════════════════════════════════════
def _render_target_recommendations(state: dict, adapters: dict) -> None:
    """
    記述子のの選択方法を8つのタブで提供する。
    1. 🎯 プリセットで選ぶ — 目的別ワンクリックプリセット
    2. 🔬 目的変数で選ぶ — recommender.pyの推奨DBから選択
    3. 📈 相関係数で選ぶ — 目的変数との相関上位を選択
    4. 🔢 数えあげ系 — カウント系記述子（Num*, RingCount等）
    5. 🤖 LLMによる推奨 — LLMを活用した記述子推奨
    6. ⚙️ エンジンから選ぶ — ライブラリ単位で選択
    7. 🔍 テキスト検索 — 記述子名の部分一致検索
    8. 📊 分散ベース — 情報量が大きい記述子を選択
    """
    precalc_df = state.get("precalc_df")
    if precalc_df is None:
        return

    all_descs = list(precalc_df.columns)
    n_total = len(all_descs)

    # ── カタログから化学的意味を収集（全タブ共有） ──
    from frontend_nicegui.components.descriptor_catalog import (
        get_catalog as _get_catalog, SUPPORTED_ENGINES as _SUPP_ENGINES,
    )
    _catalog_meanings: dict[str, str] = {}  # name → short説明
    for _ec in _SUPP_ENGINES:
        _cat = _get_catalog(_ec)
        if _cat:
            for _cat_items in _cat.values():
                for _ci in _cat_items:
                    if not _ci["name"].startswith("_"):
                        _catalog_meanings[_ci["name"]] = _ci.get("short", "")

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 12px;"
        "background: rgba(0,20,40,0.25);"
    ):
        n_selected = len(state.get("selected_descriptors", all_descs))
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("playlist_add_check", color="cyan").classes("text-h5")
            ui.label("記述子を選択する").classes("text-h6")
            ui.badge(f"{n_total}個利用可能", color="cyan").props("outline")
            ui.badge(f"{n_selected}個選択中", color="green").props("outline")

        # ── 8タブ ──
        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as tabs:
            tab_preset = ui.tab("preset", label="🎯 プリセット", icon="auto_awesome")
            tab_target = ui.tab("target", label="🔬 目的変数", icon="science")
            tab_corr = ui.tab("corr", label="📈 相関係数", icon="trending_up")
            tab_count = ui.tab("count", label="🔢 数えあげ系", icon="calculate")
            tab_llm = ui.tab("llm", label="🤖 LLMによる推奨", icon="smart_toy")
            tab_engine = ui.tab("engine", label="⚙️ エンジン", icon="settings")
            tab_search = ui.tab("search", label="🔍 検索", icon="search")
            tab_variance = ui.tab("variance", label="📊 分散", icon="bar_chart")

        with ui.tab_panels(tabs, value="preset").classes("full-width"):

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ1: プリセットで選ぶ（NEW）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("preset"):
                ui.label(
                    "解析の目的に合わせてエンジンをまとめてON/OFFできます。"
                    "ワンクリックで記述子セットが切り替わります。"
                ).classes("text-caption text-grey q-mb-sm")

                for preset_name, preset_info in _PRESETS.items():
                    preset_engines = preset_info["engines"]
                    preset_desc = preset_info["desc"]

                    # 全エンジンが利用可能かチェック
                    all_avail = all(
                        _is_available(adapters, e) for e in preset_engines
                    )
                    n_avail = sum(
                        1 for e in preset_engines if _is_available(adapters, e)
                    )
                    # 厳選記述子数（表示用）
                    n_curated = len(preset_info.get("descriptors", []))

                    def _apply_preset(engines=preset_engines, pname=preset_name, pinfo=preset_info):
                        # 全エンジンOFF → 選択エンジンのみON
                        for eng in _ENGINE_INFO:
                            key = f"use_{eng['cls'].replace('Adapter', '').lower()}"
                            state[key] = eng["cls"] in engines

                        # 厳選記述子リストがある場合はそれを優先
                        curated = pinfo.get("descriptors")
                        fp_prefixes = pinfo.get("include_fp_prefix", [])
                        if curated:
                            selected = [d for d in all_descs if d in curated]
                            # FPプレフィックス指定があればそれも追加
                            for pfx in fp_prefixes:
                                selected += [d for d in all_descs if d.startswith(pfx)]
                            state["selected_descriptors"] = selected
                            ui.notify(
                                f"{pname}: {len(selected)}個の厳選記述子を適用",
                                type="positive",
                            )
                            return

                        # 選択エンジンに属する記述子のみを選択
                        engine_cls_set = set(engines)
                        engine_to_cls = {}
                        for eng in _ENGINE_INFO:
                            engine_to_cls[eng["cls"]] = eng["label"]
                        selected = []
                        for d in all_descs:
                            dl = d.lower()
                            matched = False
                            if "RDKitAdapter" in engine_cls_set:
                                if not any(dl.startswith(p) for p in [
                                    "xtb_", "cosmo_", "mu_", "ln_gamma", "joback_",
                                    "pka", "mordred_", "mrd_", "morgan", "maccs",
                                    "avalon", "molfeat_", "mol2vec_", "chemprop_",
                                    "uma_", "padel_", "ds_",
                                ]):
                                    matched = True
                            if "XTBAdapter" in engine_cls_set and (
                                dl.startswith("xtb_") or d in (
                                    "HomoEnergy", "LumoEnergy", "HomoLumoGap",
                                    "DipoleMoment", "Polarizability",
                                )
                            ):
                                matched = True
                            if "CosmoAdapter" in engine_cls_set and (
                                dl.startswith("cosmo_") or dl.startswith("mu_")
                            ):
                                matched = True
                            if "GroupContribAdapter" in engine_cls_set and (
                                dl.startswith("joback_") or d in (
                                    "CohesiveEnergy", "VanDerWaalsVolume",
                                )
                            ):
                                matched = True
                            if "MordredAdapter" in engine_cls_set and (
                                dl.startswith("mordred_") or dl.startswith("mrd_")
                            ):
                                matched = True
                            if "SkfpAdapter" in engine_cls_set and any(
                                dl.startswith(p) for p in ["morgan", "maccs", "avalon"]
                            ):
                                matched = True
                            if "MolfeatAdapter" in engine_cls_set and dl.startswith("molfeat_"):
                                matched = True
                            if "MolAIAdapter" in engine_cls_set and dl.startswith("molai_"):
                                matched = True
                            if "Mol2VecAdapter" in engine_cls_set and dl.startswith("mol2vec_"):
                                matched = True
                            if "ChempropAdapter" in engine_cls_set and dl.startswith("chemprop_"):
                                matched = True
                            if "UniPkaAdapter" in engine_cls_set and (dl.startswith("pka") or d == "pKa_pred"):
                                matched = True
                            if "UMAAdapter" in engine_cls_set and dl.startswith("uma_"):
                                matched = True
                            if "PaDELAdapter" in engine_cls_set and dl.startswith("padel_"):
                                matched = True
                            if "DescriptaStorusAdapter" in engine_cls_set and dl.startswith("ds_"):
                                matched = True
                            if matched:
                                selected.append(d)

                        if not selected:
                            selected = list(all_descs)
                        state["selected_descriptors"] = selected
                        state["active_descriptors"] = selected
                        ui.notify(
                            f"✅ {pname} を適用 ({len(selected)}記述子選択)",
                            type="positive",
                        )

                    # プリセットに含まれる記述子を事前計算
                    _preview_descs = []
                    for d in all_descs:
                        dl = d.lower()
                        _pm = False
                        if "RDKitAdapter" in preset_engines and not any(dl.startswith(p) for p in [
                            "xtb_", "cosmo_", "mu_", "ln_gamma", "joback_",
                            "pka", "mordred_", "mrd_", "morgan", "maccs",
                            "avalon", "molfeat_", "mol2vec_", "chemprop_",
                            "uma_", "padel_", "ds_",
                        ]):
                            _pm = True
                        if "XTBAdapter" in preset_engines and (dl.startswith("xtb_") or d in ("HomoEnergy", "LumoEnergy", "HomoLumoGap", "DipoleMoment", "Polarizability")):
                            _pm = True
                        if "GroupContribAdapter" in preset_engines and (dl.startswith("joback_") or d in ("CohesiveEnergy", "VanDerWaalsVolume")):
                            _pm = True
                        if "MordredAdapter" in preset_engines and (dl.startswith("mordred_") or dl.startswith("mrd_")):
                            _pm = True
                        if "SkfpAdapter" in preset_engines and any(dl.startswith(p) for p in ["morgan", "maccs", "avalon", "rdkitfp_", "atompairfp_", "topologicaltorsionfp_"]):
                            _pm = True
                        if "MolfeatAdapter" in preset_engines and dl.startswith("molfeat_"):
                            _pm = True
                        if "MolAIAdapter" in preset_engines and dl.startswith("molai_"):
                            _pm = True
                        if "Mol2VecAdapter" in preset_engines and dl.startswith("mol2vec_"):
                            _pm = True
                        if "CosmoAdapter" in preset_engines and (dl.startswith("cosmo_") or dl.startswith("mu_")):
                            _pm = True
                        if "ChempropAdapter" in preset_engines and dl.startswith("chemprop_"):
                            _pm = True
                        if "UMAAdapter" in preset_engines and dl.startswith("uma_"):
                            _pm = True
                        if _pm:
                            _preview_descs.append(d)
                    n_preview = len(_preview_descs)

                    # FPグループ化して表示用に整理
                    _pv_non_fp, _pv_fp_groups = _group_fp_descriptors(_preview_descs)
                    _summary_parts = []
                    for d in _pv_non_fp[:8]:
                        _summary_parts.append(d)
                    for gl, bits in sorted(_pv_fp_groups.items(), key=lambda x: -len(x[1])):
                        _summary_parts.append(f"{gl}({len(bits)}bit)")
                    _remaining = n_preview - len(_pv_non_fp[:8]) - sum(len(b) for b in _pv_fp_groups.values())
                    preview_text = ", ".join(_summary_parts[:12])
                    if len(_summary_parts) > 12:
                        preview_text += f" 他{len(_summary_parts)-12}種"

                    # エンジン名のラベルマップ
                    _eng_label = {e["cls"]: e["label"] for e in _ENGINE_INFO}

                    border = "rgba(0,212,255,0.4)" if all_avail else "rgba(255,200,0,0.3)"
                    with ui.card().classes("full-width q-mb-md").style(
                        f"border: 1px solid {border}; border-radius: 8px;"
                        "background: rgba(0,20,40,0.3); padding: 0;"
                    ):
                        # --- P-B案 UI実装 ---
                        with ui.row().classes("items-center full-width justify-between q-pa-sm"):
                            with ui.column().classes("q-gutter-none").style("flex: 1;"):
                                ui.label(preset_name).classes("text-body1 text-bold")
                                ui.label(preset_desc).classes("text-caption text-grey")
                                # エンジンバッジ
                                with ui.row().classes("q-gutter-xs q-mt-xs"):
                                    for pe in preset_engines:
                                        lbl = _eng_label.get(pe, pe.replace("Adapter", ""))
                                        avl = _is_available(adapters, pe)
                                        ui.badge(
                                            lbl, color="teal" if avl else "grey",
                                        ).props("outline dense").style("font-size: 0.78rem;")

                            with ui.row().classes("items-center q-gutter-sm"):
                                # 厳選バッジ（「2/2」等の内部都合を排し、人間がわかる表示へ）
                                display_badge_text = f"{n_curated}個厳選" if n_curated > 0 else f"全{n_preview}個"
                                ui.badge(
                                    display_badge_text,
                                    color="green" if all_avail else "amber",
                                ).props("outline")
                                ui.button(
                                    "適用", on_click=_apply_preset,
                                ).props(
                                    f"{'unelevated' if all_avail else 'outline'}"
                                    " size=md no-caps color=cyan"
                                )
                        
                        # --- アコーディオン展開で透明性確保（研究者・数学者用） ---
                        with ui.expansion("内訳を表示", icon="view_list").classes("full-width bg-transparent"):
                            with ui.row().classes("q-gutter-xs q-pa-sm"):
                                _disp_limit = 50
                                for dp in _pv_non_fp[:_disp_limit]:
                                    _meaning = _catalog_meanings.get(dp, "")
                                    _tt = _meaning if _meaning else "詳細情報なし"
                                    ui.chip(dp, icon="science", color="cyan").props("outline size=sm").tooltip(_tt)
                                
                                for gl, bits in sorted(_pv_fp_groups.items(), key=lambda x: -len(x[1])):
                                    ui.chip(f"{gl}({len(bits)}bit)", icon="fingerprint", color="purple").props("outline size=sm")
                                
                                if len(_pv_non_fp) > _disp_limit:
                                    ui.chip(f"他 {len(_pv_non_fp) - _disp_limit} 個の機能...", color="grey").props("outline size=sm")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ2: 目的変数で選ぶ
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("target"):
                ui.label(
                    "予測したい目的変数を選ぶと、物理化学的な根拠に基づいた"
                    "推奨記述子セットが適用されます（各8個以上）。"
                ).classes("text-caption text-grey q-mb-sm")

                try:
                    from backend.chem.recommender import (
                        get_target_categories,
                        get_targets_by_category,
                    )
                    categories = get_target_categories()
                    for cat_name in categories:
                        cat_recs = get_targets_by_category(cat_name)
                        with ui.expansion(
                            f"■ {cat_name} ({len(cat_recs)}件)",
                        ).classes("full-width q-mb-xs").props("dense"):
                            for rec in cat_recs:
                                def _apply_rec(r=rec):
                                    desc_names = [d.name for d in r.descriptors]
                                    valid = [d for d in desc_names if d in all_descs]
                                    state["active_descriptors"] = valid
                                    state["selected_descriptors"] = valid
                                    state["_applied_recommendation"] = r
                                    ui.notify(
                                        f"{r.target_name} の推奨記述子 {len(valid)}個を適用",
                                        type="positive",
                                    )

                                                                # ── 推奨記述子の表示 ──
                                _valid_descs = [d for d in rec.descriptors if d.name in all_descs]
                                _n_valid = len(_valid_descs)
                                _n_total_rec = len(rec.descriptors)
                                _valid_dn = [d.name for d in _valid_descs]
                                _n_sel = sum(1 for dn in _valid_dn if dn in state.get("selected_descriptors", []))

                                # 全て選択されているかを判定
                                _is_all_on = (_n_sel == _n_valid and _n_valid > 0)

                                with ui.card().classes("full-width q-pa-xs q-ma-xs").style(
                                    "background: rgba(255,255,255,0.03);"
                                ):
                                    # ヘッダー行: チェックボックス + 名前 + バッジ
                                    with ui.row().classes("full-width items-center q-gutter-x-sm"):
                                        ui.checkbox(
                                            rec.target_name,
                                            value=_is_all_on,
                                            on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)
                                        ).props("dense")

                                        ui.badge(
                                            f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",
                                            color="teal" if _n_valid == _n_total_rec else "amber",
                                        ).props("outline dense")

                                        if rec.category:
                                            ui.badge(rec.category, color="blue").props("outline dense")

                                    # サマリー
                                    if rec.summary:
                                        ui.label(rec.summary).classes("text-caption text-grey q-ml-lg")

                                    # 全選択/全解除ボタン
                                    if _valid_descs:
                                        with ui.row().classes("q-gutter-xs q-ml-lg q-mt-xs"):
                                            def _select_all_rec(val, names=_valid_dn, r=rec):
                                                s = set(state.get("selected_descriptors", []))
                                                if val:
                                                    s.update(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}: {len(names)}件を選択", type="positive")
                                                else:
                                                    s -= set(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}: 全解除しました", type="info")

                                            ui.button(
                                                "全選択",
                                                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(True, n, r)
                                            ).props("outline size=xs no-caps color=blue")
                                            ui.button(
                                                "全解除",
                                                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(False, n, r)
                                            ).props("outline size=xs no-caps color=red")

                                    # 個別記述子のチェックボックスリスト
                                    if _valid_descs:
                                        ui.separator().classes("q-my-xs")
                                        ui.label("個別選択:").classes("text-caption text-grey q-ml-lg")

                                        for desc in _valid_descs:
                                            def _toggle_individual(val, dn=desc.name):
                                                s = set(state.get("selected_descriptors", []))
                                                if val:
                                                    s.add(dn)
                                                else:
                                                    s.discard(dn)
                                                state["selected_descriptors"] = list(s)

                                            _is_sel = desc.name in state.get("selected_descriptors", [])
                                            with ui.row().classes(
                                                "items-center q-gutter-xs q-py-xs"
                                            ).style("border-bottom: 1px solid rgba(255,255,255,0.03);"):
                                                ui.checkbox(
                                                    desc.name, value=_is_sel,
                                                    on_change=lambda e, dn=desc.name: _toggle_individual(e.value, dn),
                                                ).props("dense").style("min-width: 180px;")
                                                ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                                if desc.meaning:
                                                    ui.label(desc.meaning).classes("text-caption text-grey").style(
                                                        "font-size: 0.8rem;"
                                                    )

                                def _toggle_rec(val, ds=_valid_dn, r=rec):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.update(ds)
                                        ui.notify(f"{r.target_name}: {len(ds)}個の記述子を追加", type="positive")
                                    else:
                                        s -= set(ds)
                                    state["selected_descriptors"] = list(s)

                                _is_all_on = (_n_sel == _n_valid and _n_valid > 0)
                                
                                def _toggle_rec(val, ds=_valid_dn, r=rec):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.update(ds)
                                        ui.notify(f"{r.target_name}: {len(ds)}個の記述子を追加", type="positive")
                                    else:
                                        s -= set(ds)
                                    state["selected_descriptors"] = list(s)

                                with ui.row().classes("full-width items-center q-gutter-x-sm q-py-xs"):
                                    ui.checkbox(
                                        rec.target_name, 
                                        value=_is_all_on,
                                        on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)
                                    )
                                    ui.badge(
                                        f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",
                                        color="teal" if _n_valid == _n_total_rec else "amber",
                                    ).props("outline")


                except ImportError:
                    ui.label("recommender.py が利用できません").classes("text-warning")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ3: 相関係数で選ぶ
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("corr"):
                target_col = state.get("target_col")
                df = state.get("df")

                if not target_col or df is None or target_col not in df.columns:
                    ui.label(
                        "目的変数が設定されていません。「列の役割」タブで目的変数を指定してください。"
                    ).classes("text-body2 text-amber")
                else:
                    corr_dict: dict[str, float] = {}
                    try:
                        target_s = df[target_col]
                        if pd.api.types.is_numeric_dtype(target_s):
                            aligned = target_s.iloc[:len(precalc_df)]
                            corr_dict = precalc_df.iloc[:len(aligned)].corrwith(
                                aligned.reset_index(drop=True), method="pearson"
                            ).abs().dropna().to_dict()
                    except Exception:
                        pass

                    if not corr_dict:
                        ui.label(
                            "相関係数を計算できませんでした。"
                        ).classes("text-body2 text-amber")
                    else:
                        sorted_descs = sorted(
                            corr_dict.keys(), key=lambda d: corr_dict[d], reverse=True
                        )
                        ui.label(
                            f"目的変数「{target_col}」との|r|で上位を選択できます。"
                        ).classes("text-caption text-grey q-mb-sm")

                        # 相関閾値スライダー
                        with ui.row().classes("items-center q-gutter-sm q-mb-sm full-width"):
                            ui.label("相関閾値:").classes("text-body2")
                            corr_thresh = ui.slider(
                                min=0.0, max=1.0, value=0.1, step=0.05,
                            ).props("label-always").classes("full-width")
                            corr_thresh_label = ui.label("0.10").classes("text-caption text-cyan")

                        def _update_thresh(e):
                            corr_thresh_label.text = f"{e.value:.2f}"
                        corr_thresh.on_value_change(_update_thresh)

                        def _sel_by_thresh():
                            t = corr_thresh.value
                            sel = [d for d in sorted_descs if corr_dict.get(d, 0) >= t]
                            state["selected_descriptors"] = sel
                            state["active_descriptors"] = sel
                            ui.notify(f"|r|≥{t:.2f} の {len(sel)}件を選択", type="positive")

                        with ui.row().classes("q-gutter-sm q-mb-sm"):
                            ui.button(
                                "閾値以上を選択", on_click=_sel_by_thresh,
                            ).props("outline size=sm no-caps color=teal")
                            for n in [10, 20, 30, 50]:
                                def _sel_top(n=n, descs=sorted_descs):
                                    state["selected_descriptors"] = list(descs[:n])
                                    state["active_descriptors"] = list(descs[:n])
                                    ui.notify(f"|r|上位{n}件を選択", type="positive")

                                ui.button(
                                    f"上位{n}件",
                                    on_click=_sel_top,
                                ).props("outline size=sm no-caps color=cyan")

                        # 上位20件のミニテーブル
                        _corr_selected = set(state.get("selected_descriptors", all_descs))
                        for d in sorted_descs[:20]:
                            r_val = corr_dict[d]
                            d_meaning = _catalog_meanings.get(d, "")
                            with ui.row().classes(
                                "items-center full-width q-py-xs q-gutter-xs"
                            ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
                                def _toggle_corr(val, dn=d):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.add(dn)
                                    else:
                                        s.discard(dn)
                                    state["selected_descriptors"] = list(s)

                                ui.checkbox(
                                    d, value=(d in _corr_selected),
                                    on_change=lambda e, dn=d: _toggle_corr(
                                        e.value, dn
                                    ),
                                ).props("dense").style("min-width: 180px;")
                                if d_meaning:
                                    ui.label(d_meaning).classes(
                                        "text-caption text-grey"
                                    ).style("font-size: 0.82rem; min-width: 140px;")
                                ui.linear_progress(
                                    value=int(r_val * 100) / 100, color="cyan",
                                ).style("width: 120px; height: 6px;").props("rounded")
                                ui.label(f"{r_val:.4f}").classes("text-caption text-cyan")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ4: エンジンから選ぶ（カタログ統合・個別選択）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("engine"):
                ui.label(
                    "計算エンジンごとに記述子を化学カテゴリ別に表示。"
                    "各記述子の化学的意味を確認しながら個別にON/OFFできます。"
                ).classes("text-caption text-grey q-mb-sm")

                # ── 記述子をエンジン別に分類 ──
                engine_descs: dict[str, list[str]] = {}
                for d in all_descs:
                    eng = "RDKit"
                    dl = d.lower()
                    if dl.startswith("xtb_") or d in (
                        "HomoEnergy", "LumoEnergy", "HomoLumoGap",
                        "DipoleMoment", "Polarizability", "IonizationPotential",
                        "ElectronAffinity",
                    ):
                        eng = "XTB"
                    elif dl.startswith("cosmo_") or dl.startswith("mu_") or dl.startswith("ln_gamma"):
                        eng = "COSMO-RS"
                    elif dl.startswith("joback_") or d in (
                        "CohesiveEnergy", "CohesiveEnergyDensity", "FreeVolume",
                        "Tg_estimated", "EntanglementMW", "VanDerWaalsVolume",
                    ):
                        eng = "原子団寄与法"
                    elif dl.startswith("pka") or d == "pKa_pred":
                        eng = "UniPKa"
                    elif dl.startswith("mordred_") or dl.startswith("mrd_"):
                        eng = "Mordred"
                    elif dl.startswith("morgan") or dl.startswith("maccs") or dl.startswith("avalon"):
                        eng = "scikit-FP"
                    elif dl.startswith("molfeat_"):
                        eng = "Molfeat"
                    elif dl.startswith("mol2vec_"):
                        eng = "Mol2Vec"
                    elif dl.startswith("chemprop_"):
                        eng = "Chemprop"
                    elif dl.startswith("uma_"):
                        eng = "UMA"
                    elif dl.startswith("padel_"):
                        eng = "PaDEL"
                    elif dl.startswith("ds_"):
                        eng = "DescriptaStorus"
                    engine_descs.setdefault(eng, []).append(d)

                selected = set(state.get("selected_descriptors", all_descs))

                # ── エンジン情報マップ ──
                _eng_info_map = {e["cls"].replace("Adapter", ""): e for e in _ENGINE_INFO}
                _eng_display = {
                    "RDKit": ("🧪", "物理化学的性質", "⚡高速", "~200"),
                    "XTB": ("⚛️", "量子化学", "🔴低速", "~20"),
                    "COSMO-RS": ("🌊", "溶媒和熱力学", "🔴低速", "~10"),
                    "原子団寄与法": ("🔬", "基団寄与法", "⚡高速", "~15"),
                    "Mordred": ("📐", "包括的2D記述子", "🟡中速", "~1800"),
                    "scikit-FP": ("🔑", "分子FP", "⚡高速", "~2200"),
                    "Molfeat": ("🔗", "統合FP", "⚡高速", "可変"),
                    "Mol2Vec": ("🤖", "Word2Vec表現", "🟡中速", "300"),
                    "Chemprop": ("🧠", "GNN表現", "🔴低速", "可変"),
                    "UMA": ("⚛️", "DFT物性", "🔴低速", "~7"),
                    "PaDEL": ("📐", "PaDEL互換", "🟡中速", "~1800"),
                    "DescriptaStorus": ("💊", "Merck記述子", "⚡高速", "~200"),
                    "UniPKa": ("🧪", "酸解離定数", "🟡中速", "~5"),
                }

                for eng_name, descs in sorted(
                    engine_descs.items(), key=lambda x: -len(x[1])
                ):
                    n_sel = sum(1 for d in descs if d in selected)
                    sel_color = "green" if n_sel == len(descs) else "amber" if n_sel > 0 else "grey"
                    icon, purpose, speed, dims = _eng_display.get(
                        eng_name, ("🔧", "", "", "")
                    )

                    with ui.expansion(
                        f"{icon} {eng_name}  【選択: {n_sel} / 全{len(descs)}個】",
                        icon="memory",
                    ).classes("full-width q-mb-xs").props("dense"):
                        # ── エンジン情報ヘッダー ──
                        if purpose:
                            ui.label(
                                f"{purpose} | {speed} | {dims}次元"
                            ).classes("text-caption text-grey").style("font-size: 0.7rem;")

                        # ── エンジン単位の一括操作 ──
                        with ui.row().classes("q-gutter-sm q-mb-sm"):
                            def _sel_all_eng(ds=descs):
                                s = set(state.get("selected_descriptors", []))
                                s.update(ds)
                                state["selected_descriptors"] = list(s)
                                ui.notify(f"{len(ds)}個追加", type="positive")

                            def _desel_all_eng(ds=descs):
                                s = set(state.get("selected_descriptors", []))
                                s -= set(ds)
                                state["selected_descriptors"] = list(s)
                                ui.notify(f"{len(ds)}個解除", type="info")

                            ui.button("全選択", on_click=_sel_all_eng).props(
                                "outline size=xs no-caps color=cyan"
                            )
                            ui.button("全解除", on_click=_desel_all_eng).props(
                                "flat size=xs no-caps color=grey"
                            )
                            ui.badge(
                                f"選択中 {n_sel} / 全{len(descs)}",
                                color=sel_color
                            ).props("outline")

                        # ── カタログベースのグループ分け ──
                        # カタログがあるエンジンはカタログのカテゴリを使用
                        catalog = _get_catalog(eng_name)
                        if catalog:
                            # カタログのカテゴリ→記述子名のマッピングを構築
                            cat_to_descs: dict[str, list[tuple[str, str]]] = {}
                            cataloged_names: set[str] = set()
                            for cat_name, cat_items in catalog.items():
                                for item in cat_items:
                                    if item["name"].startswith("_"):
                                        continue
                                    if item["name"] in set(descs):
                                        cat_to_descs.setdefault(cat_name, []).append(
                                            (item["name"], item.get("short", ""))
                                        )
                                        cataloged_names.add(item["name"])
                            # カタログにない記述子は「その他」に
                            uncataloged = [d for d in descs if d not in cataloged_names]
                            if uncataloged:
                                cat_to_descs["📋 その他（カタログ未登録）"] = [
                                    (d, _catalog_meanings.get(d, "")) for d in uncataloged
                                ]
                        else:
                            # カタログがないエンジン→名前推定でグループ分け
                            groups = _group_descriptors_by_subcategory(eng_name, descs)
                            cat_to_descs = {}
                            for gname, gdescs in groups.items():
                                cat_to_descs[gname] = [
                                    (d, _catalog_meanings.get(d, "")) for d in gdescs
                                ]

                        # ── グループ別展開+個別チェックボックス+化学的意味 ──
                        for group_name, group_items in cat_to_descs.items():
                            g_names = [t[0] for t in group_items]
                            g_sel = sum(1 for d in g_names if d in selected)
                            g_color = "green" if g_sel == len(g_names) else "amber" if g_sel > 0 else "grey"

                            with ui.expansion(
                                f"  {group_name}  【選択: {g_sel} / 全{len(g_names)}個】",
                            ).classes("full-width q-mb-xs").props("dense"):
                                # グループ単位の一括操作
                                with ui.row().classes("q-gutter-xs q-mb-xs"):
                                    def _sg(ds=g_names):
                                        s = set(state.get("selected_descriptors", []))
                                        s.update(ds)
                                        state["selected_descriptors"] = list(s)

                                    def _dg(ds=g_names):
                                        s = set(state.get("selected_descriptors", []))
                                        s -= set(ds)
                                        state["selected_descriptors"] = list(s)

                                    ui.button("✓全", on_click=_sg).props(
                                        "flat size=xs no-caps color=cyan dense"
                                    )
                                    ui.button("✕全", on_click=_dg).props(
                                        "flat size=xs no-caps color=grey dense"
                                    )
                                    ui.badge(
                                        f"{g_sel}/{len(g_names)}", color=g_color,
                                    ).props("outline dense").classes("text-xs")

                                # 個別チェックボックス+化学的意味
                                for desc_name, desc_short in group_items:
                                    is_on = desc_name in selected

                                    def _toggle(val, dn=desc_name):
                                        s = set(state.get("selected_descriptors", []))
                                        if val:
                                            s.add(dn)
                                        else:
                                            s.discard(dn)
                                        state["selected_descriptors"] = list(s)

                                    with ui.row().classes(
                                        "items-center q-gutter-xs"
                                    ).style("min-height: 26px;"):
                                        ui.checkbox(
                                            desc_name,
                                            value=is_on,
                                            on_change=lambda e, dn=desc_name: _toggle(
                                                e.value, dn
                                            ),
                                        ).props("dense").style("min-width: 200px;")
                                        if desc_short:
                                            ui.label(desc_short).classes(
                                                "text-caption text-grey"
                                            ).style("font-size: 0.7rem;")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ5: テキスト検索（NEW）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("search"):
                ui.label(
                    "記述子名を部分一致で検索し、まとめて選択/解除できます。"
                ).classes("text-caption text-grey q-mb-sm")

                search_input = ui.input(
                    "記述子名を検索", placeholder="例: LogP, morgan, TPSA...",
                ).props("outlined dense clearable").classes("full-width q-mb-sm")

                search_results = ui.column().classes("full-width")

                def _do_search():
                    search_results.clear()
                    query = (search_input.value or "").strip().lower()
                    if not query:
                        with search_results:
                            ui.label("検索語を入力してください").classes("text-grey text-caption")
                        return

                    matches = [d for d in all_descs if query in d.lower()]
                    cur_sel = set(state.get("selected_descriptors", []))

                    with search_results:
                        if not matches:
                            ui.label(f"「{query}」に一致する記述子はありません").classes("text-amber")
                            return

                        ui.label(f"{len(matches)}件ヒット").classes("text-caption text-cyan q-mb-xs")

                        with ui.row().classes("q-gutter-sm q-mb-sm"):
                            def _add_all_search(ms=matches):
                                s = set(state.get("selected_descriptors", []))
                                s.update(ms)
                                state["selected_descriptors"] = list(s)
                                ui.notify(f"{len(ms)}件を選択に追加", type="positive")

                            def _remove_all_search(ms=matches):
                                s = set(state.get("selected_descriptors", []))
                                s -= set(ms)
                                state["selected_descriptors"] = list(s)
                                ui.notify(f"{len(ms)}件を選択から除外", type="info")

                            ui.button("全部追加", on_click=_add_all_search).props(
                                "outline size=sm no-caps color=cyan"
                            )
                            ui.button("全部除外", on_click=_remove_all_search).props(
                                "outline size=sm no-caps color=grey"
                            )

                        for m in matches[:50]:
                            in_sel = m in cur_sel
                            with ui.row().classes(
                                "items-center q-py-xs q-gutter-xs"
                            ).style(
                                "border-bottom: 1px solid rgba(255,255,255,0.05);"
                            ):
                                def _toggle_search(val, dn=m):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.add(dn)
                                    else:
                                        s.discard(dn)
                                    state["selected_descriptors"] = list(s)

                                ui.checkbox(
                                    m, value=in_sel,
                                    on_change=lambda e, dn=m: _toggle_search(
                                        e.value, dn
                                    ),
                                ).props("dense").style("min-width: 200px;")
                                _m_meaning = _catalog_meanings.get(m, "")
                                if _m_meaning:
                                    ui.label(_m_meaning).classes(
                                        "text-caption text-grey"
                                    ).style("font-size: 0.82rem;")

                        if len(matches) > 50:
                            ui.label(f"...他 {len(matches) - 50}件").classes(
                                "text-caption text-grey"
                            )

                search_input.on("keyup.enter", lambda: _do_search())
                ui.button("検索", on_click=_do_search).props(
                    "outline size=sm no-caps color=cyan"
                )

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ6: 分散ベース（NEW）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("variance"):
                ui.label(
                    "分散（情報量）が大きい記述子を優先選択します。"
                    "定数列や低情報量の記述子を自動除外できます。"
                ).classes("text-caption text-grey q-mb-sm")

                try:
                    variances = precalc_df.var(numeric_only=True).dropna()
                    # 正規化（0-1スケール）
                    v_max = variances.max()
                    if v_max > 0:
                        norm_var = (variances / v_max).to_dict()
                    else:
                        norm_var = variances.to_dict()

                    sorted_by_var = sorted(
                        norm_var.keys(), key=lambda d: norm_var[d], reverse=True
                    )

                    # 定数列除外
                    n_const = sum(1 for v in variances if v == 0)
                    if n_const > 0:
                        ui.label(
                            f"⚠️ {n_const}個の定数列（分散=0）が検出されました"
                        ).classes("text-amber text-caption q-mb-xs")

                    def _remove_const():
                        const_cols = [c for c, v in variances.items() if v == 0]
                        s = set(state.get("selected_descriptors", all_descs))
                        s -= set(const_cols)
                        state["selected_descriptors"] = list(s)
                        ui.notify(f"定数列 {len(const_cols)}件を除外", type="info")

                    with ui.row().classes("q-gutter-sm q-mb-sm"):
                        if n_const > 0:
                            ui.button("定数列を除外", on_click=_remove_const).props(
                                "outline size=sm no-caps color=amber"
                            )
                        for n in [10, 20, 50, 100]:
                            if n <= len(sorted_by_var):
                                def _sel_var_top(n=n, descs=sorted_by_var):
                                    state["selected_descriptors"] = list(descs[:n])
                                    state["active_descriptors"] = list(descs[:n])
                                    ui.notify(f"分散上位{n}件を選択", type="positive")

                                ui.button(
                                    f"上位{n}件", on_click=_sel_var_top,
                                ).props("outline size=sm no-caps color=teal")

                    # 分散ランキング表示
                    ui.label("分散ランキング（上位20件）").classes("text-subtitle2 q-mt-sm")
                    _var_selected = set(state.get("selected_descriptors", all_descs))
                    for d in sorted_by_var[:20]:
                        nv = norm_var[d]
                        raw_v = variances.get(d, 0)
                        with ui.row().classes(
                            "items-center full-width q-py-xs q-gutter-xs"
                        ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
                            def _toggle_var(val, dn=d):
                                s = set(state.get("selected_descriptors", []))
                                if val:
                                    s.add(dn)
                                else:
                                    s.discard(dn)
                                state["selected_descriptors"] = list(s)

                            ui.checkbox(
                                d, value=(d in _var_selected),
                                on_change=lambda e, dn=d: _toggle_var(e.value, dn),
                            ).props("dense").style("min-width: 180px;")
                            _v_meaning = _catalog_meanings.get(d, "")
                            if _v_meaning:
                                ui.label(_v_meaning).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.82rem; min-width: 140px;")
                            ui.linear_progress(
                                value=nv, color="teal",
                            ).style("width: 120px; height: 6px;").props("rounded")
                            ui.label(f"{raw_v:.4g}").classes("text-caption text-teal")

                except Exception as ex:
                    ui.label(f"分散計算エラー: {ex}").classes("text-red")




            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ4: 数えあげ系 — カウント系記述子
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("count"):
                ui.label(
                    "分子内の原子数、環数、官能基数など「数えあげ系」の記述子を選択します。"
                    "NumHDonors、RingCount、FractionCSP3 などが含まれます。"
                ).classes("text-caption text-grey q-mb-sm")

                # 数えあげ系記述子の特定
                _counting_patterns = ("num", "count", "n_", "heavyatom", "ringcount")
                counting_descs = [
                    d for d in all_descs
                    if any(p in d.lower() for p in _counting_patterns)
                ]

                # 代表的数えあげ記述子を優先表示
                _priority_counting = [
                    "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors",
                    "NumRotatableBonds", "NumAromaticRings", "RingCount",
                    "NumSaturatedRings", "NumAliphaticRings", "NumHeterocycles",
                    "NumAromaticHeterocycles", "FractionCSP3", "BertzCT",
                ]
                priority_set = [d for d in _priority_counting if d in counting_descs]
                other_counting = [d for d in counting_descs if d not in priority_set]
                sorted_counting = priority_set + sorted(other_counting)

                if not sorted_counting:
                    ui.label("数えあげ系記述子が見つかりませんでした。").classes("text-grey")
                else:
                    # 全選択/全解除ボタン
                    with ui.row().classes("q-gutter-sm q-mb-sm"):
                        def _select_all_count():
                            s = set(state.get("selected_descriptors", [])) | set(sorted_counting)
                            state["selected_descriptors"] = list(s)
                            ui.notify(f"数えあげ系 {len(sorted_counting)}件を選択", type="positive")

                        def _deselect_all_count():
                            s = set(state.get("selected_descriptors", [])) - set(sorted_counting)
                            state["selected_descriptors"] = list(s)
                            ui.notify(f"数えあげ系を全解除", type="info")

                        ui.button("全選択", on_click=_select_all_count).props(
                            "outline size=sm no-caps color=blue"
                        )
                        ui.button("全解除", on_click=_deselect_all_count).props(
                            "outline size=sm no-caps color=red"
                        )
                        ui.label(f"計 {len(sorted_counting)} 件").classes("text-caption text-grey q-ml-sm")

                    # 個別選択チェックボックス
                    _count_selected = set(state.get("selected_descriptors", []))
                    for d in sorted_counting:
                        _meaning = _catalog_meanings.get(d, "")
                        with ui.row().classes(
                            "items-center full-width q-py-xs q-gutter-xs"
                        ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
                            def _toggle_count(val, dn=d):
                                s = set(state.get("selected_descriptors", []))
                                if val:
                                    s.add(dn)
                                else:
                                    s.discard(dn)
                                state["selected_descriptors"] = list(s)

                            ui.checkbox(
                                d, value=(d in _count_selected),
                                on_change=lambda e, dn=d: _toggle_count(e.value, dn),
                            ).props("dense").style("min-width: 200px;")
                            if _meaning:
                                ui.label(_meaning).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.82rem;")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ5: LLMによる推奨 — LLM推奨記述子
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("llm"):
                ui.label(
                    "LLM（大規模言語モデル）を活用して、目的変数に適した記述子を推奨します。"
                    "化学的知見に基づいた推奨を行います。"
                ).classes("text-caption text-grey q-mb-sm")

                target_col = state.get("target_col")
                if not target_col:
                    ui.label(
                        "⚠️ 目的変数が設定されていません。「列の役割」タブで目的変数を指定してください。"
                    ).classes("text-amber text-caption q-mb-sm")

                # LLM推奨の取得
                try:
                    from backend.chem.recommender import (
                        get_target_recommendation_by_name, get_all_target_recommendations,
                        get_target_names, get_target_categories,
                    )

                    # 利用可能な記述子セットを取得
                    precalc_df = state.get("precalc_df")
                    available_set = set(precalc_df.columns) if precalc_df is not None and not precalc_df.empty else set()

                    # 目的変数に基づく推奨
                    rec = None
                    if target_col:
                        rec = get_target_recommendation_by_name(target_col)

                    # 推奨が見つからない場合は全推奨から選択可能にする
                    all_recs = get_all_target_recommendations()
                    target_names = get_target_names()
                    target_categories = get_target_categories()

                    if not rec and target_col:
                        # 目的変数名に部分一致する推奨を探す
                        for r in all_recs:
                            if target_col.lower() in r.target_name.lower() or                                r.target_name.lower() in target_col.lower():
                                rec = r
                                break

                    # カテゴリ別に推奨を表示
                    if target_categories:
                        ui.label("物性カテゴリ別推奨記述子").classes("text-subtitle2 q-mt-sm q-mb-xs")

                        for cat in target_categories:
                            cat_recs = [r for r in all_recs if r.category == cat]
                            if not cat_recs:
                                continue

                            with ui.expansion(f"{cat} ({len(cat_recs)}件)", icon="category").classes("full-width q-mb-xs"):
                                for r in cat_recs:
                                    with ui.card().classes("full-width q-pa-xs q-ma-xs").style(
                                        "background: rgba(255,255,255,0.05);"
                                    ):
                                        with ui.row().classes("items-center q-gutter-sm"):
                                            ui.label(f"📌 {r.target_name}").classes("text-bold")
                                            if r.category:
                                                ui.badge(r.category, color="blue").props("outline")

                                        ui.label(r.summary).classes("text-caption text-grey q-mt-xs")

                                        # 推奨記述子の表示
                                        _rec_descs = [d for d in r.descriptors if d.name in available_set]
                                        if _rec_descs:
                                            ui.label("推奨記述子:").classes("text-caption text-bold q-mt-xs")

                                            # 全選択/全解除ボタン
                                            _rec_names = [d.name for d in _rec_descs]
                                            with ui.row().classes("q-gutter-xs q-mb-xs"):
                                                def _select_rec_descs(val, names=_rec_names):
                                                    s = set(state.get("selected_descriptors", []))
                                                    if val:
                                                        s.update(names)
                                                    else:
                                                        s -= set(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}の推奨 {len(names)}件を{'選択' if val else '解除'}", type="positive")

                                                ui.button(
                                                    "全選択",
                                                    on_click=lambda e, n=_rec_names: _select_rec_descs(True, n)
                                                ).props("outline size=xs no-caps color=blue")
                                                ui.button(
                                                    "全解除",
                                                    on_click=lambda e, n=_rec_names: _select_rec_descs(False, n)
                                                ).props("outline size=xs no-caps color=red")

                                            for desc in _rec_descs:
                                                def _toggle_rec(val, dn=desc.name):
                                                    s = set(state.get("selected_descriptors", []))
                                                    if val:
                                                        s.add(dn)
                                                    else:
                                                        s.discard(dn)
                                                    state["selected_descriptors"] = list(s)

                                                _is_sel = desc.name in state.get("selected_descriptors", [])
                                                with ui.row().classes(
                                                    "items-center q-gutter-xs"
                                                ).style(
                                                    "border-bottom: 1px solid rgba(255,255,255,0.03);"
                                                ):
                                                    ui.checkbox(
                                                        desc.name, value=_is_sel,
                                                        on_change=lambda e, dn=desc.name: _toggle_rec(e.value, dn),
                                                    ).props("dense").style("min-width: 180px;")
                                                    ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                                    if desc.meaning:
                                                        ui.label(desc.meaning).classes("text-caption text-grey").style(
                                                            "font-size: 0.8rem;"
                                                        )

                                        # 現在の目的変数に一致する場合はハイライト
                                        if rec and rec.target_name == r.target_name:
                                            ui.label("⭐ 現在の目的変数に一致").classes("text-caption text-cyan q-mt-xs")

                    # 現在の目的変数の推奨をハイライト表示
                    if rec:
                        ui.separator().classes("q-my-sm")
                        with ui.card().classes("full-width q-pa-sm").style(
                            "background: rgba(0,200,255,0.1); border: 1px solid rgba(0,200,255,0.3);"
                        ):
                            ui.label(f"⭐ 現在の目的変数「{target_col}」の推奨").classes("text-bold text-cyan")
                            ui.label(rec.summary).classes("text-caption q-mt-xs")

                            _rec_descs = [d for d in rec.descriptors if d.name in available_set]
                            if _rec_descs:
                                _rec_names = [d.name for d in _rec_descs]

                                with ui.row().classes("q-gutter-sm q-mt-sm"):
                                    def _select_all_llm():
                                        s = set(state.get("selected_descriptors", [])) | set(_rec_names)
                                        state["selected_descriptors"] = list(s)
                                        ui.notify(f"LLM推奨 {len(_rec_names)}件を選択", type="positive")

                                    def _deselect_all_llm():
                                        s = set(state.get("selected_descriptors", [])) - set(_rec_names)
                                        state["selected_descriptors"] = list(s)
                                        ui.notify("LLM推奨を全解除", type="info")

                                    ui.button("全選択", on_click=_select_all_llm).props(
                                        "outline size=sm no-caps color=cyan"
                                    )
                                    ui.button("全解除", on_click=_deselect_all_llm).props(
                                        "outline size=sm no-caps color=red"
                                    )

                                for desc in _rec_descs:
                                    def _toggle_llm(val, dn=desc.name):
                                        s = set(state.get("selected_descriptors", []))
                                        if val:
                                            s.add(dn)
                                        else:
                                            s.discard(dn)
                                        state["selected_descriptors"] = list(s)

                                    _is_sel = desc.name in state.get("selected_descriptors", [])
                                    with ui.row().classes("items-center q-gutter-xs q-py-xs").style(
                                        "border-bottom: 1px solid rgba(255,255,255,0.05);"
                                    ):
                                        ui.checkbox(
                                            desc.name, value=_is_sel,
                                            on_change=lambda e, dn=desc.name: _toggle_llm(e.value, dn),
                                        ).props("dense").style("min-width: 180px;")
                                        ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                        if desc.meaning:
                                            ui.label(desc.meaning).classes("text-caption text-grey").style(
                                                "font-size: 0.8rem;"
                                            )

                except Exception as e:
                    ui.label(f"LLM推奨の取得エラー: {e}").classes("text-red")

# ═══════════════════════════════════════════════════════════
# サブカテゴリ分類ルール
# ═══════════════════════════════════════════════════════════
def _group_descriptors_by_subcategory(engine: str, descs: list[str]) -> dict[str, list[str]]:
    """エンジン名と記述子リストからサブカテゴリに分類する。"""
    if engine == "RDKit":
        return _group_rdkit(descs)
    elif engine == "XTB":
        return _group_xtb(descs)
    elif engine == "COSMO-RS":
        return _group_cosmo(descs)
    elif engine == "原子団寄与法":
        return _group_joback(descs)
    elif engine == "scikit-FP":
        return _group_skfp(descs)
    elif engine == "Mordred":
        return _group_mordred(descs)
    else:
        # デフォルト: 50件以下ならそのまま、多ければアルファベット順で分割
        if len(descs) <= 50:
            return {"全記述子": descs}
        groups: dict[str, list[str]] = {}
        for d in sorted(descs):
            prefix = d[0].upper() if d else "?"
            groups.setdefault(f"{prefix}〜", []).append(d)
        return groups


def _group_rdkit(descs: list[str]) -> dict[str, list[str]]:
    """RDKit記述子をサブカテゴリに分類（2000+個を整理）。"""
    groups: dict[str, list[str]] = {
        "📊 EState指標": [],
        "🔗 トポロジカル指標": [],
        "⚗️ 物理化学（LogP/MR/TPSA等）": [],
        "📐 ＭＯＥ型記述子": [],
        "🧩 フラグメント・官能基カウント": [],
        "⚛️ 原子数・元素カウント": [],
        "💍 環構造記述子": [],
        "🔗 結合・接続性": [],
        "⚖️ 分子量関連": [],
        "📏 分子面積・体積": [],
        "📈 グラフ・行列指標": [],
        "🔢 カッパ形状指標": [],
        "🔬 その他記述子": [],
        "🔑 フィンガープリント": [],
    }
    for d in descs:
        dl = d.lower()
        if "estate" in dl:
            groups["📊 EState指標"].append(d)
        elif any(dl.startswith(p) for p in ["chi", "hall", "kier"]):
            groups["🔗 トポロジカル指標"].append(d)
        elif any(k in dl for k in ["logp", "mollogp", "crippen", "wildman", "tpsa", "labute"]):
            groups["⚗️ 物理化学（LogP/MR/TPSA等）"].append(d)
        elif dl.startswith("slogp_") or dl.startswith("smr_") or dl.startswith("peoe_"):
            groups["📐 ＭＯＥ型記述子"].append(d)
        elif dl.startswith("fr_") or dl.startswith("numh"):
            groups["🧩 フラグメント・官能基カウント"].append(d)
        elif any(dl.startswith(p) for p in ["numatom", "numheavy", "numhetero", "num"]) and "ring" not in dl:
            groups["⚛️ 原子数・元素カウント"].append(d)
        elif "ring" in dl or "aromatic" in dl.replace("numaromaticrings", ""):
            groups["💍 環構造記述子"].append(d)
        elif any(k in dl for k in ["bond", "rotatable", "hba", "hbd", "connect"]):
            groups["🔗 結合・接続性"].append(d)
        elif any(k in dl for k in ["molwt", "exactmolwt", "heavyatommolwt", "molmr"]):
            groups["⚖️ 分子量関連"].append(d)
        elif any(k in dl for k in ["molarea", "mcgowan", "volume", "vsa"]):
            groups["📏 分子面積・体積"].append(d)
        elif any(k in dl for k in ["balaban", "bertz", "ipc", "graph"]):
            groups["📈 グラフ・行列指標"].append(d)
        elif dl.startswith("kappa") or "kappa" in dl:
            groups["🔢 カッパ形状指標"].append(d)
        elif any(dl.startswith(p) for p in ["morgan", "maccs", "avalon", "ecfp", "rdk_fp"]):
            groups["🔑 フィンガープリント"].append(d)
        else:
            groups["🔬 その他記述子"].append(d)

    # 空グループを削除
    return {k: v for k, v in groups.items() if v}


def _group_xtb(descs: list[str]) -> dict[str, list[str]]:
    """XTB記述子の分類。"""
    groups: dict[str, list[str]] = {
        "⚡ 軌道エネルギー": [],
        "🧲 電気的性質": [],
        "🌡️ 熱力学量": [],
        "📐 構造パラメータ": [],
    }
    for d in descs:
        dl = d.lower()
        if any(k in dl for k in ["homo", "lumo", "gap", "orbital"]):
            groups["⚡ 軌道エネルギー"].append(d)
        elif any(k in dl for k in ["dipole", "polariz", "charge", "electro"]):
            groups["🧲 電気的性質"].append(d)
        elif any(k in dl for k in ["energy", "enthalpy", "entropy", "heat", "gibbs"]):
            groups["🌡️ 熱力学量"].append(d)
        else:
            groups["📐 構造パラメータ"].append(d)
    return {k: v for k, v in groups.items() if v}


def _group_cosmo(descs: list[str]) -> dict[str, list[str]]:
    """COSMO-RS記述子の分類。"""
    groups: dict[str, list[str]] = {
        "🧪 化学ポテンシャル": [],
        "📊 σプロファイル": [],
        "🌊 溶媒和": [],
    }
    for d in descs:
        dl = d.lower()
        if any(k in dl for k in ["mu_", "chemical_potential"]):
            groups["🧪 化学ポテンシャル"].append(d)
        elif any(k in dl for k in ["sigma", "profile", "moment"]):
            groups["📊 σプロファイル"].append(d)
        else:
            groups["🌊 溶媒和"].append(d)
    return {k: v for k, v in groups.items() if v}


def _group_joback(descs: list[str]) -> dict[str, list[str]]:
    """Joback原子団寄与法記述子の分類。"""
    groups: dict[str, list[str]] = {
        "🌡️ 熱物性（Tb/Tm/Tc等）": [],
        "🔬 体積・密度関連": [],
        "⚡ エネルギー関連": [],
        "📐 その他物性": [],
    }
    for d in descs:
        dl = d.lower()
        if any(k in dl for k in ["tb", "tm", "tc", "tg", "boil", "melt", "glass"]):
            groups["🌡️ 熱物性（Tb/Tm/Tc等）"].append(d)
        elif any(k in dl for k in ["vol", "density", "free_vol", "vanderwaals", "molar"]):
            groups["🔬 体積・密度関連"].append(d)
        elif any(k in dl for k in ["energy", "cohesive", "entangle"]):
            groups["⚡ エネルギー関連"].append(d)
        else:
            groups["📐 その他物性"].append(d)
    return {k: v for k, v in groups.items() if v}


def _group_skfp(descs: list[str]) -> dict[str, list[str]]:
    """scikit-fingerprints記述子の分類。"""
    groups: dict[str, list[str]] = {
        "🔑 Morgan/ECFP": [],
        "🏷️ MACCS": [],
        "🔷 Avalon": [],
        "🧩 その他FP": [],
    }
    for d in descs:
        dl = d.lower()
        if "morgan" in dl or "ecfp" in dl:
            groups["🔑 Morgan/ECFP"].append(d)
        elif "maccs" in dl:
            groups["🏷️ MACCS"].append(d)
        elif "avalon" in dl:
            groups["🔷 Avalon"].append(d)
        else:
            groups["🧩 その他FP"].append(d)
    return {k: v for k, v in groups.items() if v}


def _group_mordred(descs: list[str]) -> dict[str, list[str]]:
    """Mordred記述子の分類（数が多いためアルファベット先頭で分割）。"""
    if len(descs) <= 50:
        return {"全記述子": descs}
    groups: dict[str, list[str]] = {}
    for d in sorted(descs):
        # プレフィックスを除去してグループ化
        name = d.replace("mordred_", "").replace("mrd_", "")
        prefix = name[0].upper() if name else "?"
        groups.setdefault(f"Mordred {prefix}〜", []).append(d)
    return groups


# ═══════════════════════════════════════════════════════════
# 記述子テーブル（メイン表示）
# ═══════════════════════════════════════════════════════════
# ─── フィンガープリント系記述子の化学的意味を自動推定 ───
import re as _re

# プレフィクス → (化学的意味, カテゴリ, ソース)
_FP_MEANING_MAP: list[tuple[str, str, str, str]] = [
    # Morgan / ECFP系 — 円形の部分構造パターン
    ("Morgan_r2_", "半径2の原子近傍パターン（ECFP4相当）— 原子を中心に結合2つ分の局所構造をハッシュ化。活性に寄与する官能基・骨格断片の有無を捉える", "フィンガープリント", "RDKit"),
    ("Morgan_r3_", "半径3の広域部分構造（ECFP6相当）— 半径2より広い文脈で官能基配置を捕捉。立体選択性や結合部位の認識に有効", "フィンガープリント", "RDKit"),
    ("Morgan_r1_", "半径1の近接原子パターン（ECFP2相当）— 各原子の直接結合環境をコード化。原子タイプ別の存在・頻度を反映", "フィンガープリント", "RDKit"),
    ("ECFP4_", "拡張接続性FP(半径2) — Morgan FPと同等、原子の円環境を半径2でハッシュ化。QSAR標準指標", "フィンガープリント", "RDKit"),
    ("ECFP6_", "拡張接続性FP(半径3) — より広い局所環境を捉え、類似骨格の識別精度が向上", "フィンガープリント", "RDKit"),
    ("ECFP2_", "拡張接続性FP(半径1) — 最小範囲の原子環境。計算が軽く大規模スクリーニング向き", "フィンガープリント", "RDKit"),
    ("FCFP4_", "薬理学的特徴FP(半径2) — 原子タイプの代わりに薬理学的特徴(HBD/HBA/正負電荷/疎水性/芳香族)を用いたMorgan変種", "フィンガープリント", "RDKit"),
    ("FCFP6_", "薬理学的特徴FP(半径3) — FCFP4の広域版。より広い薬理学的文脈を反映", "フィンガープリント", "RDKit"),
    ("FCFP2_", "薬理学的特徴FP(半径1) — 最小範囲の薬理学的特徴環境", "フィンガープリント", "RDKit"),

    # RDKitFP — パス型フィンガープリント
    ("RDKitFP_", "結合パスFP — 分子グラフ上の1-7結合の線形パスをハッシュ化。分子骨格の接続パターンを表現し、構造類似性検索の基盤", "フィンガープリント", "RDKit"),

    # MACCS — 事前定義の構造鍵
    ("MACCS_", "MACCSキー — MDLが定義した166個の部分構造パターンの有無。製薬で広く使用される標準的な構造フィルター", "フィンガープリント", "RDKit"),

    # AtomPair — 原子ペア
    ("AtomPairFP_", "原子ペアFP — 全原子ペアの(原子タイプ, パス長, 原子タイプ)三つ組をコード化。分子の大域的な原子配置を捉える", "フィンガープリント", "RDKit"),

    # TopologicalTorsion — ねじれ角
    ("TopologicalTorsionFP_", "トポロジカルねじれFP — 連続4原子のパス(A-B-C-D)をコード化。分子の局所的な三次元形状を間接的に反映", "フィンガープリント", "RDKit"),
    ("TorsionFP_", "ねじれFP — 4原子パスの部分構造パターン。立体配座の多様性を間接的に表現", "フィンガープリント", "RDKit"),

    # Avalon
    ("AvalonFP_", "AvalonFP — パスベース＋構造鍵のハイブリッド。高速な類似性検索と部分構造マッチングの両立に最適化", "フィンガープリント", "RDKit"),

    # scikit-fingerprints系
    ("MHFP_", "MinHash FP — 分子グラフのMinHashベース表現。ハミング距離で高速な類似性評価が可能", "フィンガープリント", "scikit-fingerprints"),
    ("MAP4_", "MinHash原子ペアFP — 原子ペアのMinHash。分子サイズに不変で大小様々な分子の比較に適する", "フィンガープリント", "scikit-fingerprints"),
    ("ErG_", "拡張削減グラフFP — 薬理学的特徴ノードの距離行列から導出。ファーマコフォア型の構造表現", "フィンガープリント", "scikit-fingerprints"),
    ("Lingo_", "Lingo FP — SMILES文字列のq-gramベース表現。2D構造に依存しないテキスト的類似性", "フィンガープリント", "scikit-fingerprints"),
    ("Layered_", "レイヤードFP — 結合タイプ・環帰属等の複数レイヤーを組合せたリッチな表現", "フィンガープリント", "RDKit"),
    ("Pattern_", "パターンFP — SMARTS定義の部分構造パターンの存在を検出", "フィンガープリント", "RDKit"),
    ("E3FP_", "3D拡張接続性FP — 3D座標から原子環境を球状にコード化。立体構造を直接反映", "フィンガープリント", "scikit-fingerprints"),

    # Molfeat/DeepChem系
    ("molfeat_", "Molfeat特徴量 — 学習済みモデルによる分子埋め込み表現", "深層学習特徴量", "molfeat"),
    ("mol2vec_", "Mol2Vec — Word2Vecで学習した部分構造のベクトル表現。意味的類似性を捕捉", "深層学習特徴量", "mol2vec"),
    ("chemprop_", "Chemprop — メッセージパッシングNNによる分子グラフ埋め込み", "深層学習特徴量", "chemprop"),
    ("molai_", "MolAI — 自社深層学習モデルのPCA圧縮特徴量", "深層学習特徴量", "MolAI"),
    ("uma_", "UMA — Universal Molecular Attention。大規模事前学習モデルの分子表現", "深層学習特徴量", "UMA"),

    # PaDEL
    ("PaDEL_", "PaDEL記述子 — Javaベースの記述子計算ライブラリ由来の特徴量", "記述子", "PaDEL"),

    # DescriptaStorus
    ("DescriptaStorus_", "DescriptaStorus記述子 — 高速な記述子計算ライブラリの特徴量", "記述子", "DescriptaStorus"),
]

_FP_BIT_PATTERN = _re.compile(r"^(.+?_)(\d+)$")


def _infer_fp_meaning(desc_name: str) -> tuple[str, str, str] | None:
    """記述子名からフィンガープリント系の意味を推定する。

    Returns:
        (meaning, category, library) or None
    """
    # まずプレフィクス完全一致
    for prefix, meaning, cat, lib in _FP_MEANING_MAP:
        if desc_name.startswith(prefix):
            return meaning, cat, lib

    # それでもなければ _数字 パターンで汎用FPとして扱う
    m = _FP_BIT_PATTERN.match(desc_name)
    if m:
        prefix = m.group(1).rstrip("_")
        return f"{prefix}系フィンガープリントのビット特徴量", "フィンガープリント", ""

    return None


def _group_fp_descriptors(desc_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """記述子リストをFPグループとその他に分離する。

    Returns:
        (non_fp_list, fp_groups_dict)
        fp_groups_dict: {group_label: [bit_names]}
    """
    fp_groups: dict[str, list[str]] = {}
    non_fp: list[str] = []

    for d in desc_names:
        m = _FP_BIT_PATTERN.match(d)
        if m:
            prefix = m.group(1)
            # グループラベル
            label = prefix.rstrip("_")
            fp_groups.setdefault(label, []).append(d)
        else:
            non_fp.append(d)

    return non_fp, fp_groups


def _render_descriptor_table(state: dict) -> None:
    """計算済み記述子の一覧 — 3層構造UI。

    第1層: 正負TOP5サマリーバー（常に表示）
    第2層: グループ別集計（折りたたみ）
    第3層: 詳細テーブル（折りたたみ、検索+フィルター+バー表示）
    """
    import numpy as np
    from scipy import stats as sp_stats

    precalc_df = state.get("precalc_df")
    if precalc_df is None:
        return

    target_col = state.get("target_col")
    df = state.get("df")
    all_descs = list(precalc_df.columns)
    n_total = len(all_descs)

    # ── 相関係数の計算（符号付き）──
    corr_signed: dict[str, float] = {}
    corr_abs: dict[str, float] = {}
    p_values: dict[str, float] = {}
    if target_col and df is not None and target_col in df.columns:
        try:
            target_s = df[target_col]
            if pd.api.types.is_numeric_dtype(target_s):
                aligned = target_s.iloc[:len(precalc_df)].reset_index(drop=True)
                for col in all_descs:
                    try:
                        vals = precalc_df[col].astype(float).iloc[:len(aligned)]
                        mask = np.isfinite(vals) & np.isfinite(aligned)
                        if mask.sum() < 5:
                            continue
                        r, p = sp_stats.pearsonr(vals[mask], aligned[mask])
                        if np.isfinite(r):
                            corr_signed[col] = float(r)
                            corr_abs[col] = abs(float(r))
                            p_values[col] = float(p)
                    except Exception:
                        pass
        except Exception:
            pass

    # ── FPグループ化 ──
    non_fp, fp_groups = _group_fp_descriptors(all_descs)

    # ── グループ分類 ──
    from frontend_nicegui.components.descriptor_selector_dialog import (
        _classify_descriptor_group, _GROUP_DISPLAY,
    )
    group_map: dict[str, str] = {}
    group_members: dict[str, list[str]] = {}
    for col in all_descs:
        g = _classify_descriptor_group(col)
        group_map[col] = g
        group_members.setdefault(g, []).append(col)

    # ── 推薦記述子 ──
    rec_names = set()
    applied_rec = state.get("_applied_recommendation")
    if applied_rec:
        rec_names = {d.name for d in applied_rec.descriptors}

    # ── 選択状態 ──
    selected = set(state.get("selected_descriptors", all_descs))

    # ═══════════════════════════════════════════════
    # ヘッダー + 一括操作
    # ═══════════════════════════════════════════════
    with ui.card().classes("full-width q-pa-md q-mb-sm glass-card"):
        with ui.row().classes("items-center q-gutter-sm full-width"):
            ui.icon("table_chart").classes("text-cyan text-h6")
            ui.label(f"記述子一覧: {n_total}個計算済み").classes("text-subtitle1")
            if corr_signed:
                ui.badge(
                    f"|r|計算済み ({len(corr_signed)}列)", color="teal",
                ).props("outline")

        ui.separator().classes("q-my-xs")

        # 一括操作
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            if applied_rec:
                ui.chip(
                    f"📌 適用中: {applied_rec.target_name}",
                    icon="auto_awesome", color="cyan",
                ).props("outline dense")

            if corr_abs:
                sorted_descs = sorted(
                    all_descs, key=lambda d: corr_abs.get(d, 0), reverse=True,
                )

                def _select_top(n: int, descs=sorted_descs):
                    state["selected_descriptors"] = descs[:n]
                    ui.notify(f"✅ 相関上位{n}件を選択", type="positive")

                ui.button("上位10件", on_click=lambda: _select_top(10)).props(
                    "outline size=sm no-caps color=cyan"
                )
                ui.button("上位30件", on_click=lambda: _select_top(30)).props(
                    "outline size=sm no-caps color=cyan"
                )
                ui.button("上位50件", on_click=lambda: _select_top(50)).props(
                    "outline size=sm no-caps color=grey"
                )
                ui.button(
                    "高相関のみ(|r|≧0.3)",
                    on_click=lambda: (
                        state.update({
                            "selected_descriptors": [
                                d for d in all_descs if corr_abs.get(d, 0) >= 0.3
                            ]
                        }),
                        ui.notify(
                            f"✅ |r|≧0.3: {len([d for d in all_descs if corr_abs.get(d,0)>=0.3])}件を選択",
                            type="positive",
                        ),
                    ),
                ).props("outline size=sm no-caps color=teal")

            ui.button(
                "全選択 (ALL SET)",
                on_click=lambda: state.update({"selected_descriptors": list(all_descs)}),
            ).props("flat size=sm no-caps color=cyan")
            ui.button(
                "全解除 (RESET)",
                on_click=lambda: state.update({"selected_descriptors": []}),
            ).props("flat size=sm no-caps color=red-4")

        # ═══════════════════════════════════════════════
        # 第1層：相関サマリー（正負TOP5バー）— 常に表示
        # ═══════════════════════════════════════════════
        if corr_signed:
            with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
                "border: 1px solid rgba(0,188,212,0.25); border-radius: 10px;"
                "background: rgba(0,20,40,0.3);"
            ):
                ui.label("📊 相関サマリー：目的変数 vs 記述子").classes(
                    "text-subtitle2 text-cyan q-mb-xs"
                )

                # 正負に分けてソート
                positives = sorted(
                    [(d, r) for d, r in corr_signed.items() if r > 0],
                    key=lambda x: x[1], reverse=True,
                )[:5]
                negatives = sorted(
                    [(d, r) for d, r in corr_signed.items() if r < 0],
                    key=lambda x: x[1],
                )[:5]

                with ui.row().classes("full-width q-gutter-md"):
                    # ── 正の相関 TOP5 ──
                    with ui.column().classes("col"):
                        ui.label("🔵 正の相関 TOP5").classes(
                            "text-body2 text-bold q-mb-xs"
                        ).style("color: #4fc3f7;")
                        max_pos = positives[0][1] if positives else 1.0
                        for name, r_val in positives:
                            bar_pct = abs(r_val) / max(abs(max_pos), 0.01) * 100
                            with ui.row().classes(
                                "items-center q-gutter-xs full-width"
                            ).style("min-height: 22px;"):
                                ui.label(name[:25]).classes(
                                    "text-caption"
                                ).style(
                                    "min-width: 150px; max-width: 150px;"
                                    "overflow: hidden; text-overflow: ellipsis;"
                                    "white-space: nowrap;"
                                )
                                with ui.element("div").style(
                                    "flex:1; height:14px; background:rgba(255,255,255,0.08);"
                                    "border-radius:4px; overflow:hidden;"
                                ):
                                    ui.element("div").style(
                                        f"width:{bar_pct:.0f}%; height:100%;"
                                        "background: linear-gradient(90deg, #1565c0, #42a5f5);"
                                        "border-radius:4px;"
                                    )
                                ui.label(f"+{r_val:.3f}").classes(
                                    "text-caption text-bold"
                                ).style("min-width:55px; color:#4fc3f7; text-align:right;")

                        if not positives:
                            ui.label("（なし）").classes("text-caption text-grey")

                    # ── 負の相関 TOP5 ──
                    with ui.column().classes("col"):
                        ui.label("🔴 負の相関 TOP5").classes(
                            "text-body2 text-bold q-mb-xs"
                        ).style("color: #ef5350;")
                        max_neg = abs(negatives[0][1]) if negatives else 1.0
                        for name, r_val in negatives:
                            bar_pct = abs(r_val) / max(max_neg, 0.01) * 100
                            with ui.row().classes(
                                "items-center q-gutter-xs full-width"
                            ).style("min-height: 22px;"):
                                ui.label(name[:25]).classes(
                                    "text-caption"
                                ).style(
                                    "min-width: 150px; max-width: 150px;"
                                    "overflow: hidden; text-overflow: ellipsis;"
                                    "white-space: nowrap;"
                                )
                                with ui.element("div").style(
                                    "flex:1; height:14px; background:rgba(255,255,255,0.08);"
                                    "border-radius:4px; overflow:hidden;"
                                    "display:flex; justify-content:flex-end;"
                                ):
                                    ui.element("div").style(
                                        f"width:{bar_pct:.0f}%; height:100%;"
                                        "background: linear-gradient(270deg, #c62828, #ef5350);"
                                        "border-radius:4px;"
                                    )
                                ui.label(f"{r_val:.3f}").classes(
                                    "text-caption text-bold"
                                ).style("min-width:55px; color:#ef5350; text-align:right;")

                        if not negatives:
                            ui.label("（なし）").classes("text-caption text-grey")

                # 凡例
                with ui.row().classes("q-gutter-sm q-mt-xs items-center"):
                    with ui.element("div").style(
                        "width:12px;height:12px;background:#42a5f5;border-radius:2px;"
                    ):
                        pass
                    ui.label("正の相関").classes("text-caption text-grey")
                    with ui.element("div").style(
                        "width:12px;height:12px;background:#ef5350;border-radius:2px;"
                    ):
                        pass
                    ui.label("負の相関").classes("text-caption text-grey")
                    ui.label("（バー長 = |相関係数|）").classes("text-caption text-grey")

        # ═══════════════════════════════════════════════
        # 第2層：グループ別集計（折りたたみ）
        # ═══════════════════════════════════════════════
        with ui.expansion(
            "📋 記述子グループ別集計", icon="analytics",
        ).classes("full-width q-mb-sm").props("dense"):
            group_stats = []
            for g_key, members in sorted(
                group_members.items(),
                key=lambda x: max((corr_abs.get(m, 0) for m in x[1]), default=0),
                reverse=True,
            ):
                g_info = _GROUP_DISPLAY.get(g_key, {"label": g_key, "icon": "📄"})
                corrs_in_group = [corr_abs.get(m, 0) for m in members if m in corr_abs]
                max_r = max(corrs_in_group) if corrs_in_group else 0
                avg_r = sum(corrs_in_group) / len(corrs_in_group) if corrs_in_group else 0
                high_n = sum(1 for c in corrs_in_group if c >= 0.3)
                n_sel = sum(1 for m in members if m in selected)
                group_stats.append({
                    "group": f"{g_info['icon']} {g_info['label']}",
                    "count": len(members),
                    "max_r": f"{max_r:.3f}",
                    "avg_r": f"{avg_r:.3f}",
                    "high_n": high_n,
                    "selected": f"{n_sel}/{len(members)}",
                    "_max_r_raw": max_r,
                })

            g_columns = [
                {"name": "group", "label": "グループ", "field": "group"},
                {"name": "count", "label": "個数", "field": "count", "sortable": True},
                {"name": "max_r", "label": "最大|r|", "field": "max_r", "sortable": True},
                {"name": "avg_r", "label": "平均|r|", "field": "avg_r", "sortable": True},
                {"name": "high_n", "label": "高相関(≧0.3)", "field": "high_n", "sortable": True},
                {"name": "selected", "label": "選択済", "field": "selected"},
            ]
            ui.table(
                columns=g_columns, rows=group_stats, row_key="group",
            ).classes("full-width").props("dense flat bordered")

            # 合計行
            total_high = sum(r["high_n"] for r in group_stats)
            total_sel = sum(1 for d in all_descs if d in selected)
            with ui.row().classes("q-gutter-sm q-mt-xs"):
                ui.label(
                    f"合計: {n_total}記述子 / 高相関(|r|≧0.3): {total_high}個"
                    f" / 選択済: {total_sel}個"
                ).classes("text-caption text-grey")

        # ═══════════════════════════════════════════════
        # 第3層：詳細テーブル（折りたたみ、検索+バー+p値）
        # ═══════════════════════════════════════════════
        with ui.expansion(
            "🔬 詳細テーブル（全記述子）", icon="table_rows",
        ).classes("full-width q-mb-sm").props("dense"):

            # ── メタ情報読み込み ──
            desc_meta: dict[str, dict] = {}
            try:
                from backend.chem.recommender import get_all_target_recommendations
                for rec in get_all_target_recommendations():
                    for d in rec.descriptors:
                        if d.name not in desc_meta:
                            desc_meta[d.name] = {
                                "meaning": d.meaning,
                                "category": d.category,
                                "library": d.library,
                            }
            except ImportError:
                pass
            from frontend_nicegui.components.descriptor_catalog import (
                get_catalog as _gc, SUPPORTED_ENGINES as _se,
            )
            for _e in _se:
                _c = _gc(_e)
                if _c:
                    for _cn, _ci_list in _c.items():
                        for _ci in _ci_list:
                            dn = _ci["name"]
                            if dn.startswith("_"):
                                continue
                            if dn not in desc_meta:
                                desc_meta[dn] = {
                                    "meaning": _ci.get("short", ""),
                                    "category": _ci.get("cat", _cn),
                                    "library": _e,
                                }

            # 通常記述子テーブル行
            sorted_list = sorted(
                non_fp,
                key=lambda d: corr_abs.get(d, 0),
                reverse=True,
            )

            detail_rows = []
            for d in sorted_list:
                meta = desc_meta.get(d, {})
                meaning = meta.get("meaning", "")
                library = meta.get("library", "")
                if not meaning:
                    inferred = _infer_fp_meaning(d)
                    if inferred:
                        meaning = inferred[0]
                        library = inferred[2]

                r_signed = corr_signed.get(d)
                r_abs_val = corr_abs.get(d, 0)
                p_val = p_values.get(d)

                # バー表示用HTML（NiceGUIのcell slotで表現）
                if r_signed is not None:
                    sign = "+" if r_signed >= 0 else ""
                    corr_text = f"{sign}{r_signed:.3f}"
                else:
                    corr_text = "—"

                p_text = ""
                if p_val is not None:
                    if p_val < 0.001:
                        p_text = "<0.001"
                    elif p_val < 0.01:
                        p_text = f"{p_val:.3f}"
                    elif p_val < 0.05:
                        p_text = f"{p_val:.3f}"
                    else:
                        p_text = f"{p_val:.3f} ⚠"

                detail_rows.append({
                    "name": d,
                    "sel": "✅" if d in selected else "",
                    "rec": "⭐" if d in rec_names else "",
                    "corr": corr_text,
                    "corr_raw": r_abs_val,
                    "p_val": p_text,
                    "library": library,
                    "meaning": meaning[:50],
                    "group": _GROUP_DISPLAY.get(
                        group_map.get(d, ""), {"label": ""}
                    )["label"],
                })

            # FPグループ行
            for gl, bits in sorted(fp_groups.items(), key=lambda x: -len(x[1])):
                n_bits = len(bits)
                n_sel = sum(1 for b in bits if b in selected)
                corrs = [corr_abs.get(b, 0) for b in bits if b in corr_abs]
                avg_c = sum(corrs) / len(corrs) if corrs else 0
                max_c = max(corrs) if corrs else 0
                fp_info = _infer_fp_meaning(bits[0]) if bits else None
                detail_rows.append({
                    "name": f"📦 {gl} ({n_bits}ビット, {n_sel}選択)",
                    "sel": "✅" if n_sel == n_bits else ("⬜" if n_sel > 0 else ""),
                    "rec": "",
                    "corr": f"avg:{avg_c:.3f} max:{max_c:.3f}" if corrs else "—",
                    "corr_raw": avg_c,
                    "p_val": "",
                    "library": fp_info[2] if fp_info else "",
                    "meaning": fp_info[0][:50] if fp_info else f"{gl}系FP",
                    "group": "FP",
                })

            d_columns = [
                {"name": "sel", "label": "✓", "field": "sel", "align": "center",
                 "style": "width:30px;"},
                {"name": "rec", "label": "⭐", "field": "rec", "align": "center",
                 "style": "width:30px;"},
                {"name": "name", "label": "記述子名", "field": "name", "sortable": True},
                {"name": "corr", "label": "相関係数(r)", "field": "corr", "sortable": True},
                {"name": "p_val", "label": "p値", "field": "p_val", "sortable": True},
                {"name": "group", "label": "グループ", "field": "group", "sortable": True},
                {"name": "library", "label": "ソース", "field": "library", "sortable": True},
                {"name": "meaning", "label": "意味", "field": "meaning"},
            ]

            detail_table = ui.table(
                columns=d_columns,
                rows=detail_rows,
                row_key="name",
                selection="multiple",
                pagination={"rowsPerPage": 50, "sortBy": "corr_raw", "descending": True},
            ).classes("full-width").props("dense flat bordered")

            # 選択状態の同期
            non_group = [r for r in detail_rows if not r["name"].startswith("📦")]
            detail_table.selected = [r for r in non_group if r["name"] in selected]

            def _on_detail_sel(e):
                sel_names = [r["name"] for r in e.selection if not r["name"].startswith("📦")]
                existing_fp = [
                    d for d in state.get("selected_descriptors", [])
                    if _FP_BIT_PATTERN.match(d)
                ]
                state["selected_descriptors"] = sel_names + existing_fp

            detail_table.on_select(_on_detail_sel)

        # ── FPグループ一括ON/OFFボタン ──
        if fp_groups:
            with ui.row().classes("q-gutter-sm q-mt-sm"):
                ui.label("📦 FPグループ一括:").classes("text-caption text-grey-5")
                for gl, bits in sorted(fp_groups.items(), key=lambda x: -len(x[1])):
                    n = len(bits)
                    n_sel = sum(1 for b in bits if b in selected)
                    is_all_on = (n_sel == n)

                    def _toggle_group(group_bits=bits, group_name=gl, all_on=is_all_on):
                        cur = set(state.get("selected_descriptors", []))
                        if all_on:
                            cur -= set(group_bits)
                            ui.notify(f"📦 {group_name} 全{len(group_bits)}ビットOFF", type="info")
                        else:
                            cur |= set(group_bits)
                            ui.notify(
                                f"📦 {group_name} 全{len(group_bits)}ビットON", type="positive",
                            )
                        state["selected_descriptors"] = list(cur)

                    ui.button(
                        f"{gl} ({n_sel}/{n})",
                        on_click=_toggle_group,
                    ).props(
                        f"{'unelevated' if is_all_on else 'outline'} dense size=sm no-caps "
                        f"color={'cyan' if is_all_on else 'grey-6'}"
                    ).classes("text-xs")


# ═══════════════════════════════════════════════════════════
# エンジン詳細（上級者向け折りたたみ）
# ═══════════════════════════════════════════════════════════
def _render_engine_details(adapters: dict, state: dict) -> None:
    """エンジンの有効/無効状態を表示（情報提供のみ、ON/OFF操作は行わない）。"""
    # カテゴリごとにグルーピング
    categories: dict[str, list[dict]] = {}
    for eng in _ENGINE_INFO:
        cat = eng.get("category", "その他")
        categories.setdefault(cat, []).append(eng)

    for cat_name, engines in categories.items():
        ui.label(f"■ {cat_name}").classes("text-subtitle2 q-mt-sm q-mb-xs")
        with ui.row().classes("q-gutter-sm full-width"):
            for eng in engines:
                cls_name = eng["cls"]
                avail = _is_available(adapters, cls_name)
                border = "rgba(0,212,255,0.4)" if avail else "rgba(255,255,255,0.08)"
                bg = "rgba(0,30,50,0.4)" if avail else "rgba(30,30,30,0.3)"

                with ui.card().classes("q-pa-sm").style(
                    f"border: 1px solid {border}; border-radius: 8px; background: {bg};"
                    "min-width: 200px; max-width: 300px;"
                ):
                    with ui.row().classes("items-center no-wrap q-gutter-xs"):
                        if avail:
                            ui.icon("check_circle", color="green").classes("text-body1")
                        else:
                            ui.icon("cancel", color="grey").classes("text-body1")
                        ui.label(eng["label"]).classes("text-body2 text-bold")

                    ui.label(
                        f"{eng['speed']} | {eng['dims']}次元 | {eng['desc']}"
                    ).classes("text-caption text-grey").style("font-size: 0.7rem;")

                    # 詳細選択ボタン（カタログが存在するエンジンのみ）
                    _engine_catalog_map = {
                        "RDKitAdapter": "RDKit",
                        "XTBAdapter": "XTB",
                        "GroupContribAdapter": "原子団寄与法",
                        "SkfpAdapter": "scikit-FP",
                        "MordredAdapter": "Mordred",
                        "MolfeatAdapter": "Molfeat",
                        "CosmoAdapter": "COSMO-RS",
                    }
                    catalog_name = _engine_catalog_map.get(cls_name)
                    manual_url = eng.get("manual_url", "")

                    with ui.row().classes("items-center q-gutter-xs q-mt-xs"):
                        if catalog_name and avail:
                            ui.button(
                                "詳細選択",
                                on_click=lambda cn=catalog_name: open_descriptor_detail_dialog(
                                    cn, state
                                ),
                            ).props("flat dense size=xs no-caps color=cyan").tooltip(
                                "個別の記述子を選択・解除"
                            )

                        if manual_url:
                            ui.button(
                                "📖 マニュアル",
                                on_click=lambda url=manual_url: ui.run_javascript(
                                    f"window.open('{url}', '_blank')"
                                ),
                            ).props("flat dense size=xs no-caps color=grey-5").tooltip(
                                f"公式ドキュメントを開く: {manual_url}"
                            )




def _apply_recommendation(rec, state: dict) -> None:
    """推薦記述子セットを適用（エンジンは全OFFにしない）。"""
    # 推薦記述子名をstateに保存
    state["selected_descriptors"] = [d.name for d in rec.descriptors]
    state["_applied_recommendation"] = rec

    ui.notify(
        f"✅ 「{rec.target_name}」の推薦記述子セット適用 ({len(rec.descriptors)}個)",
        type="positive", timeout=5000,
    )


# ═══════════════════════════════════════════════════════════
# MolAI PCA 設定
# ═══════════════════════════════════════════════════════════
def _render_molai_pca(state: dict) -> None:
    """MolAI PCA次元数設定 + 累積寄与率に基づく自動最適化。"""
    n_samples = len(state["df"]) if state.get("df") is not None else 25
    auto_n = min(max(n_samples // 2, 5), 128)
    current_n = state.get("molai_n_components", auto_n)

    ui.label(f"サンプル数: {n_samples} → 推奨PCA次元: {auto_n}").classes("text-caption text-grey")

    # ── 手動スライダー ──
    with ui.row().classes("items-center full-width q-gutter-sm"):
        ui.label("PCA次元数:").classes("text-body2")
        slider = ui.slider(
            min=1, max=min(256, n_samples), value=current_n, step=1,
        ).props("label-always").classes("full-width")
        dim_label = ui.label(f"{current_n}次元").classes("text-body2 text-cyan text-bold")

    def _update_pca(e):
        state["molai_n_components"] = int(e.value)
        dim_label.text = f"{int(e.value)}次元"
    slider.on_value_change(_update_pca)

    # ── 累積寄与率に基づく自動最適化 ──
    mev = state.get("molai_explained_variance")

    with ui.card().classes("full-width q-pa-sm q-mt-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 8px;"
        "background: rgba(0,30,50,0.3);"
    ):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.icon("auto_fix_high", color="amber").classes("text-body1")
            ui.label("累積寄与率で自動決定").classes("text-body2 text-bold")

        ui.label(
            "累積寄与率の閾値を指定すると、その閾値を超える最小の次元数を自動計算します。"
        ).classes("text-caption text-grey q-mb-xs")

        threshold = state.get("pca_threshold", 95)
        with ui.row().classes("items-center q-gutter-sm full-width"):
            ui.label("閾値:").classes("text-body2")
            thresh_slider = ui.slider(
                min=80, max=99, value=threshold, step=1,
            ).props("label-always").classes("full-width")
            thresh_label = ui.label(f"{threshold}%").classes("text-body2 text-amber text-bold")

        def _update_threshold(e):
            state["pca_threshold"] = int(e.value)
            thresh_label.text = f"{int(e.value)}%"
        thresh_slider.on_value_change(_update_threshold)

        def _auto_optimize():
            mev_data = state.get("molai_explained_variance")
            if not mev_data or not mev_data.get("cumulative"):
                ui.notify(
                    "PCA寄与率データがありません。先にMolAI記述子を計算してください。",
                    type="warning",
                )
                return
            cum = mev_data["cumulative"]
            target_threshold = state.get("pca_threshold", 95) / 100.0
            optimal_n = len(cum)  # デフォルトは全次元
            for i, c in enumerate(cum):
                if c >= target_threshold:
                    optimal_n = i + 1
                    break
            state["molai_n_components"] = optimal_n
            slider.value = optimal_n
            dim_label.text = f"{optimal_n}次元"
            actual_var = cum[min(optimal_n - 1, len(cum) - 1)] * 100
            ui.notify(
                f"✅ 累積寄与率 {actual_var:.1f}% → {optimal_n}次元を自動選択",
                type="positive",
            )

        ui.button(
            "🔍 自動最適化",
            on_click=_auto_optimize,
        ).props("unelevated size=sm no-caps color=amber").classes("q-mt-xs")

        if mev and mev.get("cumulative"):
            cum = mev["cumulative"]
            t = state.get("pca_threshold", 95) / 100.0
            opt = len(cum)
            for i, c in enumerate(cum):
                if c >= t:
                    opt = i + 1
                    break
            actual = cum[min(opt - 1, len(cum) - 1)] * 100
            ui.label(
                f"💡 現在の閾値 {int(t*100)}% → 推奨: {opt}次元 (実際の累積寄与率: {actual:.1f}%)"
            ).classes("text-caption text-teal q-mt-xs")

    # ── PCA寄与率グラフ ──
    if mev and mev.get("ratio"):
        _render_pca_chart(mev, state.get("molai_n_components", auto_n))


def _render_pca_chart(mev: dict, selected_n: int | None = None) -> None:
    """MolAI PCA累積寄与率のグラフ。選択中の次元に縦線を追加。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        ui.label("Plotly未インストール — グラフ表示不可").classes("text-grey")
        return

    evr = mev["ratio"]
    evc = mev["cumulative"]
    pcs = [f"PC{i+1}" for i in range(len(evr))]

    fig = go.Figure()
    fig.add_bar(x=pcs, y=[v * 100 for v in evr], name="寄与率 (%)", marker_color="#4c9be8")
    fig.add_scatter(
        x=pcs, y=[v * 100 for v in evc], name="累積寄与率 (%)",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#f4a261", width=2), marker=dict(size=5),
    )

    # 選択中の次元に縦線を描画
    if selected_n and 1 <= selected_n <= len(pcs):
        fig.add_vline(
            x=selected_n - 1, line_dash="dash", line_color="#00e5ff",
            annotation_text=f"選択: PC{selected_n}",
            annotation_font_color="#00e5ff",
        )

    fig.update_layout(
        yaxis=dict(title="寄与率 (%)", range=[0, max(v * 100 for v in evr) * 1.15]),
        yaxis2=dict(title="累積寄与率 (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
        legend=dict(orientation="h", y=1.15),
        height=280, margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"),
    )
    ui.plotly(fig).classes("full-width")


# ═══════════════════════════════════════════════════════════
# 電荷・スピン設定
# ═══════════════════════════════════════════════════════════
def _render_charge_settings(state: dict) -> None:
    """形式電荷・スピン多重度・プロトン化モード。"""
    ui.label(
        "量子化学計算（XTB/COSMO等）の電荷・スピン設定。通常は変更不要。"
    ).classes("text-caption text-grey q-mb-sm")

    with ui.row().classes("q-gutter-md full-width"):
        charge = ui.number(
            "形式電荷", value=state.get("formal_charge", 0),
            min=-3, max=3, step=1,
        ).props("dense outlined").classes("w-32")

        def _update_charge(e):
            state["formal_charge"] = int(e.value) if e.value is not None else 0
        charge.on_value_change(_update_charge)

        spin = ui.select(
            {1: "1 (閉殻)", 2: "2 (ラジカル)", 3: "3 (三重項)"},
            value=state.get("spin_multiplicity", 1),
            label="スピン多重度",
        ).props("dense outlined").classes("w-40")

        def _update_spin(e):
            state["spin_multiplicity"] = e.value
        spin.on_value_change(_update_spin)

    with ui.row().classes("q-gutter-sm q-mt-sm full-width items-center"):
        prot_mode = ui.select(
            {"as_is": "SMILESのまま", "auto_ph": "pH指定で自動プロトン化", "neutral": "全て中性化"},
            value=state.get("protonate_mode", "as_is"),
            label="プロトン化モード",
        ).props("dense outlined").classes("w-48")

        def _update_prot(e):
            state["protonate_mode"] = e.value
        prot_mode.on_value_change(_update_prot)

        if state.get("protonate_mode") == "auto_ph":
            ph = ui.number(
                "pH", value=state.get("target_ph", 7.4), min=0, max=14, step=0.1,
            ).props("dense outlined").classes("w-24")

            def _update_ph(e):
                state["target_ph"] = float(e.value) if e.value is not None else 7.4
            ph.on_value_change(_update_ph)

    pcm = ui.select(
        {"gasteiger": "Gasteiger", "xtb_mulliken": "XTB Mulliken", "none": "なし"},
        value=state.get("partial_charge_model", "gasteiger"),
        label="部分電荷モデル",
    ).props("dense outlined").classes("w-40 q-mt-sm")

    def _update_pcm(e):
        state["partial_charge_model"] = e.value
    pcm.on_value_change(_update_pcm)


# ═══════════════════════════════════════════════════════════
# カスタムプラグイン管理
# ═══════════════════════════════════════════════════════════
def _render_custom_plugins(state: dict) -> None:
    """カスタムプラグインのアップロード/削除/テンプレート/既存アダプタからの作成UI。"""
    from backend.chem.descriptors import get_custom_dir, invalidate_cache

    ui.label(
        "custom/ ディレクトリに .py ファイルを追加して独自記述子を作成できます。"
        "既存の記述子アダプタをベースにして変更を加える方法が最も効率的です。"
    ).classes("text-grey text-caption q-mb-sm")

    custom_dir = get_custom_dir()
    custom_files = [f for f in custom_dir.glob("*.py") if not f.name.startswith("_")]

    # ── 既存カスタムプラグインの一覧 ──
    if custom_files:
        ui.label(f"📁 登録済みカスタムプラグイン ({len(custom_files)})").classes("text-subtitle2")
        with ui.column().classes("q-gutter-xs"):
            for cf in custom_files:
                with ui.row().classes("items-center q-gutter-xs"):
                    ui.icon("description", color="amber").classes("text-body1")
                    ui.label(cf.name).classes("text-body2")
                    # ソースコード閲覧ボタン
                    ui.button(
                        icon="visibility",
                        on_click=lambda _, f=cf: _show_source_dialog(f),
                    ).props("flat dense color=cyan size=sm").tooltip("ソースコードを確認")
                    ui.button(
                        icon="delete",
                        on_click=lambda _, f=cf: _delete_custom(f),
                    ).props("flat dense color=red size=sm")
    else:
        ui.label("カスタムプラグインはまだありません").classes("text-grey text-caption")

    ui.separator()

    # ── アップロード ──
    async def _on_upload(e):
        content = e.content.read()
        filename = e.name
        if not filename.endswith(".py"):
            ui.notify("⚠️ .py ファイルのみアップロードできます", type="warning")
            return
        dest = custom_dir / filename
        dest.write_bytes(content)
        invalidate_cache()
        ui.notify(f"✅ {filename} をアップロードしました", type="positive")

    ui.upload(
        label="カスタム記述子 .py をアップロード",
        on_upload=_on_upload,
        auto_upload=True,
    ).props("accept='.py' flat dense").classes("full-width")

    # ── テンプレート ──
    with ui.row().classes("q-gutter-sm q-mt-sm"):
        ui.label("テンプレート:").classes("text-caption text-grey")
        for tpl_name in ["_template_simple.py", "_template_with_config.py"]:
            tpl_path = custom_dir / tpl_name
            if tpl_path.exists():
                ui.button(
                    tpl_name.replace("_template_", "").replace(".py", ""),
                    on_click=lambda _, p=tpl_path: ui.download(str(p)),
                ).props("flat dense size=sm color=cyan no-caps")

    # ── 既存記述子アダプタからコピーして作成 ──────────────────────────────────
    ui.separator()
    ui.label("📋 既存記述子からカスタムプラグインを作成").classes("text-subtitle2 q-mt-sm")
    ui.label(
        "既存のSMILES記述子アダプタのソースコードをコピーし、"
        "カスタムプラグインとして保存してから編集できます。"
    ).classes("text-grey text-caption q-mb-sm")

    adapter_sources = _detect_adapter_sources()
    if adapter_sources:
        with ui.row().classes("q-gutter-sm flex-wrap"):
            for _adp_name, _adp_path in adapter_sources.items():
                ui.button(
                    f"📄 {_adp_name}",
                    on_click=lambda _, n=_adp_name, p=_adp_path: _copy_adapter_to_custom(
                        n, p, custom_dir,
                    ),
                ).props("outline dense size=sm no-caps color=teal").tooltip(
                    f"{_adp_path.name} をカスタムプラグインとしてコピー"
                )
    else:
        ui.label("利用可能なアダプタが見つかりません").classes("text-grey text-caption")

    # ── AI で記述子を生成（実験的・上級者向け） ─────────────────────────────
    ui.separator().classes("q-mt-md")
    with ui.expansion(
        "🤖 AI で記述子を生成（実験的・上級者向け）",
        icon="auto_awesome",
    ).classes("full-width text-grey-6").props("dense"):
        ui.label(
            "⚠️ この機能は実験的です。外部の生成AIまたはローカルモデルを使って"
            "カスタム記述子のコードを自動生成できます。"
        ).classes("text-grey-6 text-caption q-mb-xs q-mt-xs")

        with ui.tabs().props("dense active-color=grey-7 indicator-color=grey-7 align=left") as _ai_tabs:
            _tab_ext = ui.tab("external", label="🌐 外部 生成AI", icon="language")
            _tab_int = ui.tab("internal", label="🤖 内部AI（ローカルモデル）", icon="smart_toy")

        with ui.tab_panels(_ai_tabs, value=_tab_ext).classes("full-width q-pt-sm"):

            # ══════════════════════════════════════════════
            # 🌐 外部AI タブ
            # ══════════════════════════════════════════════
            with ui.tab_panel(_tab_ext):
                _ext_state: dict = {
                    "library": "", "what": "", "output_type": "single",
                    "notes": "", "generated_prompt": "", "pasted_code": "",
                }

                ui.label(
                    "ChatGPT / Copilot / Claude / Gemini など、任意の生成AIに貼り付けるプロンプトを生成します。"
                    "生成されたコードをアプリに貼り付けるだけで記述子が使えるようになります。"
                ).classes("text-grey text-caption q-mb-sm")

                # ① 意図の入力
                with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
                    "border: 1px solid rgba(99,102,241,0.3); border-radius: 8px;"
                ):
                    ui.label("① 作りたい記述子を教えてください").classes("text-caption text-bold text-indigo q-mb-xs")

                    ui.input(
                        "使用ライブラリ（任意）",
                        placeholder="例: rdkit, padelpy, descriptastorus, molfeat（空欄＝独自実装）",
                        on_change=lambda e: _ext_state.update({"library": e.value}),
                    ).props("dense outlined").classes("full-width q-mb-xs").tooltip(
                        "空欄にした場合、外部ライブラリに依存しない独自の特徴量計算コードが生成されます。\n"
                        "特定のライブラリを指定すると、そのAPIを使ったコードが生成されます。"
                    )

                    # 目的変数名でデフォルトテキストを設定
                    _target_name = state.get("target_col", "")
                    _default_what = f"{_target_name} を予測するのに適した特徴量" if _target_name else ""
                    _ext_state["what"] = _default_what

                    ui.textarea(
                        "計算したい記述子・物性",
                        value=_default_what,
                        placeholder=(
                            "例: 屈折率を表すのにいい特徴量\n"
                            "例: TDA（位相的データ解析）を使った分子の持続的ホモロジー記述子\n"
                            "例: HOMO-LUMOギャップのエネルギー（eV単位）"
                        ),
                        on_change=lambda e: _ext_state.update({"what": e.value}),
                    ).props("outlined dense rows=3").classes("full-width q-mb-xs")

                    with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
                        ui.label("出力形式:").classes("text-caption text-grey")
                        ui.toggle(
                            {"single": "1つの値", "multi": "複数の値（1特徴量1関数）"},
                            value="single",
                            on_change=lambda e: _ext_state.update({"output_type": e.value}),
                        ).props("dense")

                    ui.input(
                        "追加の注意事項（任意）",
                        placeholder="例: NumPyのみ使用、インターネット接続不可、Windowsで動作必須",
                        on_change=lambda e: _ext_state.update({"notes": e.value}),
                    ).props("dense outlined").classes("full-width")

                # ② プロンプト生成
                _prompt_container = ui.column().classes("full-width")

                def _on_build_prompt():
                    from backend.llm.prompt_builder import DescriptorIntent, build_external_llm_prompt
                    intent = DescriptorIntent(
                        library=_ext_state.get("library", ""),
                        what_to_calc=_ext_state.get("what", ""),
                        output_type=_ext_state.get("output_type", "single"),
                        extra_notes=_ext_state.get("notes", ""),
                    )
                    if not intent.is_valid:
                        ui.notify("「計算したい記述子」を入力してください", type="warning")
                        return

                    prompt_text = build_external_llm_prompt(intent)
                    _ext_state["generated_prompt"] = prompt_text

                    _prompt_container.clear()
                    with _prompt_container:
                        with ui.card().classes("full-width q-pa-sm").style(
                            "border: 1px solid rgba(99,102,241,0.4); border-radius: 8px;"
                            "background: rgba(10,10,40,0.3);"
                        ):
                            with ui.row().classes("items-center justify-between q-mb-xs"):
                                ui.label("② 生成されたプロンプト").classes("text-caption text-indigo text-bold")
                                ui.button(
                                    "📋 コピー",
                                    on_click=lambda: (
                                        ui.run_javascript(
                                            f"navigator.clipboard.writeText({repr(prompt_text)})"
                                        ),
                                        ui.notify("プロンプトをコピーしました", type="positive", timeout=2000),
                                    ),
                                ).props("flat dense size=sm no-caps color=indigo")

                            ui.textarea(
                                value=prompt_text,
                            ).props("outlined dense rows=14 readonly").classes("full-width").style(
                                "font-family: monospace; font-size: 0.75rem;"
                                "background: rgba(0,0,0,0.2);"
                            )

                            ui.label(
                                "👆 このプロンプトをChatGPT / Copilot / Claude にそのまま貼り付けてください"
                            ).classes("text-caption text-grey q-mt-xs")

                ui.button(
                    "🔨 プロンプトを生成",
                    on_click=_on_build_prompt,
                ).props("unelevated no-caps color=indigo q-mb-sm")

                # ③ コードを貼り付けて保存
                ui.separator().classes("q-my-sm")
                with ui.card().classes("full-width q-pa-sm").style(
                    "border: 1px solid rgba(20,184,166,0.3); border-radius: 8px;"
                ):
                    ui.label("③ 生成されたコードを貼り付けて保存").classes("text-caption text-bold text-teal q-mb-xs")
                    ui.label(
                        "外部AIが出力したPythonコードをそのまま貼り付けてください。"
                        "保存前にセキュリティ検証・形式チェックを自動実行します。"
                    ).classes("text-grey text-caption q-mb-xs")

                    _paste_area = ui.textarea(
                        "ここにAI生成コードを貼り付け",
                        placeholder="DESCRIPTOR_NAME = \"...\"\ndef compute(smiles_list): ...",
                        on_change=lambda e: _ext_state.update({"pasted_code": e.value}),
                    ).props("outlined dense rows=10").classes("full-width q-mb-sm").style(
                        "font-family: monospace; font-size: 0.8rem;"
                    )

                    _paste_result = ui.column().classes("full-width")

                    # よくある失敗パターンのヘルプ
                    with ui.expansion("❓ 読み込み失敗した場合のチェックリスト", icon="help_outline").classes(
                        "full-width q-mb-sm"
                    ).style("border: 1px solid rgba(99,102,241,0.2); border-radius: 8px;"):
                        _problems = [
                            ("🚫 コードブロック記号が残っている", "AIが ``` python ... ``` の形式で出力した場合。→ 自動除去します"),
                            ("🚫 DESCRIPTOR_NAME が未定義",     "先頭に DESCRIPTOR_NAME = '英語名' を追加してください"),
                            ("🚫 compute() 関数がない",         "def compute(smiles_list): ... を実装してください"),
                            ("🚫 import os / subprocess を使用", "外部コマンド実行は禁止です。RDKit等のライブラリ内APIを使用してください"),
                            ("⚠️ 戻り値がリストでない",          "compute()は list[float|None] か pd.DataFrame を返す必要があります"),
                            ("⚠️ 戻り値の長さが入力と異なる",    "len(result) == len(smiles_list) になるよう、各SMILES1件に1値を返してください"),
                            ("⚠️ エラーが例外を投げる",          "try/except で囲み、失敗した分子は None を返してください"),
                        ]
                        with ui.column().classes("q-gutter-xs q-pa-xs"):
                            for prob, fix in _problems:
                                with ui.row().classes("items-start q-gutter-sm"):
                                    ui.label(prob).classes("text-body2 col-5")
                                    ui.label(fix).classes("text-caption text-grey col-7")

                    def _on_validate_save():
                        from backend.llm.generator import _check_security, _validate_code_format, _strip_code_fences
                        from backend.chem.descriptors import get_custom_dir, invalidate_cache
                        from backend.chem.descriptors.base import validate_plugin, safe_compute
                        import re as _re, importlib.util as _ilu, tempfile, pathlib

                        raw_code = _ext_state.get("pasted_code", "").strip()
                        if not raw_code:
                            ui.notify("コードを貼り付けてください", type="warning")
                            return

                        # ── STEP 1: コードブロック記号を自動除去 ──────────────────
                        code = _strip_code_fences(raw_code)
                        stripped = (code != raw_code)

                        _paste_result.clear()
                        with _paste_result:
                            if stripped:
                                with ui.row().classes("items-center q-gutter-xs q-mb-xs"):
                                    ui.icon("auto_fix_high", color="cyan").classes("text-body2")
                                    ui.label("```python ... ``` を自動除去しました").classes("text-caption text-cyan")

                            # ── STEP 2: セキュリティチェック ──────────────────────
                            warns = _check_security(code)
                            blocked = [w for w in warns if "[BLOCKED]" in w]
                            if blocked:
                                with ui.card().classes("full-width q-pa-sm").style(
                                    "border: 1px solid rgba(239,68,68,0.6); border-radius:8px;"
                                ):
                                    ui.label("🚫 セキュリティ検証 — 保存ブロック").classes("text-red text-body2 text-bold")
                                    for b in blocked:
                                        ui.label(f"  • {b.replace('[BLOCKED] ', '')}").classes("text-caption text-red")
                                    with ui.expansion("💡 対策方法", icon="lightbulb").classes("q-mt-xs"):
                                        ui.label(
                                            "外部コマンド実行（os.system等）・ファイルI/O（open）・eval/execは禁止です。\n"
                                            "RDKit・NumPy・pandas のような科学計算ライブラリのAPIのみ使用可能です。\n"
                                            "プロンプトに「os, subprocess, eval, exec を一切使用しないこと」と追記して再生成してください。"
                                        ).classes("text-caption text-grey").style("white-space: pre-line;")
                                return

                            # ── STEP 3: 形式チェック ──────────────────────────────
                            plugin_name, ferr = _validate_code_format(code)
                            if ferr:
                                with ui.card().classes("full-width q-pa-sm").style(
                                    "border: 1px solid rgba(251,191,36,0.5); border-radius:8px;"
                                ):
                                    ui.label(f"⚠️ 形式チェック失敗: {ferr}").classes("text-amber text-body2 text-bold")
                                    with ui.expansion("💡 修正方法", icon="lightbulb").classes("q-mt-xs"):
                                        ui.code(
                                            "DESCRIPTOR_NAME = \"MyDescriptor\"     # ← 必須\n"
                                            "DESCRIPTOR_CATEGORY = \"物理化学\"      # ← 必須\n"
                                            "DESCRIPTOR_ENGINE = \"RDKit\"           # ← 必須\n\n"
                                            "def compute(smiles_list: list[str]) -> list[float | None]:  # ← 必須\n"
                                            "    results = []\n"
                                            "    for smi in smiles_list:\n"
                                            "        try:\n"
                                            "            results.append(1.0)  # ← 計算ロジック\n"
                                            "        except Exception:\n"
                                            "            results.append(None)\n"
                                            "    return results"
                                        ).classes("text-caption")
                                return

                            # ── STEP 4: ランタイムテスト（サンプルSMILESで実際に実行）──
                            _TEST_SMILES = [
                                "CC(=O)Oc1ccccc1C(=O)O",  # アスピリン
                                "c1ccc(cc1)N",             # アニリン
                                "INVALID_SMILES_TEST",     # 壊れたSMILES（Noneを返すべき）
                                "C",                       # メタン
                            ]
                            runtime_ok = True
                            runtime_msg = ""
                            runtime_result = None

                            try:
                                # 一時ファイルを使ってモジュールとしてロード
                                tmp = pathlib.Path(tempfile.mktemp(suffix=".py"))
                                tmp.write_text(code, encoding="utf-8")
                                spec = _ilu.spec_from_file_location("__ext_test__", tmp)
                                mod = _ilu.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                tmp.unlink(missing_ok=True)

                                info = validate_plugin(mod, "<test>")
                                if info is None:
                                    runtime_ok = False
                                    runtime_msg = "validate_plugin が None を返しました（形式チェック参照）"
                                else:
                                    runtime_result = safe_compute(info, _TEST_SMILES)

                                    # 戻り値型チェック
                                    import pandas as _pd, numpy as _np
                                    if isinstance(runtime_result, _pd.DataFrame):
                                        if len(runtime_result) != len(_TEST_SMILES):
                                            runtime_ok = False
                                            runtime_msg = (
                                                f"DataFrame の行数({len(runtime_result)})が"
                                                f"入力SMILES数({len(_TEST_SMILES)})と一致しません。\n"
                                                "各SMILESに1行対応するよう rows.append() を全ループで実行してください。"
                                            )
                                    elif isinstance(runtime_result, (list, _np.ndarray)):
                                        if len(runtime_result) != len(_TEST_SMILES):
                                            runtime_ok = False
                                            runtime_msg = (
                                                f"戻り値の長さ({len(runtime_result)})が"
                                                f"入力SMILES数({len(_TEST_SMILES)})と一致しません。\n"
                                                "ループで各SMILESに1値を results.append() してから return results してください。"
                                            )
                                    elif runtime_result is None or not hasattr(runtime_result, "__len__"):
                                        runtime_ok = False
                                        runtime_msg = (
                                            f"戻り値の型が不正です: {type(runtime_result).__name__}\n"
                                            "compute() は list[float|None] または pd.DataFrame を返す必要があります。"
                                        )
                            except Exception as ex:
                                runtime_ok = False
                                runtime_msg = f"実行時エラー: {ex}"

                            if not runtime_ok:
                                with ui.card().classes("full-width q-pa-sm").style(
                                    "border: 1px solid rgba(239,68,68,0.5); border-radius:8px;"
                                ):
                                    ui.label("🔴 ランタイムテスト失敗").classes("text-red text-body2 text-bold")
                                    ui.label(runtime_msg).classes("text-caption text-red").style("white-space: pre-line;")
                                    with ui.expansion("💡 対策方法", icon="lightbulb").classes("q-mt-xs"):
                                        ui.label(
                                            "外部AIにこのエラーメッセージをそのままフィードバックして修正を依頼してください:\n\n"
                                            f"「以下のエラーが発生しました。修正してください:\n{runtime_msg}」"
                                        ).classes("text-caption text-grey").style("white-space: pre-line;")
                                return

                            # ── STEP 5: 全チェック通過 → 保存 ────────────────────
                            safe_name = _re.sub(r"[^\w]", "_", plugin_name.lower())[:40]
                            save_path = get_custom_dir() / f"ext_{safe_name}.py"
                            i = 2
                            while save_path.exists():
                                save_path = get_custom_dir() / f"ext_{safe_name}_{i}.py"
                                i += 1
                            save_path.write_text(code, encoding="utf-8")
                            invalidate_cache()

                            with ui.card().classes("full-width q-pa-sm").style(
                                "border: 1px solid rgba(74,222,128,0.6); border-radius:8px;"
                            ):
                                with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
                                    ui.icon("check_circle", color="green").classes("text-h6")
                                    ui.label(f"✅ 登録完了: {plugin_name}").classes("text-green text-body2 text-bold")

                                # ランタイムテスト結果サマリー
                                import pandas as _pd2
                                if isinstance(runtime_result, _pd2.DataFrame):
                                    ui.label(
                                        f"テスト計算: {runtime_result.shape[1]}列の記述子 × {runtime_result.shape[0]}分子"
                                    ).classes("text-caption text-grey")
                                else:
                                    ok_count = sum(1 for v in runtime_result if v is not None)
                                    ui.label(
                                        f"テスト計算: {ok_count}/{len(_TEST_SMILES)}分子 正常（NULLは壊れたSMILES分）"
                                    ).classes("text-caption text-grey")

                                ui.label(f"保存先: {save_path.name}").classes("text-caption text-grey")
                                if warns:
                                    ui.label(f"⚠️ 非致命的警告: {len(warns)}件").classes("text-caption text-amber")

                            ui.notify(f"✅ {save_path.name} — テスト済みで保存しました", type="positive")

                    _ext_review_area = ui.column().classes("full-width q-mt-xs")

                    with ui.row().classes("q-gutter-sm items-center q-mt-sm"):
                        ui.button(
                            "\U0001f50d 検証して保存（ランタイムテスト付き）",
                            on_click=_on_validate_save,
                        ).props("unelevated no-caps color=teal")

                        async def _on_ext_review():
                            raw_code = _ext_state.get("pasted_code", "").strip()
                            if not raw_code:
                                ui.notify("レビュー対象のコードを貼り付けてください", type="warning")
                                return
                            from frontend_nicegui.components.internal_llm_ui import (
                                _do_review,
                            )
                            from backend.llm.providers.hf_provider import load_hf_config
                            cfg = load_hf_config()
                            mid = cfg.get("model_id", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
                            intent = _ext_state.get("what", "")
                            await _do_review(raw_code, intent, mid, _ext_review_area)

                        ui.button(
                            "🔍 LLMでレビュー",
                            on_click=lambda: __import__("asyncio").ensure_future(_on_ext_review()),
                        ).props("unelevated no-caps color=purple")

            # ══════════════════════════════════════════════
            # 🤖 内部AI（HuggingFace ローカルLLM）タブ
            # ══════════════════════════════════════════════
            with ui.tab_panel(_tab_int):
                try:
                    from frontend_nicegui.components.internal_llm_ui import render_internal_llm_tab
                    render_internal_llm_tab(state)
                except Exception as _hf_err:
                    ui.label(
                        f"内部LLMモジュールの読み込みに失敗しました: {_hf_err}"
                    ).classes("text-red text-caption")

def _detect_adapter_sources() -> dict[str, Path]:
    """backend/chem/以下のSMILES記述子アダプタのソースファイルを検出。"""
    import inspect
    adapters = {}
    adapter_dir = Path(__file__).resolve().parent.parent.parent / "backend" / "chem"
    if not adapter_dir.exists():
        return adapters

    # 既知のアダプタファイル
    known = [
        ("RDKit", "rdkit_adapter.py"),
        ("Mordred", "mordred_adapter.py"),
        ("XTB", "xtb_adapter.py"),
        ("COSMO-RS", "cosmo_adapter.py"),
        ("Uni-pKa", "unipka_adapter.py"),
        ("UMA", "uma_adapter.py"),
        ("scikit-FP", "scikitfp_adapter.py"),
        ("MolAI", "molai_adapter.py"),
    ]
    for name, filename in known:
        filepath = adapter_dir / filename
        if filepath.exists():
            adapters[name] = filepath

    return adapters


def _copy_adapter_to_custom(adapter_name: str, source_path: Path, custom_dir: Path) -> None:
    """既存アダプタのソースをカスタムプラグインとしてコピー。"""
    from backend.chem.descriptors import invalidate_cache

    dest_name = f"custom_{source_path.stem}.py"
    dest = custom_dir / dest_name

    if dest.exists():
        ui.notify(f"⚠️ {dest_name} は既に存在します。名前を変えてください。", type="warning")
        return

    try:
        # ヘッダーコメントを追加
        original_source = source_path.read_text(encoding="utf-8")
        header = (
            f'"""\n'
            f"カスタムプラグイン: {adapter_name}アダプタから作成\n"
            f"元ファイル: {source_path.name}\n"
            f"\n"
            f"このファイルを自由に編集して、独自の記述子計算ロジックを実装してください。\n"
            f"クラス名やcompute()メソッドのシグネチャを変更すると読み込まれなくなるため注意。\n"
            f'"""\n\n'
        )
        dest.write_text(header + original_source, encoding="utf-8")
        invalidate_cache()
        ui.notify(f"✅ {dest_name} を作成しました。編集してカスタマイズしてください。", type="positive")
    except Exception as e:
        ui.notify(f"⚠️ コピーエラー: {e}", type="warning")


def _show_source_dialog(filepath: Path) -> None:
    """ソースコードを表示するダイアログ。"""
    try:
        source = filepath.read_text(encoding="utf-8")
    except Exception:
        source = "(読み込めません)"

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("width: 80vw; max-width: 900px;"):
        ui.label(f"📄 {filepath.name}").classes("text-h6")
        # NiceGUIのcode要素でソースコード表示
        ui.code(source, language="python").classes("full-width").style(
            "max-height: 60vh; overflow-y: auto; font-size: 0.8rem;"
        )
        ui.button("閉じる", on_click=dlg.close).props("outline color=cyan")
    dlg.open()


def _delete_custom(filepath: Path) -> None:
    """カスタムプラグインを削除。"""
    try:
        filepath.unlink()
        from backend.chem.descriptors import invalidate_cache
        invalidate_cache()
        ui.notify(f"🗑️ {filepath.name} を削除しました", type="info")
    except Exception as e:
        ui.notify(f"⚠️ 削除エラー: {e}", type="warning")


# ═══════════════════════════════════════════════════════════
# アダプタパラメータ動的UI
# ═══════════════════════════════════════════════════════════
def _render_adapter_params(adapters: dict, state: dict) -> None:
    """
    各SMILESアダプタのパラメータを introspect_params で自動検出し、
    動的UIを生成する。パラメータが0個のエンジンはスキップ。
    """
    if "adapter_params" not in state:
        state["adapter_params"] = {}

    ui.label(
        "各エンジンの引数を自動検出して表示しています。"
        "変更しなければデフォルト設定で計算されます。"
    ).classes("text-caption text-grey q-mb-sm")

    # data_tab._ALL_ENGINES の (name, mod_path, cls_name, kwargs) タプル形式に対応
    # descriptor_plugins_ui._ENGINE_INFO の dict 形式にも対応
    _ALL_ENGINES_COMPAT = [
        ("RDKit",           "backend.chem.rdkit_adapter",           "RDKitAdapter",           {}),
        ("Mordred",         "backend.chem.mordred_adapter",         "MordredAdapter",         {}),
        ("GroupContrib",    "backend.chem.group_contrib_adapter",   "GroupContribAdapter",    {}),
        ("DescriptaStorus", "backend.chem.descriptastorus_adapter", "DescriptaStorusAdapter", {}),
        ("MolAI",           "backend.chem.molai_adapter",           "MolAIAdapter",           {}),
        ("scikit-FP",       "backend.chem.skfp_adapter",            "SkfpAdapter",            {}),
        ("XTB",             "backend.chem.xtb_adapter",             "XTBAdapter",             {}),
        ("Mol2Vec",         "backend.chem.mol2vec_adapter",         "Mol2VecAdapter",         {}),
        ("Molfeat",         "backend.chem.molfeat_adapter",         "MolfeatAdapter",         {}),
    ]

    any_shown = False
    for name, mod_path, cls_name, _kwargs in _ALL_ENGINES_COMPAT:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            adapter_cls = getattr(mod, cls_name)

            from backend.ui.param_schema import introspect_params
            specs = introspect_params(adapter_cls)
            if not specs:
                continue  # パラメータなしのエンジンはスキップ

            any_shown = True
            with ui.expansion(
                f"🔹 {name} ({len(specs)}パラメータ)", icon="settings",
            ).classes("full-width q-mb-xs"):
                try:
                    from frontend_nicegui.components.auto_params_ui import render_param_editor
                    existing = state["adapter_params"].get(name, {})
                    values = render_param_editor(
                        specs, title=name, values=existing, compact=True,
                    )
                    state["adapter_params"][name] = values
                except Exception as ex:
                    ui.label(f"⚠️ {ex}").classes("text-amber")
        except Exception:
            pass  # インポート不可のエンジンはスキップ

    if not any_shown:
        ui.label("設定可能なパラメータを持つエンジンはありません").classes("text-grey text-caption")


