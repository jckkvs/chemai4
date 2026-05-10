"""
frontend_nicegui/components/llm_analysis_dialog.py

LLM対話型分析ダイアログ — データ読込後に起動し、
ユーザーと対話しながら分析方針を決定する。

主な決定事項:
- 表データの場合: どの変数を単調性制約をかけるか
- SMILESデータの場合: どの特徴量(記述子)を使うか
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from nicegui import ui, app

logger = logging.getLogger(__name__)

# グローバル状態
_dialog_instance = None


def trigger_llm_analysis(state: dict, force_new: bool = False) -> None:
    """
    LLM分析ダイアログを起動する。

    Args:
        state: アプリケーション状態
        force_new: True の場合、新しい対話を開始する
    """
    global _dialog_instance

    df = state.get("df")
    if df is None or df.empty:
        ui.notify("先にデータを読み込んでください", type="warning")
        return

    if _dialog_instance is None:
        _dialog_instance = LLMAnalysisDialog(state)

    _dialog_instance.open(force_new=force_new)


class LLMAnalysisDialog:
    """
    LLM対話型分析ダイアログ。

    機能:
    1. データ要約の表示
    2. LLMの分析結果表示
    3. 対話型Q&A
    4. 主決定事項の選択UI
    5. 決定内容のstateへの適用
    """

    def __init__(self, state: dict):
        self.state = state
        self.dialog = None
        self.conversation_container = None
        self.main_decision_container = None
        self.user_input = None
        self.analysis_result = None
        self.suggestions = {}
        self.is_smiles_data = False

    def open(self, force_new: bool = False) -> None:
        """ダイアログを開く。"""
        if self.dialog is None:
            self._build_dialog()

        self.dialog.open()

        # 初期分析を実行
        self._run_initial_analysis(force_new=force_new)

    def _build_dialog(self) -> None:
        """ダイアログのUIを構築する。"""
        with ui.dialog() as self.dialog:
            with ui.card().classes("full-width").style(
                "max-width: 900px; width: 90vw; max-height: 85vh;"
                "display: flex; flex-direction: column;"
            ):
                # ヘッダー
                with ui.row().classes("items-center justify-between full-width q-pa-sm"):
                    ui.label("🤖 LLMデータ分析アシスタント").classes(
                        "text-h6 text-bold hero-gradient"
                    )
                    ui.button(
                        icon="close",
                        on_click=lambda: self.dialog.close(),
                    ).props("flat round dense").style("margin-left: auto;")

                # データ種別バッジ
                self.type_badge = ui.badge("", color="blue").props("outline")
                self.type_badge.style("margin-left: 8px;")

                ui.separator()

                # スクロール可能なコンテンツ領域
                with ui.scroll_area().classes("full-width flex-grow").style(
                    "min-height: 400px; max-height: 60vh;"
                ):
                    # データ要約セクション
                    with ui.expansion("📊 データ要約", icon="info").classes(
                        "full-width q-mb-sm"
                    ).props("default-opened"):
                        self.summary_container = ui.column().classes("full-width")

                    ui.separator()

                    # ドキュメント内容セクション（ドキュメントがある場合のみ）
                    self.document_container = ui.column().classes("full-width")
                    ui.separator()

                    # LLM分析結果セクション
                    with ui.expansion("🤖 LLM分析結果", icon="psychology").classes(
                        "full-width q-mb-sm"
                    ).props("default-opened"):
                        self.analysis_container = ui.column().classes("full-width")

                    ui.separator()

                    # 主決定事項セクション
                    with ui.expansion("🎯 主決定事項", icon="gavel").classes(
                        "full-width q-mb-sm"
                    ).props("default-opened"):
                        self.main_decision_container = ui.column().classes("full-width")

                    ui.separator()

                    # 対話履歴セクション
                    with ui.expansion("💬 対話履歴", icon="chat").classes(
                        "full-width q-mb-sm"
                    ):
                        self.conversation_container = ui.column().classes("full-width")

                # 入力エリア（フッター固定）
                ui.separator()
                with ui.row().classes("items-center full-width q-pa-sm"):
                    self.user_input = ui.textarea(
                        placeholder="LLMに質問・补充情報を入力してください..."
                    ).classes("flex-grow").props("outlined dense autogrow rows=2")
                    ui.button(
                        "送信",
                        icon="send",
                        on_click=self._on_send_message,
                    ).props("unelevated no-caps color=cyan dense")

                # アクションボタン
                ui.separator()
                with ui.row().classes("items-center justify-between full-width q-pa-sm"):
                    with ui.row().classes("q-gutter-sm"):
                        self.apply_btn = ui.button(
                            "✅ 決定内容を適用",
                            on_click=self._on_apply,
                        ).props("unelevated no-caps color=positive dense")
                        self.apply_btn.disable()  # 初期は無効

                    ui.button(
                        "🔄 再分析",
                        on_click=lambda: self._run_initial_analysis(force_new=True),
                    ).props("flat no-caps color=grey dense")

    async def _run_initial_analysis(self, force_new: bool = False) -> None:
        """初期分析を実行する。"""
        df = self.state.get("df")
        if df is None:
            return

        # データ種別を判定
        smiles_col = self.state.get("smiles_col", "")
        self.is_smiles_data = bool(smiles_col and smiles_col in df.columns)

        # バッジ更新
        if self.is_smiles_data:
            self.type_badge.set_text("SMILESデータ")
            self.type_badge.props("color=purple")
        else:
            self.type_badge.set_text("表(タブラー)データ")
            self.type_badge.props("color=blue")

        # 要約表示
        self.summary_container.clear()
        with self.summary_container:
            ui.label(f"行数: {len(df):,}行 × {len(df.columns)}列").classes(
                "text-caption"
            )
            if self.is_smiles_data:
                ui.label(f"SMILES列: {smiles_col}").classes("text-caption")
            target_col = self.state.get("target_col", "")
            if target_col:
                ui.label(f"目的変数: {target_col}").classes("text-caption")

        # ドキュメント内容表示
        self.document_container.clear()
        document_text = self.state.get("document_text", "")
        document_meta = self.state.get("document_metadata", {})
        document_filename = self.state.get("document_filename", "")

        if document_text:
            with self.document_container:
                with ui.expansion(f"📄 ドキュメント: {document_filename}", icon="description").classes(
                    "full-width q-mb-sm"
                ).props("default-opened"):
                    # メタデータ表示
                    if document_meta:
                        meta_parts = []
                        if "paragraphs" in document_meta:
                            meta_parts.append(f"段落数: {document_meta['paragraphs']}")
                        if "slides" in document_meta:
                            meta_parts.append(f"スライド数: {document_meta['slides']}")
                        if "pages" in document_meta:
                            meta_parts.append(f"ページ数: {document_meta['pages']}")
                        if "tables" in document_meta:
                            meta_parts.append(f"テーブル数: {document_meta['tables']}")
                        if meta_parts:
                            ui.label(", ".join(meta_parts)).classes("text-caption")

                    # テキスト要約（最初の500文字）
                    preview_text = document_text[:500]
                    if len(document_text) > 500:
                        preview_text += "..."
                    ui.label("内容プレビュー:").classes("text-caption text-bold")
                    ui.label(preview_text).classes("text-caption")
                    ui.label(f"全文字数: {len(document_text)}文字").classes("text-caption text-grey")

        # LLM分析実行
        self.analysis_container.clear()
        with self.analysis_container:
            ui.label("⏳ LLMが分析中...").classes("text-grey-5 q-pa-md")
            ui.spinner("dots", size="lg").style("margin: 0 auto; display: block;")

        # バックグラウンドでLLM呼び出し
        try:
            from backend.llm.data_analyst import get_data_analyst

            analyst = get_data_analyst()

            # 非同期でLLMを呼び出す
            from nicegui import run as ng_run

            def _do_analysis():
                return analyst.analyze(self.state, reset=force_new)

            result = await ng_run.io_bound(_do_analysis)

            self.analysis_result = result
            self.suggestions = result.get("suggestions", {})

            # 分析結果を表示
            self.analysis_container.clear()
            with self.analysis_container:
                reply = result.get("reply", "")
                if reply:
                    # テキストをMarkdownとして表示
                    ui.markdown(self._format_reply(reply)).classes("text-body2")

                # 対話履歴を表示
                if self.conversation_container:
                    self.conversation_container.clear()
                    with self.conversation_container:
                        for msg in result.get("conversation", []):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role == "user":
                                with ui.card().classes(
                                    "glass-card q-pa-xs q-mb-xs"
                                ).style(
                                    "background: rgba(0, 212, 255, 0.05);"
                                    "margin-left: 20px;"
                                ):
                                    ui.label("👤 あなた").classes(
                                        "text-caption text-bold text-cyan"
                                    )
                                    ui.label(content).classes("text-caption")
                            elif role == "assistant":
                                with ui.card().classes(
                                    "glass-card q-pa-xs q-mb-xs"
                                ).style(
                                    "background: rgba(123, 47, 247, 0.05);"
                                    "margin-right: 20px;"
                                ):
                                    ui.label("🤖 LLM").classes(
                                        "text-caption text-bold text-purple"
                                    )
                                    ui.label(content).classes("text-caption")

                # 主決定事項UIを構築
                self._build_main_decision_ui()

        except Exception as e:
            logger.exception(f"[LLMAnalysisDialog] Analysis error: {e}")
            self.analysis_container.clear()
            with self.analysis_container:
                ui.label(f"⚠️ 分析エラー: {e}").classes("text-red q-pa-md")

    def _build_main_decision_ui(self) -> None:
        """主決定事項のUIを構築する。"""
        if self.main_decision_container is None:
            return

        self.main_decision_container.clear()

        with self.main_decision_container:
            if self.is_smiles_data:
                self._build_smiles_decisions()
            else:
                self._build_tabular_decisions()

    def _build_smiles_decisions(self) -> None:
        """SMILESデータの主決定: 記述子選択。"""
        ui.label("どの記述子(特徴量)を使用しますか？").classes(
            "text-subtitle2 q-mb-sm"
        )
        ui.label(
            "SMILESデータでは、使用する記述子エンジンを選択します。\n"
            "LLMの推奨に基づき、以下から選択してください。"
        ).classes("text-caption text-grey-6 q-mb-md")

        # 利用可能な記述子エンジン
        all_descriptors = [
            ("RDKit", "基本物理化学記述子 + フィンガープリント", True),
            ("Mordred", "1,800+ QSAR記述子（2Dトポロジカル）", True),
            ("GroupContrib", "基団寄与法による熱物性推定", True),
            ("XTB", "GFN2-xTB 量子化学記述子", False),
            ("UniPka", "pKa / LogD / 溶媒和エネルギー", False),
            ("COSMO-RS", "σ-プロファイルによる溶媒和自由エネルギー", False),
            ("MolAI", "CNN + PCA 分子潜在空間", False),
            ("scikit-FP", "ECFP, MACCS等 フィンガープリント", True),
            ("Mol2Vec", "Word2Vec分子埋め込み", False),
            ("PaDEL", "1,800+記述子 (Java必要)", False),
        ]

        # LLMの推奨を取得
        recommended = self.suggestions.get("selected_descriptors", [])

        # 選択状態を保持
        self.selected_descriptors = list(recommended) if recommended else ["RDKit", "GroupContrib"]

        with ui.column().classes("full-width q-gutter-xs"):
            for name, desc, available in all_descriptors:
                checked = name in self.selected_descriptors

                def _make_toggle(n=name, d=desc, a=available):
                    return lambda e, n=n, d=d, a=a: self._toggle_descriptor(n, d, a)

                row_classes = "items-center full-width q-pa-xs"
                if not available:
                    row_classes += " opacity-50"

                with ui.row().classes(row_classes):
                    cb = ui.checkbox(
                        name,
                        value=checked,
                        on_change=_make_toggle(),
                    ).props("dense")
                    if not available:
                        cb.props("disable")
                    ui.label(d).classes("text-caption text-grey-6")

        # LLMの推奨表示
        if recommended:
            ui.separator().classes("q-my-sm")
            ui.label("🤖 LLMの推奨:").classes("text-caption text-bold text-purple")
            with ui.row().classes("q-gutter-xs flex-wrap"):
                for desc in recommended:
                    ui.chip(desc, icon="check_circle").props(
                        "outline color=purple size=sm"
                    )

        # 適用ボタンを有効化
        self.apply_btn.enable()

    def _toggle_descriptor(self, name: str, desc: str, available: bool) -> None:
        """記述子の選択/解除を切り替える。"""
        if not available:
            ui.notify(f"{name} はインストールされていません", type="warning")
            return

        if name in self.selected_descriptors:
            self.selected_descriptors.remove(name)
        else:
            self.selected_descriptors.append(name)

    def _build_tabular_decisions(self) -> None:
        """表データの主決定: 単調性制約。"""
        ui.label("どの変数に単調性制約をかけますか？").classes(
            "text-subtitle2 q-mb-sm"
        )
        ui.label(
            "単調性制約を設定すると、モデルの予測がその変数に対して\n"
            "単調増加または単調減少になるように制約されます。\n"
            "XGBoost, LightGBM, カーネルモデルで使用可能です。"
        ).classes("text-caption text-grey-6 q-mb-md")

        df = self.state.get("df")
        if df is None:
            return

        target_col = self.state.get("target_col", "")
        exclude_cols = set(self.state.get("exclude_cols", []))
        if target_col:
            exclude_cols.add(target_col)

        # 数値列を取得
        numeric_cols = [
            c for c in df.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not numeric_cols:
            ui.label("数値列がありません").classes("text-grey-5 q-pa-md")
            return

        # LLMの推奨を取得
        recommended = self.suggestions.get("monotonic_constraints", {})

        # 選択状態を保持（辞書: {列名: 方向}）
        self.monotonic_constraints = dict(recommended) if recommended else {}

        ui.label("各変数の制約方向を選択してください:").classes(
            "text-caption q-mb-sm"
        )

        for col in numeric_cols:
            current = self.monotonic_constraints.get(col, 0)
            col_stats = df[col]

            with ui.card().classes("glass-card q-pa-xs q-mb-xs full-width"):
                with ui.row().classes("items-center full-width"):
                    ui.label(col).classes("text-body2 text-bold").style(
                        "width: 150px;"
                    )

                    # 統計情報
                    stats_text = (
                        f"範囲: [{col_stats.min():.2f}, {col_stats.max():.2f}]"
                    )
                    ui.label(stats_text).classes("text-caption text-grey-6").style(
                        "width: 200px;"
                    )

                    # ラジオボタンで方向選択
                    def _make_handler(c=col):
                        return lambda e, c=c: self._set_monotonicity(c, e.value)

                    ui.radio(
                        {0: "制約なし", 1: "↗ 単調増加", -1: "↘ 単調減少"},
                        value=current,
                        on_change=_make_handler(),
                    ).props("dense inline")

        # LLMの推奨表示
        if recommended:
            ui.separator().classes("q-my-sm")
            ui.label("🤖 LLMの推奨:").classes("text-caption text-bold text-purple")

            for col, direction in recommended.items():
                direction_label = "↗ 単調増加" if direction == 1 else "↘ 単調減少" if direction == -1 else "制約なし"
                ui.label(f"  • {col}: {direction_label}").classes(
                    "text-caption"
                )

        # 適用ボタンを有効化
        self.apply_btn.enable()

    def _set_monotonicity(self, col: str, value: int) -> None:
        """単調性制約を設定する。"""
        if value == 0:
            if col in self.monotonic_constraints:
                del self.monotonic_constraints[col]
        else:
            self.monotonic_constraints[col] = value

    async def _on_send_message(self) -> None:
        """ユーザーのメッセージを送信する。"""
        if self.user_input is None:
            return

        message = self.user_input.value.strip()
        if not message:
            return

        # 入力をクリア
        self.user_input.value = ""

        # 対話履歴に追加
        if self.conversation_container:
            with self.conversation_container:
                with ui.card().classes("glass-card q-pa-xs q-mb-xs").style(
                    "background: rgba(0, 212, 255, 0.05); margin-left: 20px;"
                ):
                    ui.label("👤 あなた").classes("text-caption text-bold text-cyan")
                    ui.label(message).classes("text-caption")

        # LLMに送信
        try:
            from backend.llm.data_analyst import get_data_analyst

            analyst = get_data_analyst()

            from nicegui import run as ng_run

            def _do_send():
                return analyst.analyze(self.state, user_message=message)

            result = await ng_run.io_bound(_do_send)

            self.analysis_result = result
            self.suggestions = result.get("suggestions", {})

            # LLMの返信を表示
            if self.conversation_container:
                with self.conversation_container:
                    with ui.card().classes("glass-card q-pa-xs q-mb-xs").style(
                        "background: rgba(123, 47, 247, 0.05); margin-right: 20px;"
                    ):
                        ui.label("🤖 LLM").classes(
                            "text-caption text-bold text-purple"
                        )
                        reply = result.get("reply", "")
                        ui.markdown(self._format_reply(reply)).classes("text-caption")

            # 主決定事項UIを更新
            self._build_main_decision_ui()

        except Exception as e:
            logger.exception(f"[LLMAnalysisDialog] Send message error: {e}")
            ui.notify(f"送信エラー: {e}", type="negative")

    def _on_apply(self) -> None:
        """決定内容をstateに適用する。"""
        try:
            from backend.llm.data_analyst import get_data_analyst

            analyst = get_data_analyst()

            # 選択内容をsuggestionsに反映
            if self.is_smiles_data:
                self.suggestions["selected_descriptors"] = self.selected_descriptors
            else:
                self.suggestions["monotonic_constraints"] = self.monotonic_constraints

            # stateに適用
            applied = analyst.apply_suggestions(self.state, self.suggestions)

            # 成功通知
            if applied:
                ui.notify(
                    f"✅ 適用完了: {', '.join(applied)}",
                    type="positive",
                    timeout=4000,
                )
            else:
                ui.notify("適用する設定がありません", type="warning")

            # ダイアログを閉じる
            self.dialog.close()

        except Exception as e:
            logger.exception(f"[LLMAnalysisDialog] Apply error: {e}")
            ui.notify(f"適用エラー: {e}", type="negative")

    def _format_reply(self, reply: str) -> str:
        """LLMの返信をフォーマットする。"""
        # JSONブロックを除去（別途表示するため）
        import re
        reply = re.sub(r"```json\s*.*?\s*```", "", reply, flags=re.DOTALL)
        # コードブロックをMarkdownで表示
        reply = reply.replace("```", "```\n")
        return reply


def render_llm_analysis_button(state: dict) -> None:
    """LLM分析ボタンを描画する。"""
    df = state.get("df")
    if df is None or df.empty:
        return

    ui.button(
        "🤖 LLMと分析方針を決める",
        icon="psychology",
        on_click=lambda: trigger_llm_analysis(state),
    ).props("outline color=purple no-caps").tooltip(
        "データをLLMが分析し、対話的に分析方針を決定します"
    )
