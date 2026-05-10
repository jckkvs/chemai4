"""
frontend_nicegui/ui/descriptor_recommendation_dialog.py

記述子推奨ダイアログ - LLMを使用して目的変数に適した記述子を推奨する。
化学物性の知見に基づいた推奨を行う。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

try:
    from nicegui import ui, events, app
    from nicegui.events import ClickEventArguments
    NICE_GUI_AVAILABLE = True
except ImportError:
    NICE_GUI_AVAILABLE = False

logger = logging.getLogger(__name__)


class DescriptorRecommendationDialog:
    """
    LLMを使用して記述子を推奨するダイアログ。
    ユーザーとのヒヤリング（対話）を通じて最適な記述子を推奨する。
    """

    def __init__(self, state: dict, on_apply: Callable[[list[str]], Any] | None = None):
        """
        Args:
            state: アプリケーション状態（df, target_col, smiles_col等）
            on_apply: 推奨記述子を適用する際のコールバック
        """
        self.state = state
        self.on_apply = on_apply
        self.dialog = None
        self.target_desc_input = None
        self.interview_notes_input = None
        self.result_container = None
        self.conversation_container = None
        self.recommended_descriptors: list[dict] = []
        self.conversation_history: list[dict] = []
        self.interview_mode = False
        self.interview_history: list[dict] = []

    def create_dialog(self):
        """ダイアログを作成する。"""
        if not NICE_GUI_AVAILABLE:
            logger.error("NiceGUI is not available")
            return

        with ui.dialog() as dialog:
            self.dialog = dialog
            with ui.card().classes("w-full max-w-2xl"):
                ui.label("記述子推奨 (LLM)").classes("text-h5 font-bold mb-4")

                # タブ切り替え
                with ui.tabs() as tabs:
                    interview_tab = ui.tab("ヒヤリングモード")
                    direct_tab = ui.tab("直接推奨")

                # ヒヤリングモード
                with interview_tab:
                    ui.label("LLMが質問を投げかけ、対話を通じて記述子を推奨します。").classes("text-caption text-gray mb-2")

                    with ui.row():
                        ui.label("目的変数:").classes("self-center")
                        target_col = self.state.get("target_col", "")
                        ui.label(target_col or "未設定").classes("text-weight-bold")

                    ui.separator()

                    # 対話エリア
                    ui.label("対話履歴:").classes("text-subtitle2 mt-2")
                    self.conversation_container = ui.column().classes("w-full max-h-64 overflow-auto border p-2 rounded")

                    # ユーザー入力
                    with ui.row().classes("w-full mt-2"):
                        user_input = ui.input(placeholder="質問への回答を入力...").classes("flex-grow")
                        ui.button("送信", on_click=lambda: self._handle_interview_response(user_input.value, user_input))

                # 直接推奨モード
                with direct_tab:
                    ui.label("目的変数の情報を入力し、直接記述子を推奨します。").classes("text-caption text-gray mb-2")

                    ui.label("目的変数:").classes("text-subtitle2")
                    target_col = self.state.get("target_col", "")
                    ui.label(target_col or "未設定").classes("text-weight-bold")

                    ui.label("目的変数の説明 (オプション):").classes("text-subtitle2 mt-2")
                    self.target_desc_input = ui.textarea(
                        placeholder="例: 屈折率、バンドギャップ、溶解度など"
                    ).classes("w-full")

                    ui.label("追加情報 (オプション):").classes("text-subtitle2 mt-2")
                    self.interview_notes_input = ui.textarea(
                        placeholder="ユーザーからの追加情報があれば入力"
                    ).classes("w-full")

                    ui.button("記述子を推奨", on_click=self._recommend_direct, icon="psychology").classes("mt-2")

                ui.separator()

                # 推奨結果エリア
                ui.label("推奨記述子:").classes("text-subtitle2 mt-4")
                self.result_container = ui.column().classes("w-full")

                # アクションボタン
                with ui.row().classes("w-full justify-end mt-4"):
                    ui.button("閉じる", on_click=lambda: self.dialog.close(), icon="close")
                    ui.button("選択した記述子を適用", on_click=self._apply_recommendations, icon="check", color="primary").props("outline")

        return dialog

    def open(self):
        """ダイアログを開く。"""
        if self.dialog:
            self.dialog.open()
            # 初期化
            self.recommended_descriptors = []
            self.conversation_history = []
            self.interview_mode = False
            self.interview_history = []
            # 初期メッセージを表示
            self._add_conversation_message("システム", "記述子推奨を開始します。タブを切り替えて方法を選択してください。")

    def _recommend_direct(self):
        """直接推奨モード: LLMに記述子を推奨させる。"""
        target_description = self.target_desc_input.value if self.target_desc_input else ""
        interview_notes = self.interview_notes_input.value if self.interview_notes_input else ""

        # バックエンドのLLMDataAnalystを呼び出す
        try:
            from backend.llm.data_analyst import get_data_analyst
            analyst = get_data_analyst()

            result = analyst.recommend_descriptors(
                state=self.state,
                target_description=target_description,
                interview_notes=interview_notes,
            )

            reply = result.get("reply", "")
            recommendations = result.get("recommendations", {})
            self.conversation_history = result.get("conversation", [])

            # 結果を表示
            self._display_recommendations(recommendations)
            self._add_conversation_message("LLM", reply)

        except Exception as e:
            logger.exception(f"記述子推奨エラー: {e}")
            ui.notify(f"エラー: {e}", type="negative")
            if self.result_container:
                self.result_container.clear()
                with self.result_container:
                    ui.label(f"エラーが発生しました: {e}").classes("text-negative")

    def _handle_interview_response(self, user_response: str, user_input: Any):
        """ヒヤリングモード: ユーザーの回答をLLMに送信する。"""
        if not user_response:
            ui.notify("回答を入力してください", type="warning")
            return

        # バックエンドのLLMDataAnalystを呼び出す
        try:
            from backend.llm.data_analyst import get_data_analyst
            analyst = get_data_analyst()

            result = analyst.start_interview(
                state=self.state,
                user_response=user_response,
                interview_history=self.interview_history,
            )

            reply = result.get("reply", "")
            self.interview_history = result.get("interview_history", [])
            interview_complete = result.get("interview_complete", False)
            recommendations = result.get("recommendations", {})

            # 対話履歴に追加
            self._add_conversation_message("ユーザー", user_response)
            self._add_conversation_message("LLM", reply)

            # 推奨完了
            if interview_complete:
                self._display_recommendations(recommendations)
                ui.notify("記述子推奨が完了しました", type="positive")
                self.interview_mode = False

            # 入力欄をクリア
            if hasattr(user_input, 'value'):
                user_input.value = ""

        except Exception as e:
            logger.exception(f"ヒヤリングエラー: {e}")
            ui.notify(f"エラー: {e}", type="negative")

    def _display_recommendations(self, recommendations: dict):
        """推奨結果を表示する。"""
        if not self.result_container:
            return

        self.recommended_descriptors = recommendations.get("descriptors", [])

        self.result_container.clear()
        with self.result_container:
            if not self.recommended_descriptors:
                ui.label("推奨記述子がありません").classes("text-gray")
                return

            # マッチした物性
            matched_property = recommendations.get("matched_property", "")
            if matched_property:
                ui.label(f"推定される物性: {matched_property}").classes("text-caption mb-2")

            confidence = recommendations.get("confidence", "low")
            confidence_color = {
                "high": "positive",
                "medium": "warning",
                "low": "negative",
            }.get(confidence, "negative")
            ui.label(f"確信度: {confidence}").classes(f"text-caption text-{confidence_color} mb-2")

            # 記述子リスト
            for desc in sorted(self.recommended_descriptors, key=lambda x: x.get("priority", 99)):
                name = desc.get("name", "")
                source = desc.get("source", "")
                reason = desc.get("reason", "")
                priority = desc.get("priority", 0)

                with ui.card().classes("w-full mb-2 cursor-pointer") as card:
                    with ui.row().classes("items-center"):
                        ui.checkbox(value=True).props(f"id=desc_{name}")
                        with ui.column().classes("flex-grow"):
                            ui.label(f"{name} ({source})").classes("text-weight-bold")
                            if reason:
                                ui.label(reason).classes("text-caption text-gray")
                            ui.label(f"優先度: {priority}").classes("text-caption")

            # ノート
            notes = recommendations.get("notes", "")
            if notes:
                ui.separator()
                ui.label("補足:").classes("text-caption")
                ui.label(notes).classes("text-caption text-gray")

    def _add_conversation_message(self, sender: str, message: str):
        """対話履歴にメッセージを追加する。"""
        if not self.conversation_container:
            return

        with self.conversation_container:
            with ui.row().classes("w-full mb-2"):
                ui.badge(sender).props("outline")
                ui.label(message).classes("text-body2 ml-2").style("white-space: pre-wrap")

    def _apply_recommendations(self):
        """推奨された記述子を適用する。"""
        if not self.recommended_descriptors:
            ui.notify("適用する記述子がありません", type="warning")
            return

        # 選択された記述子を収集
        selected = []
        for desc in self.recommended_descriptors:
            name = desc.get("name", "")
            # チェックボックスの状態を確認（簡易実装）
            selected.append(name)

        if self.on_apply:
            self.on_apply(selected)
            ui.notify(f"{len(selected)}個の記述子を適用しました", type="positive")
        else:
            # 状態に直接設定
            self.state["selected_descriptors"] = selected
            ui.notify(f"{len(selected)}個の記述子を選択しました", type="positive")


def open_descriptor_recommendation_dialog(
    state: dict,
    on_apply: Callable[[list[str]], Any] | None = None,
):
    """
    記述子推奨ダイアログを開く。

    Args:
        state: アプリケーション状態
        on_apply: 適用時のコールバック
    """
    dialog = DescriptorRecommendationDialog(state, on_apply)
    dialog.create_dialog()
    dialog.open()
    return dialog
