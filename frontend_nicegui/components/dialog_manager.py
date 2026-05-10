# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/dialog_manager.py

汎用ダイアログ管理基盤。

機能:
  - create_settings_dialog: 設定ダイアログの生成（スナップショット+復元）
  - ダイアログ共通仕様:
    * モーダル（外クリック不可）
    * ×ボタン / キャンセルボタン / 確定ボタン
    * キャンセル時にスナップショットから復元
    * 確定/キャンセル時に通知表示
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional

from nicegui import ui

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# スナップショット管理
# ═══════════════════════════════════════════════════════════

class StateSnapshot:
    """state dict の部分スナップショットを取得・復元する。

    Usage:
        snap = StateSnapshot(state, keys=["cv_config", "selected_descriptors"])
        snap.take()        # 現在の値を保存
        ...                # ユーザーが変更
        snap.restore()     # キャンセル → 元に戻す
    """

    def __init__(self, state: dict, keys: list[str]) -> None:
        self._state = state
        self._keys = keys
        self._saved: dict[str, Any] = {}

    def take(self) -> None:
        """現在の state のスナップショットを保存。"""
        for k in self._keys:
            if k in self._state:
                self._saved[k] = copy.deepcopy(self._state[k])
            else:
                self._saved[k] = None

    def restore(self) -> None:
        """保存したスナップショットに復元。"""
        for k in self._keys:
            if k in self._saved:
                if self._saved[k] is None:
                    self._state.pop(k, None)
                else:
                    self._state[k] = copy.deepcopy(self._saved[k])


# ═══════════════════════════════════════════════════════════
# ダイアログビルダー
# ═══════════════════════════════════════════════════════════

def create_settings_dialog(
    *,
    title: str,
    icon: str = "settings",
    width: str = "85vw",
    max_width: str = "900px",
    content_builder: Callable[[], None],
    state: dict,
    snapshot_keys: list[str],
    on_confirm: Optional[Callable[[], None]] = None,
    on_cancel: Optional[Callable[[], None]] = None,
    confirm_label: str = "✅ 適用",
    cancel_label: str = "❌ キャンセル",
) -> ui.dialog:
    """汎用設定ダイアログを生成して返す。

    Args:
        title: ダイアログタイトル
        icon: Material icon 名
        width: CSSのwidth
        max_width: CSSのmax-width
        content_builder: ダイアログ本体を描画するコールバック
        state: アプリケーション state dict
        snapshot_keys: キャンセル時に復元するstateキー
        on_confirm: 確定時の追加コールバック
        on_cancel: キャンセル時の追加コールバック
        confirm_label: 確定ボタンのラベル
        cancel_label: キャンセルボタンのラベル

    Returns:
        ui.dialog オブジェクト（open() で表示可能）
    """
    snap = StateSnapshot(state, snapshot_keys)
    snap.take()

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
        f"width: {width}; max-width: {max_width}; max-height: 85vh;"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center justify-between full-width q-mb-sm"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon(icon).classes("text-cyan text-h6")
                ui.label(title).classes("text-h6")
            ui.button(
                icon="close",
                on_click=lambda: _handle_cancel(dlg, snap, on_cancel),
            ).props("flat round dense color=grey-5")

        ui.separator().classes("q-mb-sm")

        # ── コンテンツ（スクロール対応）──
        with ui.scroll_area().classes("full-width").style(
            "max-height: calc(85vh - 140px);"
        ):
            content_builder()

        ui.separator().classes("q-mt-sm")

        # ── フッターボタン ──
        with ui.row().classes("full-width justify-end q-gutter-sm q-mt-sm"):
            ui.button(
                cancel_label,
                on_click=lambda: _handle_cancel(dlg, snap, on_cancel),
            ).props("flat no-caps color=grey-5")
            ui.button(
                confirm_label,
                on_click=lambda: _handle_confirm(dlg, on_confirm),
            ).props("unelevated no-caps color=cyan")

    dlg.props("persistent")
    return dlg


def _handle_cancel(
    dlg: ui.dialog,
    snap: StateSnapshot,
    on_cancel: Optional[Callable[[], None]],
) -> None:
    """キャンセル処理: スナップショット復元 → 通知 → ダイアログ閉じ。"""
    snap.restore()
    dlg.close()
    if on_cancel:
        try:
            on_cancel()
        except Exception as e:
            logger.warning(f"キャンセルコールバックエラー: {e}")
    ui.notify("❌ キャンセルしました", type="info")


def _handle_confirm(
    dlg: ui.dialog,
    on_confirm: Optional[Callable[[], None]],
) -> None:
    """確定処理: コールバック実行 → 通知 → ダイアログ閉じ。"""
    try:
        if on_confirm:
            on_confirm()
        dlg.close()
        ui.notify("✅ 設定を適用しました", type="positive")
    except Exception as e:
        logger.error(f"確定コールバックエラー: {e}")
        ui.notify(f"⚠️ エラー: {e}", type="negative")


# ═══════════════════════════════════════════════════════════
# サマリーカード（ダイアログ呼び出しボタン付き）
# ═══════════════════════════════════════════════════════════

def render_settings_summary(
    *,
    icon: str,
    title: str,
    summary_lines: list[str],
    button_label: str = "⚙️ 設定変更",
    on_click: Callable[[], None],
    badge_text: str = "",
    badge_color: str = "cyan",
) -> None:
    """メイン画面に表示するコンパクトな設定サマリー+ダイアログ呼び出しボタン。

    Args:
        icon: Material icon
        title: セクションタイトル
        summary_lines: サマリー表示する行のリスト
        button_label: ダイアログ呼び出しボタンのラベル
        on_click: ボタンクリック時のコールバック
        badge_text: バッジテキスト（任意）
        badge_color: バッジカラー
    """
    with ui.card().classes("full-width q-pa-sm q-mb-xs").style(
        "border: 1px solid rgba(0,188,212,0.2); border-radius: 8px;"
        "background: rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center justify-between full-width"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon(icon).classes("text-cyan")
                ui.label(title).classes("text-subtitle2 text-bold")
                if badge_text:
                    ui.badge(badge_text, color=badge_color).props("outline dense")

            ui.button(
                button_label,
                on_click=on_click,
            ).props("outline dense no-caps size=sm color=cyan")

        if summary_lines:
            for line in summary_lines:
                ui.label(line).classes("text-caption text-grey-5 q-ml-lg")
