"""
frontend_nicegui/components/estimator_config_dialog.py

Estimator設定ダイアログ — 3タブ構成:
    Tab 1: デフォルトパラメータ設定（既存auto_params_uiを拡張）
    Tab 2: GridSearchCV探索範囲設定
    Tab 3: OptunaSearchCV探索範囲設定

Usage:
    from frontend_nicegui.components.estimator_config_dialog import EstimatorConfigDialog

    dialog = EstimatorConfigDialog(
        model_key="rf",
        model_cls=RandomForestRegressor,
        on_save=lambda cfg: state.update({"model_config": cfg}),
    )
    dialog.open()

設計思想:
    - 全てのestimatorに対して動的にUI生成
    - ParamSpec + SearchParamSpec によるメタ駆動
    - ユーザー登録カスタムestimatorにも自動対応
    - docstringから説明を取得し、各パラメータにツールチップ表示
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from nicegui import ui

from backend.ui.param_schema import (
    ParamSpec,
    introspect_params,
    get_basic_specs,
    get_advanced_specs,
    apply_params,
)
from backend.models.search_space_generator import (
    SearchParamSpec,
    generate_search_spaces,
)

logger = logging.getLogger(__name__)


# ============================================================
# 設定結果データ構造
# ============================================================

class EstimatorConfig:
    """Estimator設定の統合データ。

    Attributes:
        model_key:      モデルキー
        model_cls:      estimatorクラス
        default_params: ユーザー設定のデフォルトパラメータ
        grid_space:     GridSearchCV用探索空間
        optuna_space:   OptunaSearchCV用探索空間
    """

    def __init__(
        self,
        model_key: str,
        model_cls: type | None = None,
        default_params: dict[str, Any] | None = None,
        grid_space: dict[str, list[Any]] | None = None,
        optuna_space: dict[str, dict[str, Any]] | None = None,
    ):
        self.model_key = model_key
        self.model_cls = model_cls
        self.default_params = default_params or {}
        self.grid_space = grid_space or {}
        self.optuna_space = optuna_space or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "default_params": self.default_params,
            "grid_space": self.grid_space,
            "optuna_space": self.optuna_space,
        }


# ============================================================
# メインダイアログクラス
# ============================================================

class EstimatorConfigDialog:
    """動的Estimator設定ダイアログ。

    3タブ構成:
        1. 📊 デフォルト — パラメータのデフォルト値設定
        2. 🔍 GridSearch — GridSearchCV用の探索値リスト設定
        3. ⚡ Optuna — OptunaSearchCV用の連続/離散探索範囲設定
    """

    def __init__(
        self,
        model_key: str,
        model_cls: type,
        *,
        model_name: str | None = None,
        on_save: Callable[[EstimatorConfig], None] | None = None,
        initial_config: EstimatorConfig | None = None,
    ):
        self.model_key = model_key
        self.model_cls = model_cls
        self.model_name = model_name or model_cls.__name__
        self.on_save = on_save

        # イントロスペクション
        self.param_specs = introspect_params(model_cls)
        self.search_spaces = generate_search_spaces(
            self.param_specs, include_advanced=True,
        )

        # 設定値のリアクティブ辞書
        self.default_values: dict[str, Any] = {}
        self.grid_config: dict[str, dict[str, Any]] = {}
        self.optuna_config: dict[str, dict[str, Any]] = {}

        # 初期値の設定
        self._init_defaults(initial_config)

        # ダイアログ参照
        self._dialog: ui.dialog | None = None

    def _init_defaults(self, initial: EstimatorConfig | None) -> None:
        """初期値を設定する。"""
        for spec in self.param_specs:
            self.default_values[spec.name] = spec.default

        for name, ss in self.search_spaces.items():
            self.grid_config[name] = {
                "enabled": ss.enabled,
                "values": ss.grid_values.copy() if ss.grid_values else [],
                "values_str": ", ".join(str(v) for v in ss.grid_values),
            }
            self.optuna_config[name] = {
                "enabled": ss.enabled,
                "type": ss.optuna_type,
                "low": ss.optuna_low,
                "high": ss.optuna_high,
                "step": ss.optuna_step,
                "log": ss.optuna_log,
                "choices": ss.optuna_choices.copy() if ss.optuna_choices else [],
                "choices_str": ", ".join(str(v) for v in ss.optuna_choices),
            }

        if initial:
            self.default_values.update(initial.default_params)

    def open(self) -> None:
        """ダイアログを開く。"""
        self._build_dialog()
        if self._dialog:
            self._dialog.open()

    def _build_dialog(self) -> None:
        """ダイアログUIを構築する。"""
        self._dialog = ui.dialog().props("maximized")

        with self._dialog:
            with ui.card().classes("w-full max-w-4xl mx-auto q-pa-none").style(
                "max-height: 90vh; overflow: hidden; "
                "background: rgba(20,20,35,0.95); backdrop-filter: blur(20px);"
            ):
                # ヘッダー
                with ui.row().classes("items-center q-pa-md full-width").style(
                    "background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);"
                ):
                    ui.icon("tune", size="md").classes("text-white")
                    ui.label(f"⚙️ {self.model_name} パラメータ設定").classes(
                        "text-h6 text-white q-ml-sm"
                    )
                    ui.space()
                    ui.button(icon="close", on_click=self._dialog.close).props(
                        "flat round dense color=white"
                    )

                # タブ
                with ui.tabs().classes("full-width").props(
                    "dense active-color=primary indicator-color=primary"
                ) as tabs:
                    tab_default = ui.tab("default", label="📊 デフォルト", icon="settings")
                    tab_grid = ui.tab("grid", label="🔍 GridSearch", icon="grid_on")
                    tab_optuna = ui.tab("optuna", label="⚡ Optuna", icon="auto_awesome")

                with ui.tab_panels(tabs, value="default").classes(
                    "full-width"
                ).style("flex: 1; overflow-y: auto; max-height: 60vh;"):

                    # ── Tab 1: デフォルトパラメータ ──
                    with ui.tab_panel("default").classes("q-pa-md"):
                        self._render_default_tab()

                    # ── Tab 2: GridSearchCV ──
                    with ui.tab_panel("grid").classes("q-pa-md"):
                        self._render_grid_tab()

                    # ── Tab 3: OptunaSearchCV ──
                    with ui.tab_panel("optuna").classes("q-pa-md"):
                        self._render_optuna_tab()

                # フッター
                with ui.row().classes("q-pa-md full-width justify-end q-gutter-sm").style(
                    "border-top: 1px solid rgba(255,255,255,0.1);"
                ):
                    ui.button("キャンセル", on_click=self._dialog.close).props(
                        "flat color=grey"
                    )
                    ui.button(
                        "💾 保存",
                        on_click=self._on_save_click,
                    ).props("color=primary").classes("q-px-lg")

    # ──────────────────────────────────────────────────
    # Tab 1: デフォルトパラメータ
    # ──────────────────────────────────────────────────

    def _render_default_tab(self) -> None:
        """デフォルトパラメータ設定タブを描画する。"""
        basic = get_basic_specs(self.param_specs)
        advanced = get_advanced_specs(self.param_specs)

        if not self.param_specs:
            ui.label("⚙️ 設定可能なパラメータはありません").classes("text-grey-5")
            return

        ui.label("モデルのデフォルトパラメータを設定します。").classes(
            "text-caption text-grey-6 q-mb-sm"
        )

        if basic:
            with ui.card().classes("glass-card q-pa-sm full-width q-mb-sm"):
                ui.label("🔧 基本パラメータ").classes("text-subtitle2 q-mb-xs")
                for spec in basic:
                    self._render_default_widget(spec)

        if advanced:
            with ui.expansion(
                f"🔧 詳細設定 ({len(advanced)}項目)", icon="tune",
            ).classes("full-width q-mt-xs"):
                with ui.card().classes("glass-card q-pa-sm full-width"):
                    for spec in advanced:
                        self._render_default_widget(spec)

    def _render_default_widget(self, spec: ParamSpec) -> None:
        """1つのデフォルトパラメータウィジェットを描画。"""
        from frontend_nicegui.components.auto_params_ui import _render_widget
        _render_widget(spec, self.default_values)

    # ──────────────────────────────────────────────────
    # Tab 2: GridSearchCV
    # ──────────────────────────────────────────────────

    def _render_grid_tab(self) -> None:
        """GridSearchCV探索範囲設定タブ。"""
        if not self.search_spaces:
            ui.label("探索対象パラメータがありません").classes("text-grey-5")
            return

        ui.label(
            "GridSearchCV / HalvingGridSearchCV で探索する値のリストを設定します。"
        ).classes("text-caption text-grey-6 q-mb-sm")
        ui.label(
            "カンマ区切りで値を入力してください。型は自動判定されます。"
        ).classes("text-caption text-grey-7 q-mb-md")

        for name, ss in self.search_spaces.items():
            cfg = self.grid_config.get(name, {})
            with ui.card().classes("glass-card q-pa-sm full-width q-mb-xs"):
                with ui.row().classes("items-center full-width q-gutter-xs"):
                    # 有効/無効チェックボックス
                    ui.checkbox(
                        "",
                        value=cfg.get("enabled", True),
                        on_change=lambda e, n=name: self.grid_config[n].update(
                            {"enabled": e.value}
                        ),
                    ).classes("q-mr-xs")

                    # パラメータ名 + 型
                    label_text = f"{name}"
                    type_badge = ss.param_type
                    with ui.column().classes("q-mr-sm"):
                        ui.label(label_text).classes("text-bold text-body2")
                        if ss.description:
                            ui.label(ss.description[:80]).classes(
                                "text-caption text-grey-6"
                            ).style("font-size: 0.7em;")

                    ui.badge(type_badge, color="info").props("outline")
                    ui.space()

                    # 値リスト入力
                    ui.input(
                        label="探索値（カンマ区切り）",
                        value=cfg.get("values_str", ""),
                        on_change=lambda e, n=name: self._update_grid_values(
                            n, e.value
                        ),
                    ).classes("w-64").props("dense")

    def _update_grid_values(self, name: str, values_str: str) -> None:
        """Grid値の文字列を解析してリストに変換。"""
        parsed = _parse_value_list(values_str)
        self.grid_config[name]["values"] = parsed
        self.grid_config[name]["values_str"] = values_str

    # ──────────────────────────────────────────────────
    # Tab 3: OptunaSearchCV
    # ──────────────────────────────────────────────────

    def _render_optuna_tab(self) -> None:
        """OptunaSearchCV探索範囲設定タブ。"""
        if not self.search_spaces:
            ui.label("探索対象パラメータがありません").classes("text-grey-5")
            return

        ui.label(
            "Optuna / RandomizedSearchCV で探索する範囲を設定します。"
        ).classes("text-caption text-grey-6 q-mb-sm")

        for name, ss in self.search_spaces.items():
            cfg = self.optuna_config.get(name, {})
            with ui.card().classes("glass-card q-pa-sm full-width q-mb-xs"):
                with ui.row().classes("items-center full-width q-gutter-xs"):
                    # 有効/無効
                    ui.checkbox(
                        "",
                        value=cfg.get("enabled", True),
                        on_change=lambda e, n=name: self.optuna_config[n].update(
                            {"enabled": e.value}
                        ),
                    ).classes("q-mr-xs")

                    # パラメータ名
                    with ui.column().classes("q-mr-sm"):
                        ui.label(name).classes("text-bold text-body2")
                        if ss.description:
                            ui.label(ss.description[:80]).classes(
                                "text-caption text-grey-6"
                            ).style("font-size: 0.7em;")

                optuna_type = cfg.get("type", "float")

                if optuna_type in ("int", "float"):
                    # 数値範囲設定
                    with ui.row().classes("items-center q-gutter-xs q-mt-xs q-pl-lg"):
                        ui.select(
                            label="型",
                            options=["int", "float"],
                            value=optuna_type,
                            on_change=lambda e, n=name: self.optuna_config[n].update(
                                {"type": e.value}
                            ),
                        ).classes("w-24").props("dense")

                        ui.number(
                            label="最小値",
                            value=cfg.get("low", 0),
                            on_change=lambda e, n=name: self.optuna_config[n].update(
                                {"low": e.value}
                            ),
                            format="%.6g",
                        ).classes("w-28").props("dense")

                        ui.number(
                            label="最大値",
                            value=cfg.get("high", 1),
                            on_change=lambda e, n=name: self.optuna_config[n].update(
                                {"high": e.value}
                            ),
                            format="%.6g",
                        ).classes("w-28").props("dense")

                        if optuna_type == "int":
                            ui.number(
                                label="ステップ",
                                value=cfg.get("step", 1),
                                min=1,
                                on_change=lambda e, n=name: self.optuna_config[
                                    n
                                ].update({"step": e.value}),
                            ).classes("w-20").props("dense")

                        ui.checkbox(
                            "対数",
                            value=cfg.get("log", False),
                            on_change=lambda e, n=name: self.optuna_config[n].update(
                                {"log": e.value}
                            ),
                        ).tooltip(
                            "対数スケールで探索（learning_rate等に有効）"
                        )

                elif optuna_type == "categorical":
                    # カテゴリカル設定
                    with ui.row().classes("items-center q-gutter-xs q-mt-xs q-pl-lg"):
                        ui.badge("categorical", color="warning").props("outline")
                        ui.input(
                            label="選択肢（カンマ区切り）",
                            value=cfg.get("choices_str", ""),
                            on_change=lambda e, n=name: self._update_optuna_choices(
                                n, e.value
                            ),
                        ).classes("w-64").props("dense")

    def _update_optuna_choices(self, name: str, choices_str: str) -> None:
        """Optuna categoricalの選択肢文字列を解析。"""
        parsed = _parse_value_list(choices_str)
        self.optuna_config[name]["choices"] = parsed
        self.optuna_config[name]["choices_str"] = choices_str

    # ──────────────────────────────────────────────────
    # 保存
    # ──────────────────────────────────────────────────

    def _on_save_click(self) -> None:
        """保存ボタンクリック時の処理。"""
        # デフォルトパラメータ: 変更されたもののみ
        changed_params = apply_params(self.param_specs, self.default_values)

        # GridSearchCV用
        grid_space: dict[str, list[Any]] = {}
        for name, cfg in self.grid_config.items():
            if cfg.get("enabled") and cfg.get("values"):
                grid_space[name] = cfg["values"]

        # OptunaSearchCV用
        optuna_space: dict[str, dict[str, Any]] = {}
        for name, cfg in self.optuna_config.items():
            if not cfg.get("enabled"):
                continue
            t = cfg.get("type", "float")
            if t in ("int", "float"):
                entry: dict[str, Any] = {
                    "type": t,
                    "low": cfg.get("low", 0),
                    "high": cfg.get("high", 1),
                    "log": cfg.get("log", False),
                }
                if t == "int":
                    entry["step"] = int(cfg.get("step", 1))
                optuna_space[name] = entry
            elif t == "categorical":
                if cfg.get("choices"):
                    optuna_space[name] = {
                        "type": "categorical",
                        "choices": cfg["choices"],
                    }

        config = EstimatorConfig(
            model_key=self.model_key,
            model_cls=self.model_cls,
            default_params=changed_params,
            grid_space=grid_space,
            optuna_space=optuna_space,
        )

        if self.on_save:
            self.on_save(config)

        if self._dialog:
            self._dialog.close()

        ui.notify(f"✅ {self.model_name} の設定を保存しました", type="positive")


# ============================================================
# 複数モデル一括設定パネル
# ============================================================

def render_model_config_panel(
    model_entries: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    task: str = "regression",
) -> None:
    """モデル選択リストの各モデルに設定ボタンを追加する。

    Args:
        model_entries: factory.list_models() の結果
        state: アプリケーションのグローバル状態辞書
        task: "regression" | "classification"
    """
    if "model_configs" not in state:
        state["model_configs"] = {}

    for entry in model_entries:
        key = entry["key"]
        name = entry["name"]
        cls = entry.get("class")

        if cls is None:
            continue  # factory関数のみのモデルはスキップ

        with ui.row().classes("items-center q-gutter-xs"):
            # 設定ボタン
            def _open_dialog(k=key, n=name, c=cls):
                existing = state["model_configs"].get(k)
                dialog = EstimatorConfigDialog(
                    model_key=k,
                    model_cls=c,
                    model_name=n,
                    initial_config=existing,
                    on_save=lambda cfg, k=k: state["model_configs"].update({k: cfg}),
                )
                dialog.open()

            btn = ui.button(
                icon="tune",
                on_click=_open_dialog,
            ).props("flat dense round size=sm")
            btn.tooltip(f"{name} の詳細設定 (デフォルト/Grid/Optuna)")

            # 設定済みバッジ
            if key in state.get("model_configs", {}):
                ui.badge("設定済", color="positive").props("outline").classes(
                    "q-ml-xs"
                )


# ============================================================
# ユーティリティ
# ============================================================

def _parse_value_list(text: str) -> list[Any]:
    """カンマ区切りテキストを型自動判定リストに変換する。

    Examples:
        "100, 200, 500" → [100, 200, 500]
        "0.01, 0.1, 1.0" → [0.01, 0.1, 1.0]
        "rbf, linear" → ["rbf", "linear"]
        "True, False" → [True, False]
        "None, 3, 5" → [None, 3, 5]
    """
    if not text or not text.strip():
        return []

    values: list[Any] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue

        # None
        if s.lower() == "none":
            values.append(None)
            continue

        # bool
        if s.lower() in ("true", "false"):
            values.append(s.lower() == "true")
            continue

        # int
        try:
            val = int(s)
            values.append(val)
            continue
        except ValueError:
            pass

        # float
        try:
            val = float(s)
            values.append(val)
            continue
        except ValueError:
            pass

        # str
        values.append(s.strip("\"'"))

    return values
