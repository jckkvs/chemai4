"""
backend/chem/descriptors/__init__.py

プラグイン型記述子の自動ディスカバリ & レジストリ。

_builtins/ — 組込み記述子（読取専用・ユーザー編集不可）
custom/   — ユーザーカスタム記述子（編集・追加・削除可）

同名の DESCRIPTOR_NAME がある場合、custom/ が _builtins/ を上書き。
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.descriptors.base import PluginInfo, validate_plugin, safe_compute

logger = logging.getLogger(__name__)

_DESCRIPTORS_DIR = Path(__file__).parent
_BUILTINS_DIR = _DESCRIPTORS_DIR / "_builtins"
_CUSTOM_DIR = _DESCRIPTORS_DIR / "custom"

# キャッシュ
_plugin_cache: list[PluginInfo] | None = None


def _load_module_from_path(filepath: str, module_name: str):
    """ファイルパスからモジュールをロードする。"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.warning(f"⚠️ プラグイン '{filepath}' のロードに失敗: {e}")
        return None
    return module


def discover_plugins(force_reload: bool = False) -> list[PluginInfo]:
    """
    _builtins/ と custom/ の全プラグインを検出し、PluginInfo のリストを返す。

    検出ルール:
    - _builtins/ 内の .py ファイル（_で始まるものは除外）
    - custom/ 内の .py ファイル（_で始まるものは除外）
    - custom/ の同名 DESCRIPTOR_NAME は _builtins/ を上書き
    - 不正なプラグインは Warning を出して無視
    """
    global _plugin_cache
    if _plugin_cache is not None and not force_reload:
        return _plugin_cache

    plugins: dict[str, PluginInfo] = {}  # name -> PluginInfo

    # 1. _builtins/ をスキャン
    if _BUILTINS_DIR.exists():
        for py_file in sorted(_BUILTINS_DIR.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"descriptors._builtins.{py_file.stem}"
            module = _load_module_from_path(str(py_file), module_name)
            if module is None:
                continue
            info = validate_plugin(module, str(py_file))
            if info is not None:
                plugins[info.name] = info

    # 2. custom/ をスキャン（_builtins/ を上書き可能）
    if _CUSTOM_DIR.exists():
        for py_file in sorted(_CUSTOM_DIR.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"descriptors.custom.{py_file.stem}"
            module = _load_module_from_path(str(py_file), module_name)
            if module is None:
                continue
            info = validate_plugin(module, str(py_file))
            if info is not None:
                if info.name in plugins and plugins[info.name].is_builtin:
                    logger.info(
                        f"✏️ カスタムプラグイン '{info.name}' が組込みを上書きします。"
                    )
                plugins[info.name] = info

    result = list(plugins.values())
    _plugin_cache = result
    logger.info(f"📦 記述子プラグイン: {len(result)} 個を検出（組込み: {sum(1 for p in result if p.is_builtin)}, カスタム: {sum(1 for p in result if not p.is_builtin)}）")
    return result


def get_plugins_by_engine(engine: str) -> list[PluginInfo]:
    """指定エンジンのプラグインを返す。"""
    return [p for p in discover_plugins() if p.engine.lower() == engine.lower()]


def get_plugins_by_category(category: str) -> list[PluginInfo]:
    """指定カテゴリのプラグインを返す。"""
    return [p for p in discover_plugins() if p.category == category]


def get_available_engines() -> list[str]:
    """利用可能なエンジン名のリストを返す。"""
    return sorted(set(p.engine for p in discover_plugins()))


def get_available_categories() -> list[str]:
    """利用可能なカテゴリ名のリストを返す。"""
    return sorted(set(p.category for p in discover_plugins()))


def compute_all_descriptors(
    smiles_list: list[str],
    plugin_names: list[str] | None = None,
    progress_callback: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    全プラグイン（またはplugin_namesで指定）の記述子を計算し、
    1つの DataFrame に結合して返す。

    Args:
        smiles_list: SMILESのリスト
        plugin_names: 使用するプラグイン名のリスト（Noneで全て）
        progress_callback: (step, total, message) コールバック
        **kwargs: 各プラグインに渡すオプション

    Returns:
        DataFrame (n_samples x n_descriptors)
    """
    plugins = discover_plugins()
    if plugin_names is not None:
        plugins = [p for p in plugins if p.name in plugin_names]

    if not plugins:
        logger.warning("⚠️ 有効なプラグインが見つかりません。空のDataFrameを返します。")
        return pd.DataFrame(index=range(len(smiles_list)))

    total = len(plugins)
    dfs: list[pd.DataFrame] = []
    computed_names: list[str] = []

    for i, plugin in enumerate(plugins):
        logger.info(f"🔄 記述子計算中: {plugin.engine} - {plugin.name}")
        if progress_callback:
            progress_callback(i + 1, total, f"{plugin.engine}: {plugin.name}")

        result = safe_compute(plugin, smiles_list, **kwargs)

        if result is None:
            continue

        # DataFrame の場合（複数記述子プラグイン）
        if isinstance(result, pd.DataFrame):
            result = result.reset_index(drop=True)
            dfs.append(result)
            computed_names.append(plugin.name)
        elif isinstance(result, (list, np.ndarray)):
            arr = list(result)
            if len(arr) == len(smiles_list):
                col_name = plugin.name
                df_single = pd.DataFrame({col_name: arr}, index=range(len(smiles_list)))
                dfs.append(df_single)
                computed_names.append(plugin.name)
            else:
                logger.warning(
                    f"⚠️ プラグイン '{plugin.name}': 戻り値の長さ({len(arr)})が"
                    f"入力({len(smiles_list)})と一致しません。無視します。"
                )
        else:
            logger.warning(
                f"⚠️ プラグイン '{plugin.name}': 戻り値の型({type(result)})が不正です。無視します。"
            )

    if not dfs:
        return pd.DataFrame(index=range(len(smiles_list)))

    # 全て結合
    combined = pd.concat(dfs, axis=1)
    # 重複列名を除去
    combined = combined.loc[:, ~combined.columns.duplicated()]
    # 数値型に変換
    combined = combined.apply(pd.to_numeric, errors="coerce")

    logger.info(f"✅ 記述子計算完了: {len(computed_names)}プラグイン, {combined.shape[1]}列")
    return combined


def get_custom_dir() -> Path:
    """カスタムディレクトリのパスを返す。存在しなければ作成。"""
    _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    return _CUSTOM_DIR


def get_builtins_dir() -> Path:
    """組込みディレクトリのパスを返す。"""
    return _BUILTINS_DIR


def invalidate_cache() -> None:
    """プラグインキャッシュをクリアする。次回 discover_plugins で再スキャン。"""
    global _plugin_cache
    _plugin_cache = None
