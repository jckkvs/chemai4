"""
backend/utils/optional_import.py

オプショナルライブラリの安全なimportを管理するモジュール。
importが失敗した場合、アプリはその機能を無効化して起動し続ける。
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ライブラリ可用性フラグのキャッシュ
_availability_cache: dict[str, bool] = {}


def safe_import(module_name: str, alias: str | None = None) -> Any:
    """
    指定モジュールを安全にimportする。失敗した場合は None を返す。

    Args:
        module_name: importするモジュール名（例: "rdkit.Chem"）
        alias: モジュールに付ける別名（ログ表示用）

    Returns:
        importされたモジュール、または失敗時は None
    """
    name = alias or module_name
    try:
        mod = importlib.import_module(module_name)
        _availability_cache[name] = True
        return mod
    except (ImportError, ValueError, AttributeError) as e:
        logger.warning(f"[optional] '{name}' は利用不可（環境エラー）: {e}")
        _availability_cache[name] = False
        return None
    except Exception as e:
        logger.warning(f"[optional] '{name}' のimport中に予期せぬエラー: {e}")
        _availability_cache[name] = False
        return None


def is_available(name: str) -> bool:
    """
    指定ライブラリ（またはその機能グループ）が利用可能か返す。

    Args:
        name: ライブラリ名またはエイリアス

    Returns:
        利用可能なら True、そうでなければ False
    """
    return _availability_cache.get(name, False)


def require(name: str, feature: str = "") -> None:
    """
    ライブラリが利用可能でなければ RuntimeError を送出する。
    エキスパートモードで明示的に必要なライブラリを要求する際に使用。

    Args:
        name: ライブラリ名またはエイリアス
        feature: 利用しようとしている機能名（エラーメッセージ表示用）

    Raises:
        RuntimeError: ライブラリが利用不可の場合
    """
    if not is_available(name):
        feature_str = f" (機能: {feature})" if feature else ""
        raise RuntimeError(
            f"'{name}' は現在の環境でインストールされていません{feature_str}。"
            f"`pip install {name}` または conda-forge からインストールしてください。"
        )


def get_availability_report() -> dict[str, bool]:
    """現在判明しているライブラリ可用性の一覧を返す。"""
    return dict(_availability_cache)


# ---- 既知のオプショナルライブラリをまとめて試みる ----
# バックエンド起動時に一度呼び出すことで可用性をキャッシュする

def probe_all_optional_libraries() -> dict[str, bool]:
    """
    使用するオプショナルライブラリを一括で試みてキャッシュを更新する。
    アプリ起動時に一度だけ呼び出すこと。

    Returns:
        {ライブラリ名: 利用可能か} の辞書
    """
    probes: list[tuple[str, str]] = [
        # (module_name, alias)
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
        ("optuna", "optuna"),
        ("skopt", "scikit-optimize"),
        ("umap", "umap-learn"),
        ("phate", "phate"),
        ("sage", "sage-importance"),
        ("lime", "lime"),
        ("eli5", "eli5"),
        ("boruta", "boruta"),
        ("rdkit", "rdkit"),
        ("rdkit.Chem", "rdkit.Chem"),
        ("mordred", "mordred"),
        ("dscribe", "dscribe"),
        ("xtb", "xtb-python"),
        ("mace", "mace-torch"),
        ("torchani", "torchani"),
        ("fairchem.core", "uma"),
        ("deepchem", "deepchem"),
        ("unipka", "unipka"),
        ("crest", "crest"),
        ("torch", "torch"),
        ("mlflow", "mlflow"),
        ("statsmodels", "statsmodels"),
        # パイプライン拡張
        ("skrebate", "relieff"),
        ("sklearn_genetic", "sklearn-genetic-opt"),
        ("group_lasso", "group-lasso"),
        ("linear_tree", "linear-tree"),
        ("mlxtend", "mlxtend"),
    ]

    for module_name, alias in probes:
        safe_import(module_name, alias)

    return get_availability_report()
