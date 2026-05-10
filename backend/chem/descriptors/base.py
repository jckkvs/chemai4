"""
backend/chem/descriptors/base.py

記述子プラグインの基底定義。
全てのプラグイン.pyファイルはこのプロトコルに従う。

ユーザーが記述子プラグインを作る際のルール:
  1. DESCRIPTOR_NAME (str): 記述子の識別名（必須）
  2. DESCRIPTOR_CATEGORY (str): カテゴリ（必須）例: "物理化学", "電子状態"
  3. DESCRIPTOR_ENGINE (str): 計算エンジン（必須）例: "RDKit", "XTB"
  4. DESCRIPTOR_DESCRIPTION (str): 日本語での詳細説明（任意）
  5. compute(smiles_list: list[str], **kwargs) -> list[float | None]:
     引数はSMILESリスト、戻り値は値のリスト（必須）
  6. 複数記述子を返す場合:
     compute(smiles_list, **kwargs) -> pd.DataFrame
"""
from __future__ import annotations

import inspect
import logging
import types
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParamDef:
    """プラグインの設定可能パラメータ定義。"""
    type: str = "int"            # "int", "float", "str", "bool", "choice"
    default: Any = None          # デフォルト値
    min: float | None = None     # 最小値 (int/float用)
    max: float | None = None     # 最大値 (int/float用)
    choices: list[Any] | None = None  # 選択肢 (choice用)
    description: str = ""        # 日本語説明


@dataclass
class PluginInfo:
    """ロード済みプラグインの情報を保持するデータクラス。"""
    name: str                  # DESCRIPTOR_NAME
    category: str              # DESCRIPTOR_CATEGORY
    engine: str                # DESCRIPTOR_ENGINE
    description: str           # DESCRIPTOR_DESCRIPTION
    source_path: str           # .pyファイルのパス
    is_builtin: bool           # _builtins/ にあるか
    is_multi: bool             # DataFrameを返すか (複数記述子)
    compute_fn: Any            # compute 関数への参照
    module: types.ModuleType   # ロードしたモジュール
    params: dict[str, ParamDef] = field(default_factory=dict)  # 設定可能パラメータ
    enabled: bool = True       # UI上で有効/無効

    @property
    def display_name(self) -> str:
        """UI表示用の名前。"""
        return f"[{self.engine}] {self.name}"

    @property
    def has_params(self) -> bool:
        """設定可能パラメータがあるか。"""
        return bool(self.params)


def validate_plugin(module: types.ModuleType, filepath: str) -> PluginInfo | None:
    """
    プラグインモジュールを検証し、PluginInfo を返す。
    不正な場合は Warning をログに出力して None を返す。
    """
    # DESCRIPTOR_NAME チェック
    name = getattr(module, "DESCRIPTOR_NAME", None)
    if not name or not isinstance(name, str):
        logger.warning(
            f"⚠️ プラグイン '{filepath}' に DESCRIPTOR_NAME が未定義または不正です。スキップします。"
        )
        return None

    # DESCRIPTOR_CATEGORY チェック
    category = getattr(module, "DESCRIPTOR_CATEGORY", "未分類")
    if not isinstance(category, str):
        category = "未分類"

    # DESCRIPTOR_ENGINE チェック
    engine = getattr(module, "DESCRIPTOR_ENGINE", "不明")
    if not isinstance(engine, str):
        engine = "不明"

    # DESCRIPTOR_DESCRIPTION チェック
    description = getattr(module, "DESCRIPTOR_DESCRIPTION", "")
    if not isinstance(description, str):
        description = ""

    # compute 関数チェック
    compute_fn = getattr(module, "compute", None)
    if compute_fn is None or not callable(compute_fn):
        logger.warning(
            f"⚠️ プラグイン '{name}' ({filepath}) に compute 関数がありません。スキップします。"
        )
        return None

    # compute のシグネチャ検証: 最低1引数 (smiles_list)
    try:
        sig = inspect.signature(compute_fn)
        params = list(sig.parameters.values())
        # 引数なし = 不正
        if len(params) < 1:
            logger.warning(
                f"⚠️ プラグイン '{name}': compute() に引数がありません。"
                f"compute(smiles_list) の形式が必要です。スキップします。"
            )
            return None
    except (ValueError, TypeError):
        pass  # シグネチャ取得失敗は許容

    # 複数記述子を返すかの判定 (MULTI_DESCRIPTOR フラグ)
    is_multi = getattr(module, "MULTI_DESCRIPTOR", False)

    # 設定可能パラメータの読み取り
    raw_params = getattr(module, "DESCRIPTOR_PARAMS", {})
    params: dict[str, ParamDef] = {}
    if isinstance(raw_params, dict):
        for pname, pdef in raw_params.items():
            if isinstance(pdef, dict):
                params[pname] = ParamDef(
                    type=pdef.get("type", "str"),
                    default=pdef.get("default"),
                    min=pdef.get("min"),
                    max=pdef.get("max"),
                    choices=pdef.get("choices"),
                    description=pdef.get("description", pname),
                )

    return PluginInfo(
        name=name,
        category=category,
        engine=engine,
        description=description,
        source_path=filepath,
        is_builtin=("_builtins" in filepath),
        is_multi=bool(is_multi),
        compute_fn=compute_fn,
        module=module,
        params=params,
    )


def safe_compute(
    plugin: PluginInfo,
    smiles_list: list[str],
    **kwargs: Any,
) -> list[float | None] | Any:
    """
    プラグインの compute を安全に呼び出す。
    エラー発生時は Warning を出し、None リストを返す。
    """
    try:
        # config/kwargs の受け取り可否を確認
        sig = inspect.signature(plugin.compute_fn)
        params = list(sig.parameters.values())

        if len(params) == 1:
            # compute(smiles_list) のみ
            result = plugin.compute_fn(smiles_list)
        elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            # **kwargs あり
            result = plugin.compute_fn(smiles_list, **kwargs)
        elif len(params) >= 2:
            # compute(smiles_list, config) 等
            result = plugin.compute_fn(smiles_list, **kwargs)
        else:
            result = plugin.compute_fn(smiles_list)

        # 戻り値の検証
        if result is None:
            logger.warning(
                f"⚠️ プラグイン '{plugin.name}': compute() が None を返しました。無視します。"
            )
            return [None] * len(smiles_list)

        return result

    except Exception as e:
        logger.warning(
            f"⚠️ プラグイン '{plugin.name}' の実行中にエラー: {e}。この記述子は無視されます。"
        )
        return [None] * len(smiles_list)
