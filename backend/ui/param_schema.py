"""
backend/ui/param_schema.py

パラメータ自動イントロスペクションエンジン。

任意のPythonクラス（sklearn estimator, ChemAdapter等）の
__init__シグネチャを解析し、UIウィジェット生成に必要な
ParamSpecリストに変換する。

Usage:
    from backend.ui.param_schema import introspect_params
    specs = introspect_params(RandomForestRegressor)
    # → [ParamSpec(name='n_estimators', param_type='int', default=100, ...), ...]
"""
from __future__ import annotations

import enum
import inspect
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ParamSpec: UIウィジェット生成の仕様
# ─────────────────────────────────────────────────────────────


@dataclass
class ParamSpec:
    """
    1つのパラメータのUIウィジェット仕様。

    Attributes:
        name:        パラメータ名（Pythonの引数名）
        param_type:  UIウィジェット型 ("bool", "int", "float", "str",
                     "select", "multiselect", "text", "union")
        default:     デフォルト値
        choices:     select/multiselect用の選択肢リスト
        min_val:     数値の最小値（Noneなら制限なし）
        max_val:     数値の最大値（Noneなら制限なし）
        step:        数値の刻み幅
        description: パラメータの説明文
        group:       UIグループ名 ("basic" or "advanced")
        nullable:    Noneを許容するか
        type_hint_raw: 生の型ヒント文字列（デバッグ用）
    """
    name: str
    param_type: str  # "bool", "int", "float", "str", "select", "multiselect", "text", "union"
    default: Any = None
    choices: list[Any] = field(default_factory=list)
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    description: str = ""
    group: str = "basic"       # "basic" or "advanced"
    nullable: bool = False
    type_hint_raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON直列化可能な辞書に変換（Django API用）。"""
        return {
            "name": self.name,
            "param_type": self.param_type,
            "default": self.default,
            "choices": self.choices,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "step": self.step,
            "description": self.description,
            "group": self.group,
            "nullable": self.nullable,
            "type_hint_raw": self.type_hint_raw,
        }


# ─────────────────────────────────────────────────────────────
# 上級者パラメータの自動判定
# ─────────────────────────────────────────────────────────────

# "advanced"グループに自動分類するパラメータ名パターン
_ADVANCED_PARAMS: set[str] = {
    "random_state", "n_jobs", "verbose", "warm_start",
    "copy_X", "copy", "tol", "max_iter", "solver",
    "fit_intercept", "normalize", "positive",
    "class_weight", "dual", "penalty",
    "presort", "ccp_alpha", "max_samples",
    "oob_score", "bootstrap", "criterion",
    "min_weight_fraction_leaf", "max_leaf_nodes",
    "min_impurity_decrease",
}

# 除外するパラメータ（UIに出すべきでない）
_SKIP_PARAMS: set[str] = {
    "self", "args", "kwargs",
    "progress_callback", "selected_descriptors",
}


# ─────────────────────────────────────────────────────────────
# 数値パラメータの推奨レンジ
# ─────────────────────────────────────────────────────────────

_PARAM_RANGES: dict[str, dict[str, Any]] = {
    # sklearn共通
    "n_estimators":       {"min_val": 1,    "max_val": 10000, "step": 10},
    "max_depth":          {"min_val": 1,    "max_val": 200,   "step": 1},
    "min_samples_split":  {"min_val": 2,    "max_val": 100,   "step": 1},
    "min_samples_leaf":   {"min_val": 1,    "max_val": 100,   "step": 1},
    "max_features":       {},  # auto/sqrt/None — unionとして処理
    "learning_rate":      {"min_val": 0.001, "max_val": 10.0, "step": 0.01},
    "alpha":              {"min_val": 0.0,  "max_val": 100.0, "step": 0.01},
    "l1_ratio":           {"min_val": 0.0,  "max_val": 1.0,   "step": 0.01},
    "C":                  {"min_val": 0.001, "max_val": 1000.0, "step": 0.1},
    "gamma":              {},  # "scale"/"auto" or float
    "epsilon":            {"min_val": 0.0,  "max_val": 10.0,  "step": 0.01},
    "subsample":          {"min_val": 0.1,  "max_val": 1.0,   "step": 0.05},
    "colsample_bytree":   {"min_val": 0.1,  "max_val": 1.0,   "step": 0.05},
    "reg_alpha":          {"min_val": 0.0,  "max_val": 100.0, "step": 0.01},
    "reg_lambda":         {"min_val": 0.0,  "max_val": 100.0, "step": 0.01},
    "n_components":       {"min_val": 1,    "max_val": 1000,  "step": 1},
    "latent_dim":         {"min_val": 8,    "max_val": 1024,  "step": 8},
    # ChemAdapter共通
    "morgan_radius":      {"min_val": 1,    "max_val": 6,     "step": 1},
    "morgan_bits":        {"min_val": 64,   "max_val": 8192,  "step": 64},
    "rdkit_fp_bits":      {"min_val": 64,   "max_val": 8192,  "step": 64},
    "radius":             {"min_val": 1,    "max_val": 6,     "step": 1},
    "fp_size":            {"min_val": 64,   "max_val": 8192,  "step": 64},
    "features_dim":       {"min_val": 32,   "max_val": 1024,  "step": 32},
    "gfn":                {"min_val": 0,    "max_val": 2,     "step": 1},
    "batch_size":         {"min_val": 1,    "max_val": 512,   "step": 1},
    "timeout":            {"min_val": 10,   "max_val": 3600,  "step": 10},
    "cv_folds":           {"min_val": 2,    "max_val": 30,    "step": 1},
    "timeout_seconds":    {"min_val": 10,   "max_val": 3600,  "step": 10},
    "n_iter":             {"min_val": 1,    "max_val": 1000,  "step": 10},
    "max_bins":           {"min_val": 2,    "max_val": 512,   "step": 1},
    "num_leaves":         {"min_val": 2,    "max_val": 1024,  "step": 1},
}


# select用の既知の選択肢
_KNOWN_CHOICES: dict[str, list[str]] = {
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson",
                  "gini", "entropy", "log_loss"],
    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga",
               "lbfgs", "newton-cg", "newton-cholesky", "liblinear"],
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "loss": ["squared_error", "absolute_error", "huber", "quantile",
             "log_loss", "exponential", "deviance"],
    "booster": ["gbtree", "gblinear", "dart"],
    "boosting_type": ["gbdt", "dart", "rf"],
    "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
    "scaler": ["auto", "standard", "minmax", "robust", "maxabs", "none"],
    "calculator_type": ["ecfp", "fcfp", "maccs", "rdkit", "avalon",
                        "topological_torsion", "atom_pair"],
    "descriptor_type": ["rdkit2d", "rdkit2dnormalized", "rdkitfpbits",
                        "morgancount", "morganbits"],
    "parameterization": ["default_turbomole", "fine", "bp-tzvp"],
}


# ─────────────────────────────────────────────────────────────
# メイン関数
# ─────────────────────────────────────────────────────────────

def introspect_params(
    cls: type,
    *,
    instance: Any | None = None,
    skip_params: set[str] | None = None,
    extra_descriptions: dict[str, str] | None = None,
) -> list[ParamSpec]:
    """
    任意のPythonクラスの__init__パラメータを自動解析し、
    UIウィジェット生成用のParamSpecリストを返す。

    Args:
        cls: 解析対象のクラス（sklearn estimator, ChemAdapter等）
        instance: 既存インスタンス（get_params()等からデフォルト値を取得）
        skip_params: 追加でスキップするパラメータ名
        extra_descriptions: パラメータ名→説明文の追加辞書

    Returns:
        list[ParamSpec]: UIウィジェット仕様のリスト
    """
    skip = _SKIP_PARAMS | (skip_params or set())
    descriptions = extra_descriptions or {}

    # シグネチャ取得
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        sig = inspect.signature(cls)

    # 型ヒント取得（できれば）
    try:
        hints = get_type_hints(cls.__init__)
    except Exception:
        hints = {}

    # sklearn: get_params()からデフォルト値を補完
    sklearn_defaults = {}
    if instance is not None and hasattr(instance, "get_params"):
        try:
            sklearn_defaults = instance.get_params(deep=False)
        except Exception:
            pass
    elif hasattr(cls, "get_params"):
        try:
            sklearn_defaults = cls().get_params(deep=False)
        except Exception:
            pass

    # docstringからパラメータ説明を抽出
    doc_descriptions = _extract_docstring_params(cls)

    specs: list[ParamSpec] = []
    for param_name, param in sig.parameters.items():
        if param_name in skip:
            continue
        if param_name.startswith("_"):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # デフォルト値の決定
        default = param.default
        if default is inspect.Parameter.empty:
            if param_name in sklearn_defaults:
                default = sklearn_defaults[param_name]
            else:
                default = None

        # sklearn get_params()のデフォルト値で上書き
        if param_name in sklearn_defaults:
            default = sklearn_defaults[param_name]

        # 型ヒントの取得
        hint = hints.get(param_name)
        hint_str = str(hint) if hint else ""

        # 説明文の決定（優先度: 引数 > docstring > 空）
        desc = descriptions.get(
            param_name,
            doc_descriptions.get(param_name, ""),
        )

        # グループの決定
        group = "advanced" if param_name in _ADVANCED_PARAMS else "basic"

        # 型推論 → ParamSpec生成
        spec = _infer_param_spec(
            name=param_name,
            default=default,
            hint=hint,
            hint_str=hint_str,
            description=desc,
            group=group,
        )
        specs.append(spec)

    return specs


def introspect_adapter(adapter_instance: Any) -> list[ParamSpec]:
    """
    BaseChemAdapterサブクラスのインスタンスからパラメータを取得。

    Args:
        adapter_instance: アダプタインスタンス

    Returns:
        list[ParamSpec]: UIウィジェット仕様のリスト
    """
    cls = type(adapter_instance)
    return introspect_params(cls, instance=adapter_instance)


def introspect_adapter_class(adapter_cls: type) -> list[ParamSpec]:
    """
    BaseChemAdapterサブクラスからパラメータを取得。

    Args:
        adapter_cls: アダプタクラス

    Returns:
        list[ParamSpec]: UIウィジェット仕様のリスト
    """
    return introspect_params(adapter_cls)


# ─────────────────────────────────────────────────────────────
# 型推論ロジック
# ─────────────────────────────────────────────────────────────

def _infer_param_spec(
    name: str,
    default: Any,
    hint: Any | None,
    hint_str: str,
    description: str,
    group: str,
) -> ParamSpec:
    """デフォルト値と型ヒントからParamSpecを推論する。"""

    nullable = default is None
    choices = _KNOWN_CHOICES.get(name, [])
    ranges = _PARAM_RANGES.get(name, {})

    # 1. 明示的な既知選択肢がある場合
    if choices and not isinstance(default, (bool, int, float)):
        return ParamSpec(
            name=name,
            param_type="select",
            default=default,
            choices=choices,
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    # 2. 型ヒントからの推論
    if hint is not None:
        spec = _infer_from_hint(name, default, hint, hint_str, description, group, nullable, ranges)
        if spec is not None:
            return spec

    # 3. デフォルト値からの推論
    if default is not None and default is not inspect.Parameter.empty:
        return _infer_from_default(name, default, hint_str, description, group, nullable, ranges)

    # 4. フォールバック: テキスト入力
    return ParamSpec(
        name=name,
        param_type="text",
        default="" if default is None else str(default),
        description=description,
        group=group,
        nullable=True,
        type_hint_raw=hint_str,
    )


def _infer_from_hint(
    name: str,
    default: Any,
    hint: Any,
    hint_str: str,
    description: str,
    group: str,
    nullable: bool,
    ranges: dict,
) -> ParamSpec | None:
    """型ヒントからParamSpecを生成する。"""

    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", ())

    # Union型（Optional含む）: int | None, str | int 等
    if origin is typing.Union:
        non_none_args = [a for a in args if a is not type(None)]
        has_none = type(None) in args

        if len(non_none_args) == 1:
            # Optional[X] → X として処理（nullable=True）
            return _infer_from_hint(
                name, default, non_none_args[0], hint_str,
                description, group, True, ranges,
            )
        elif len(non_none_args) >= 2:
            # int | str 等の複合型 → テキスト入力
            return ParamSpec(
                name=name,
                param_type="text",
                default=str(default) if default is not None else "",
                description=description + " (複数の型を受付)",
                group=group,
                nullable=has_none or nullable,
                type_hint_raw=hint_str,
            )

    # Literal型: Literal["auto", "scale"]
    if origin is typing.Literal:
        return ParamSpec(
            name=name,
            param_type="select",
            default=default,
            choices=list(args),
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    # list型
    if origin in (list, typing.List):
        return ParamSpec(
            name=name,
            param_type="multiselect",
            default=default if default is not None else [],
            choices=_KNOWN_CHOICES.get(name, []),
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    # 基本型
    if hint is bool:
        return ParamSpec(
            name=name,
            param_type="bool",
            default=default if isinstance(default, bool) else False,
            description=description,
            group=group,
            nullable=False,
            type_hint_raw=hint_str,
        )
    if hint is int:
        return ParamSpec(
            name=name,
            param_type="int",
            default=default if isinstance(default, int) else 0,
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
            **ranges,
        )
    if hint is float:
        return ParamSpec(
            name=name,
            param_type="float",
            default=default if isinstance(default, (int, float)) else 0.0,
            step=ranges.get("step", 0.01),
            min_val=ranges.get("min_val"),
            max_val=ranges.get("max_val"),
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )
    if hint is str:
        choices = _KNOWN_CHOICES.get(name, [])
        if choices:
            return ParamSpec(
                name=name,
                param_type="select",
                default=default,
                choices=choices,
                description=description,
                group=group,
                nullable=nullable,
                type_hint_raw=hint_str,
            )
        return ParamSpec(
            name=name,
            param_type="str",
            default=default if isinstance(default, str) else "",
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    # Enum型
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return ParamSpec(
            name=name,
            param_type="select",
            default=default.value if isinstance(default, enum.Enum) else default,
            choices=[e.value for e in hint],
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    return None


def _infer_from_default(
    name: str,
    default: Any,
    hint_str: str,
    description: str,
    group: str,
    nullable: bool,
    ranges: dict,
) -> ParamSpec:
    """デフォルト値の型から推論する。"""

    if isinstance(default, bool):
        return ParamSpec(
            name=name,
            param_type="bool",
            default=default,
            description=description,
            group=group,
            nullable=False,
            type_hint_raw=hint_str,
        )
    if isinstance(default, int):
        return ParamSpec(
            name=name,
            param_type="int",
            default=default,
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
            **ranges,
        )
    if isinstance(default, float):
        return ParamSpec(
            name=name,
            param_type="float",
            default=default,
            step=ranges.get("step", 0.01),
            min_val=ranges.get("min_val"),
            max_val=ranges.get("max_val"),
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )
    if isinstance(default, str):
        choices = _KNOWN_CHOICES.get(name, [])
        if choices:
            return ParamSpec(
                name=name,
                param_type="select",
                default=default,
                choices=choices,
                description=description,
                group=group,
                nullable=nullable,
                type_hint_raw=hint_str,
            )
        return ParamSpec(
            name=name,
            param_type="str",
            default=default,
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )
    if isinstance(default, (list, tuple)):
        return ParamSpec(
            name=name,
            param_type="multiselect",
            default=list(default),
            choices=_KNOWN_CHOICES.get(name, list(default)),
            description=description,
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )
    if isinstance(default, dict):
        return ParamSpec(
            name=name,
            param_type="text",
            default=str(default),
            description=description + " (JSON形式)",
            group=group,
            nullable=nullable,
            type_hint_raw=hint_str,
        )

    # フォールバック
    return ParamSpec(
        name=name,
        param_type="text",
        default=str(default) if default is not None else "",
        description=description,
        group=group,
        nullable=nullable,
        type_hint_raw=hint_str,
    )


# ─────────────────────────────────────────────────────────────
# Docstring解析
# ─────────────────────────────────────────────────────────────

def _extract_docstring_params(cls: type) -> dict[str, str]:
    """
    クラスのdocstringからパラメータ説明を抽出する。

    numpydoc / Google style / reStructuredText を解析する。

    numpydocの構造:
        Parameters
        ----------
        param_name : type_info, default=value
            Description line 1.
            Description line 2.

    Returns:
        {param_name: "Description line 1. Description line 2."}
        （型情報は含まない）
    """
    doc = cls.__doc__ or ""
    init_doc = getattr(cls.__init__, "__doc__", "") or ""
    full_doc = doc + "\n" + init_doc

    descriptions: dict[str, str] = {}
    lines = full_doc.split("\n")
    current_param: str | None = None
    current_desc: list[str] = []


    for i, line in enumerate(lines):
        stripped = line.strip()
        raw_indent = len(line) - len(line.lstrip()) if line.strip() else 0

        # "Parameters" セクションの検出
        if stripped in ("Parameters", "Attributes", "Keyword Arguments", "Args:", "Attributes:"):
            continue

        # "----------" セパレータの検出
        if stripped and all(c == "-" for c in stripped) and len(stripped) >= 3:
            continue
        else:
            pass

        # 他のセクション（Returns, Notes等）に入ったら終了
        if (
            stripped in ("Returns", "Yields", "Raises", "Notes", "References",
                         "See Also", "Examples", "Warnings", "Methods", "Returns:", "Yields:", "Raises:")
            and i + 1 < len(lines)
            and (lines[i + 1].strip().startswith("---") or stripped.endswith(":"))
        ):
            if current_param and current_desc:
                descriptions[current_param] = " ".join(current_desc).strip()
            current_param = None
            current_desc = []
            continue

        # numpydoc style: "param_name : type_info"
        # 条件: コロンの前が短い英数字+下線の文字列、インデントが4スペース
        if ":" in stripped and not stripped.startswith(":") and not stripped.startswith(".."):
            parts = stripped.split(":", 1)
            candidate = parts[0].strip().split("(")[0].strip()

            # パラメータ名として妥当か
            if (
                candidate
                and candidate.replace("_", "").isalnum()
                and len(candidate) < 40
                and not candidate[0].isupper()
                and not candidate.startswith("{")
                and raw_indent <= 8
            ):
                # 前のパラメータを保存
                if current_param and current_desc:
                    descriptions[current_param] = " ".join(current_desc).strip()

                current_param = candidate
                # numpydoc: コロンの後ろは型情報なので説明文に含めない
                # Google style の場合もあるので、コロンの後に文字列があればそれを説明文の先頭とする
                desc_candidate = parts[1].strip()
                current_desc = [desc_candidate] if desc_candidate else []
                continue

        # reStructuredText: ":param param_name:"
        if stripped.startswith(":param "):
            param_part = stripped[7:]
            if ":" in param_part:
                pname, pdesc = param_part.split(":", 1)
                pname = pname.strip()
                pdesc = pdesc.strip()
                if current_param and current_desc:
                    descriptions[current_param] = " ".join(current_desc).strip()
                current_param = pname
                current_desc = [pdesc] if pdesc else []
                continue

        # 続きの説明行（インデントが深い行）
        if current_param and stripped:
            # セクション区切り行でない限り追加
            if not (all(c == "-" for c in stripped) and len(stripped) >= 3):
                # .. note:: 等のディレクティブは含めても良い
                # versionchanged, versionadded, deprecated は含めない
                if stripped.startswith(".. versionchanged") or \
                   stripped.startswith(".. versionadded") or \
                   stripped.startswith(".. deprecated"):
                    continue
                current_desc.append(stripped)

        # 空行でパラメータの説明が途切れる場合（空行2連続は段落区切り）
        if not stripped and current_param and current_desc:
            # 次行がインデント付きなら同じパラメータの続き
            if i + 1 < len(lines) and lines[i + 1].strip():
                next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                if next_indent >= 8:
                    continue  # 同じパラメータの続き

    if current_param and current_desc:
        descriptions[current_param] = " ".join(current_desc).strip()

    return descriptions


# ─────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────

def apply_params(specs: list[ParamSpec], user_values: dict[str, Any]) -> dict[str, Any]:
    """
    ParamSpecリストとユーザー入力値から、コンストラクタに渡す
    パラメータ辞書を生成する。

    型変換とバリデーションを行い、デフォルト値と異なるもののみ返す。
    """
    params: dict[str, Any] = {}

    for spec in specs:
        if spec.name not in user_values:
            continue

        raw_val = user_values[spec.name]

        # 変更なし（デフォルトと同じ）ならスキップ
        if raw_val == spec.default:
            continue

        # 型変換
        try:
            converted = _convert_value(raw_val, spec)
            params[spec.name] = converted
        except (ValueError, TypeError) as e:
            logger.warning(f"パラメータ変換エラー: {spec.name}={raw_val!r}: {e}")

    return params


def _convert_value(raw_val: Any, spec: ParamSpec) -> Any:
    """ユーザー入力値をPython型に変換する。"""

    if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
        if spec.nullable:
            return None
        return spec.default

    if spec.param_type == "bool":
        if isinstance(raw_val, bool):
            return raw_val
        if isinstance(raw_val, str):
            return raw_val.lower() in ("true", "1", "yes", "on")
        return bool(raw_val)

    if spec.param_type == "int":
        return int(float(raw_val))

    if spec.param_type == "float":
        return float(raw_val)

    if spec.param_type == "str":
        return str(raw_val)

    if spec.param_type == "select":
        return raw_val

    if spec.param_type == "multiselect":
        if isinstance(raw_val, str):
            return [s.strip() for s in raw_val.split(",") if s.strip()]
        return list(raw_val)

    if spec.param_type == "text":
        text = str(raw_val).strip()
        # 数値として解釈を試みる
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            pass
        if text.lower() == "none":
            return None
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        return text

    return raw_val


def get_basic_specs(specs: list[ParamSpec]) -> list[ParamSpec]:
    """basicグループのParamSpecのみ返す。"""
    return [s for s in specs if s.group == "basic"]


def get_advanced_specs(specs: list[ParamSpec]) -> list[ParamSpec]:
    """advancedグループのParamSpecのみ返す。"""
    return [s for s in specs if s.group == "advanced"]
