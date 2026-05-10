"""
backend/descriptors/loader.py

システムの指定ディレクトリやユーザーが追加したディレクトリから、
動的に DescriptorFunction 実装クラスを検索・ロードするクラス。
また、ロードしたクラスの __call__ 引き数シグネチャをパースし、
UI（React/Streamlit/NiceGUI等）用の JSON Schema 表現を自動的に生成する。
"""
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Union

from .base import DescriptorFunction
from ..utils.warning_manager import WarningCollector
from ..utils.docstring_parser import parse_chemai_docstring

class DescriptorLoader:
    """
    記述子モジュールを動的にロードし、
    引数情報から自動で設定用JSON Schemaを生成する。
    """
    
    def __init__(self, user_dir: Union[str, Path], system_dir: Union[str, Path], protected: bool = True):
        self.user_dir = Path(user_dir)
        self.system_dir = Path(system_dir)
        self.protected = protected  # Trueならシステムモジュールは編集不可として扱う
        self._registry: Dict[str, dict] = {}
        self._warnings = WarningCollector()
    
    def load_all(self) -> Dict[str, dict]:
        """設定されたパスから全記述子をロードし、メタ情報付きで登録する。"""
        # 1. システム記述子（保護）
        if self.system_dir.exists():
            for py_file in self.system_dir.rglob("*.py"):
                if py_file.name.startswith("_"): continue
                self._load_single(py_file, is_system=True)
        else:
            self._warnings.add("WARNING", "DescriptorLoader", f"System dir {self.system_dir} not found")
        
        # 2. ユーザー記述子（編集可能）
        if self.user_dir.exists():
            for py_file in self.user_dir.rglob("*.py"):
                if py_file.name.startswith("_"): continue
                self._load_single(py_file, is_system=False)
        else:
            self._warnings.add("INFO", "DescriptorLoader", f"User dir {self.user_dir} not found, proceeding without it.")
            
        return self._registry
    
    def get_warnings(self) -> list:
        """ロードプロセスでの警告を取得する"""
        return self._warnings.get_all()
        
    def _load_single(self, py_file: Path, is_system: bool) -> None:
        """単一の.pyファイルをロードし、DescriptorFunction継承クラスを探す"""
        try:
            # 動的インポート
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # モジュール内のクラスを走査
            for name in dir(module):
                obj = getattr(module, name)
                
                # クラスであり、かつ DescriptorFunction を継承している（かつ基底クラスそのものではない）場合
                if (inspect.isclass(obj) and issubclass(obj, DescriptorFunction) and obj is not DescriptorFunction):
                    try:
                        instance = obj()
                    except Exception as ie:
                        self._warnings.add("ERROR", py_file.name, f"初期化失敗: {ie}", name)
                        continue
                    
                    # シグネチャ検証
                    valid, msg = instance.validate_signature()
                    if not valid:
                        self._warnings.add(
                            level="WARNING",
                            module=py_file.name,
                            descriptor=name,
                            message=f"シグネチャ検証失敗: {msg} → スキップ"
                        )
                        continue
                        
                    # メタ情報統合
                    metadata = getattr(instance, "metadata", {})
                    if not metadata:
                        # metadataプロパティが空の場合はdocstringパーサーから取得を試みる
                        metadata = parse_chemai_docstring(inspect.getdoc(obj))
                        
                    metadata.update({
                        "module_path": str(py_file),
                        "is_system": is_system,
                        "editable": not (is_system and self.protected),
                        "ui_schema": self._generate_ui_schema(instance, metadata),
                        "function_ref": instance,
                    })
                    
                    key = f"{metadata.get('engine', 'unknown')}_{metadata.get('name', name)}"
                    self._registry[key] = metadata
                    
        except Exception as e:
            self._warnings.add("ERROR", py_file.name, f"ロード失敗: {e}")

    def _generate_ui_schema(self, descriptor: DescriptorFunction, metadata: dict) -> dict:
        """
        __call__ メソッドのシグネチャをリフレクション探索し、
        自動的に JSON Schema フォーマットの UI 定義を生成する。
        """
        sig = inspect.signature(descriptor.__call__)
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "ui_hints": {
                "layout": "accordion",
                "searchable": True,
                "collapsible_groups": {}
            }
        }
        
        from typing import get_origin, get_args, Union
        
        for param_name, param in sig.parameters.items():
            if param_name == 'smiles' or param_name == 'self':
                continue
                
            param_info = {}
            annotation = param.annotation
            
            # 1. 型推論
            if annotation == inspect.Parameter.empty:
                if param.default != inspect.Parameter.empty:
                    if isinstance(param.default, bool): param_info["type"] = "boolean"
                    elif isinstance(param.default, int): param_info["type"] = "integer"
                    elif isinstance(param.default, float): param_info["type"] = "number"
                    elif isinstance(param.default, list):
                        param_info["type"] = "array"
                        param_info["items"] = {"type": "string"} # デフォルト
                    else:
                        param_info["type"] = "string"
                else:
                    param_info["type"] = "string"
            elif annotation is int: param_info["type"] = "integer"
            elif annotation is float: param_info["type"] = "number"
            elif annotation is bool: param_info["type"] = "boolean"
            elif annotation is str: param_info["type"] = "string"
            else:
                orig = get_origin(annotation)
                if orig is list or orig is list:  # noqa: UP006
                    param_info["type"] = "array"
                    try:
                        item_type = get_args(annotation)[0]
                        param_info["items"] = {"type": "string" if item_type is str else "number"}
                    except IndexError:
                        param_info["items"] = {"type": "string"}
                elif orig == Union:
                    args = [a for a in get_args(annotation) if a is not type(None)]
                    param_info["type"] = "string" if str in args else "number"
                else:
                    param_info["type"] = "string"
                    
            # 2. デフォルト値
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
                
            # 3. メタデータの取り込み (docstring等から)
            meta_params = metadata.get("params", {})
            if param_name in meta_params:
                meta = meta_params[param_name]
                if "description" in meta:
                    param_info["description"] = meta["description"]
                    param_info["ui:help"] = meta["description"]
                if "options" in meta:
                    param_info["enum"] = meta["options"]
                if "min" in meta: param_info["minimum"] = meta["min"]
                if "minimum" in meta: param_info["minimum"] = meta["minimum"]
                if "max" in meta: param_info["maximum"] = meta["max"]
                if "maximum" in meta: param_info["maximum"] = meta["maximum"]

            # UIヒント
            if param_info.get("type") in ["array", "object"]:
                param_info["ui:widget"] = "collapsible"
                
            schema["properties"][param_name] = param_info
            
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
                
        return schema
