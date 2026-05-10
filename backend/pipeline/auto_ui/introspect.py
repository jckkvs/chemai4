"""
backend/pipeline/auto_ui/introspect.py

任意estimatorの__init__引数を解析し、
自動でUI設定用JSON Schemaを生成するエンジン。
"""
import inspect
from typing import Dict, Any, Type, Optional
from sklearn.base import BaseEstimator

class EstimatorIntrospector:
    @classmethod
    def introspect(cls, estimator_class: Type[BaseEstimator]) -> dict:
        """
        estimatorクラスからUI schemaを自動生成する。
        """
        try:
            sig = inspect.signature(estimator_class.__init__)
        except TypeError:
            # 組み込みクラスなどで取得できない場合
            return {"type": "object", "properties": {}}
            
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "ui:layout": "accordion",
            "ui:searchable": True
        }
        
        docstring = inspect.getdoc(estimator_class.__init__) or inspect.getdoc(estimator_class) or ""
        param_docs = cls._parse_docstring_params(docstring)
        
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'args', 'kwargs'):
                continue
                
            param_schema = {}
            annotation = param.annotation
            
            # 型推論（フェイルセーフ）
            try:
                if annotation != inspect.Parameter.empty:
                    param_schema.update(cls._annotation_to_schema(annotation))
                elif param.default != inspect.Parameter.empty:
                    param_schema["type"] = cls._infer_type_from_value(param.default)
                    if isinstance(param.default, (list, tuple)) and param.default:
                        param_schema["items"] = {"type": cls._infer_type_from_value(param.default[0])}
                else:
                    param_schema["type"] = "string"
            except Exception as e:
                # 複雑な型やCallable等で失敗した場合は安全側に倒す
                import logging
                logging.getLogger(__name__).warning(f"Type inference failed for {param_name}: {e}")
                param_schema["type"] = "string"
                param_schema["ui:widget"] = "text"
                
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
                
            if param_name in param_docs:
                param_schema["description"] = param_docs[param_name]
                param_schema["ui:help"] = param_docs[param_name]
                
                # Option（Literalライクな記述）があったらenum化
                range_info = cls._extract_range_from_docstring(param_docs[param_name])
                if range_info and param_schema.get("type") in ["number", "integer"]:
                    param_schema.update(range_info)
                    
            schema["properties"][param_name] = param_schema
            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)
                
        return schema

    @staticmethod
    def _annotation_to_schema(annotation) -> dict:
        from typing import get_origin, get_args, Union
        try:
            # Literal support
            # since literal isn't easily imported depending on python version
            if hasattr(annotation, "__origin__") and str(annotation.__origin__) == "typing.Literal":
                return {
                    "type": "string",
                    "enum": list(get_args(annotation)),
                    "ui:widget": "select"
                }
        except Exception:
            pass
            
        origin = get_origin(annotation)
        if origin is None:
            if annotation is int: return {"type": "integer"}
            elif annotation is float: return {"type": "number"}
            elif annotation is bool: return {"type": "boolean"}
            elif annotation is str: return {"type": "string"}
        elif origin == Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return EstimatorIntrospector._annotation_to_schema(args[0])
            elif str in args:
                return {"type": "string"}
            else:
                return {"type": "number"}
        elif origin is list or getattr(annotation, "_name", "") == "List":
            args = get_args(annotation)
            item_schema = {"type": "string"}
            if args and args[0] != inspect.Parameter.empty:
                item_schema = EstimatorIntrospector._annotation_to_schema(args[0])
            return {"type": "array", "items": item_schema}
        return {"type": "string"}
        
    @staticmethod
    def _infer_type_from_value(value: Any) -> str:
        if isinstance(value, bool): return "boolean"
        elif isinstance(value, int): return "integer"
        elif isinstance(value, float): return "number"
        elif isinstance(value, str): return "string"
        elif isinstance(value, (list, tuple)): return "array"
        return "string"
        
    @staticmethod
    def _parse_docstring_params(docstring: str) -> Dict[str, str]:
        params = {}
        lines = docstring.split("\n")
        current_param = None
        for line in lines:
            line = line.strip()
            if line.lower().startswith("parameters"): continue
            if ":" in line and not line.startswith("-"):
                parts = line.split(":", 1)
                pname = parts[0].strip()
                if pname and not pname.startswith("Returns"):
                    current_param = pname
                    params[pname] = parts[1].strip() if len(parts) > 1 else ""
            elif current_param and line and not line.startswith("---"):
                params[current_param] += " " + line
        return params
        
    @staticmethod
    def _extract_range_from_docstring(text: str) -> Optional[Dict[str, float]]:
        import re
        pattern = r'([\d.]+)\s*[-~to]+\s*([\d.]+)'
        match = re.search(pattern, text)
        if match:
            return {
                "minimum": float(match.group(1)),
                "maximum": float(match.group(2))
            }
        return None
