"""
backend/utils/llm_sandbox.py
"""
import ast
import logging
from typing import Dict, Callable, Any

logger = logging.getLogger(__name__)
ALLOWED = {"numpy", "rdkit", "scipy", "math", "typing", "pandas", "backend.descriptors.base"}
DANGEROUS = {"eval", "exec", "compile", "open", "input", "os.system", "subprocess", "__import__"}

class SecureValidator(ast.NodeVisitor):
    def __init__(self):
        self.errs: list[str] = []
        
    def visit_Import(self, n):
        for a in n.names:
            if a.name.split(".")[0] not in ALLOWED and a.name not in ALLOWED:
                self.errs.append(f"Blocked import: {a.name}")
        self.generic_visit(n)
        
    def visit_ImportFrom(self, n):
        if n.module and n.module.split(".")[0] not in ALLOWED and n.module not in ALLOWED:
            self.errs.append(f"Blocked import from: {n.module}")
        self.generic_visit(n)
        
    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id in DANGEROUS:
            self.errs.append(f"Dangerous call: {n.func.id}")
        elif isinstance(n.func, ast.Attribute) and n.func.attr in DANGEROUS:
             self.errs.append(f"Dangerous attribute call: {n.func.attr}")
        self.generic_visit(n)

class SecurityError(Exception):
    pass

def safe_exec_llm(code: str, ctx: Dict[str, Any]) -> Callable:
    """
    LLMが生成したコード（文字列）をASTで検証してから安全に評価し、
    結果として関数などを返す。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"構文エラーがあります: {e}")

    v = SecureValidator()
    v.visit(tree)
    if v.errs:
        raise SecurityError("セキュリティ検証に失敗しました:\n" + "\n".join(v.errs))
    
    safe_builtins = {
        k: __builtins__[k] for k in (
            "print", "len", "range", "list", "dict", "int", "float", 
            "str", "tuple", "set", "abs", "min", "max", "sum", 
            "round", "enumerate", "zip", "bool", "None", "True", "False"
        ) if k in __builtins__
    }
    # Some builtins are not in dict form depending on environment, we use globals() check if needed
    ns = {"__builtins__": safe_builtins, **ctx}
    
    # 実行
    exec(compile(tree, "<llm_generated_code>", "exec"), ns)
    
    # 例えば記述子計算関数の定義を期待する場合
    if "compute_descriptor" not in ns:
        logger.warning("compute_descriptor が定義されていません。利用可能な関数を返します。")
        # 最初の関数を返すなどのフォールバック
        funcs = [v for k, v in ns.items() if callable(v) and not k.startswith("__")]
        if funcs:
            return funcs[0]
        raise ValueError("Must define an entrypoint function e.g., compute_descriptor(smiles, **kwargs)")
        
    return ns["compute_descriptor"]
