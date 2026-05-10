"""
scripts/gen_tutorial_md.py

docstringを解析して MkDocs 向けのチュートリアルやAPIリファレンスを自動生成する。
"""
import inspect
import importlib
import textwrap
from pathlib import Path
import sys
import os

# プロジェクトルートにパスを通す
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_tutorial(module_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mod_name = module_path.replace("/", ".").replace(".py", "")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        print(f"[ERROR] Failed to import module {mod_name}: {e}")
        return

    generated_count = 0
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if name.startswith("_"):
            continue
            
        doc = inspect.getdoc(obj) or "*(No documentation available)*"
        md = output_dir / f"{name.lower()}.md"
        
        content = textwrap.dedent(f"""\
        # {name}
        
        {doc}
        
        ## 使用方法
        ```python
        from {mod_name} import {name}
        
        # TODO: Here is a placeholder for your code!
        # instance = {name}()
        ```
        """)
        
        md.write_text(content, encoding="utf-8")
        generated_count += 1
        
    if generated_count > 0:
        print(f"[OK] Generated {generated_count} tutorials in {output_dir}")
    else:
        print(f"[WARN] No public classes found in {mod_name} to generate tutorials.")

if __name__ == "__main__":
    docs_dir = project_root / "docs" / "tutorials"
    target_modules = [
        "backend/descriptors/base.py",
        "backend/pipeline/core.py"
    ]
    
    for tm in target_modules:
        if (project_root / tm).exists():
            generate_tutorial(tm, docs_dir)
