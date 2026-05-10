import os
import ast

def generate_ast_markdown(directory):
    md_content = []
    
    # Exclude venv, __pycache__, .git, etc
    exclude_dirs = ['.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.chemai_cache']
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, directory)
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source)
                    
                    classes = []
                    functions = []
                    
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            classes.append(node)
                        elif isinstance(node, ast.FunctionDef):
                            functions.append(node)
                            
                    if not classes and not functions:
                        continue
                        
                    md_content.append(f"### File: `{rel_path}`")
                    
                    for c in classes:
                        bases = [b.id for b in c.bases if isinstance(b, ast.Name)]
                        base_str = f" (Bases: {', '.join(bases)})" if bases else ""
                        md_content.append(f"- **Class `{c.name}`**{base_str}:")
                        doc = ast.get_docstring(c)
                        if doc:
                            first_line = doc.strip().split('\n')[0]
                            md_content.append(f"  - *Doc:* {first_line}")
                        
                        for sub_node in c.body:
                            if isinstance(sub_node, ast.FunctionDef):
                                if not sub_node.name.startswith('__') or sub_node.name == '__init__':
                                    args = [a.arg for a in sub_node.args.args]
                                    md_content.append(f"  - `def {sub_node.name}({', '.join(args)})`")
                                    
                    for f_node in functions:
                        if not f_node.name.startswith('_'):
                            args = [a.arg for a in f_node.args.args]
                            md_content.append(f"- **Function `{f_node.name}`({', '.join(args)})**")
                            doc = ast.get_docstring(f_node)
                            if doc:
                                first_line = doc.strip().split('\n')[0]
                                md_content.append(f"  - *Doc:* {first_line}")
                                
                    md_content.append("")
                    
                except Exception as e:
                    md_content.append(f"<!-- Failed to parse {rel_path}: {str(e)} -->")
                    
    return "\n".join(md_content)

def main():
    repo_dir = r"c:\Users\horie\chemai2"
    prompt_file = os.path.join(repo_dir, "PROJECT_REPRODUCTION_PROMPT.md")
    
    # Generate content for backend, frontend_nicegui, tests
    md_content = "\n\n---\n\n## 10. 🧬 全コードベース完全解析仕様 (Total Codebase Exact AST Blueprint)\n\n"
    md_content += "以下の仕様はプロジェクトの全ファイル群、およびその内部のすべてのクラス・関数のインターフェース定義を静的解析で忠実に書き下したものである。いかなる機能も見落とさず、すべてのシグネチャを完全実装せよ。\n\n"
    
    for folder in ['backend', 'frontend_nicegui', 'tests']:
        folder_path = os.path.join(repo_dir, folder)
        if os.path.exists(folder_path):
            md_content += f"## {folder.upper()} Directory\n\n"
            md_content += generate_ast_markdown(folder_path)
            
    with open(prompt_file, 'a', encoding='utf-8') as f:
        f.write(md_content)
        
    print("Successfully appended the ultimate AST breakdown to the prompt.")

if __name__ == "__main__":
    main()
