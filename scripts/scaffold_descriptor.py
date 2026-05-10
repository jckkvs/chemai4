"""
scripts/scaffold_descriptor.py

CLIから簡単な質問に答えるか引数で指定することで、
backend/descriptors/user/ に配置するユーザー定義の記述子テンプレートコードを自動生成する。
"""
import os
import argparse
from pathlib import Path

def generate_template(name: str, engine: str, category: str, output_dir: str):
    """
    指定されたパラメータを用いてテンプレートコードを生成し、ファイルに書き込む。
    """
    template = f'''"""
:::chemai-descriptor
name: {name}
engine: {engine}
category: {category}
description: ユーザー定義の {name} 記述子
params:
  example_param:
    type: int
    default: 1
    description: テンプレートのサンプルパラメータ
    min: 1
    max: 10
:::
"""
from typing import List
import numpy as np

# backendのルートパスまたはインストール済みモジュールとしてのパスに合わせてインポート
try:
    from backend.descriptors.base import DescriptorFunction
except ImportError:
    # 開発環境用フォールバック
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from backend.descriptors.base import DescriptorFunction

class {name.replace(" ", "")}Descriptor(DescriptorFunction):
    
    @property
    def metadata(self) -> dict:
        return self.__class__.__doc__
    
    def __call__(self, smiles: List[str], example_param: int = 1) -> List[float]:
        """
        記述量計算の実装
        
        Parameters
        ----------
        smiles : List[str]
            SMILES文字列リスト
        example_param : int
            サンプルパラメータ
        
        Returns
        -------
        List[float]
            計算結果
        """
        results = []
        for smi in smiles:
            # TODO: RDKit等を利用して対象の化合物の特性を計算するロジックをここに実装します
            # 例: ダミーとしてランダムな値を返す
            results.append(np.random.normal(loc=example_param, scale=0.1))
            
        return results
'''
    os.makedirs(output_dir, exist_ok=True)
    filename = name.lower().replace(" ", "_") + ".py"
    filepath = Path(output_dir) / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(template)
        
    print(f"Scaffold successfully generated at {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate a descriptor template.")
    parser.add_argument("--name", default="MyCustomMetric", help="Descriptor display name")
    parser.add_argument("--engine", default="custom", help="Engine identifier (e.g. custom, rdkit, xtb)")
    parser.add_argument("--category", default="Other", help="Category (e.g. Structure, Properties)")
    parser.add_argument("--output", default="backend/descriptors/user/", help="Output directory path")
    args = parser.parse_args()
    
    generate_template(args.name, args.engine, args.category, args.output)

if __name__ == "__main__":
    main()
