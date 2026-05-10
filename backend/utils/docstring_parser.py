"""
backend/utils/docstring_parser.py

docstring内に埋め込まれたYAMLライクな設定（:::chemai-descriptor 等）を
解析し、辞書として抽出・パースするユーティリティ。
"""
import yaml
import re

def parse_chemai_docstring(docstring: str) -> dict:
    """
    docstringから `:::chemai-descriptor` ブロックを探し出し、内部のYAMLをパースして返す。
    
    Args:
        docstring: 関数の __doc__ などから取得した文字列
        
    Returns:
        抽出されたメタデータの辞書（見つからなかった場合やパースしっぱいの場合は空辞書）
    """
    if not docstring:
        return {}
        
    # :::chemai-descriptor から ::: までのブロックを抽出
    pattern = r":::chemai-descriptor\s*(.*?)\s*:::"
    match = re.search(pattern, docstring, re.DOTALL)
    
    if not match:
        return {}
        
    yaml_content = match.group(1).strip()
    if not yaml_content:
        return {}
        
    try:
        # yamlとしてパースを試みる
        parsed = yaml.safe_load(yaml_content)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except yaml.YAMLError as e:
        # パース失敗時は生の文字列を警告がわりに入れる（あるいは空にする）
        print(f"YAML Parse Error in docstring: {e}")
        return {}
