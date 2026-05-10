import sys
import os
sys.path.append('.')

import pandas as pd
from backend.llm.data_analyst import get_data_analyst, read_document_content

def main():
    print("Debugging LLM response...")
    
    # Load main data (CSV)
    csv_path = 'debug_samples/solubility_data.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded main data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Load reference data (text file)
    txt_path = 'debug_samples/chemical_solubility_study.txt'
    if not os.path.exists(txt_path):
        print(f"Text file not found: {txt_path}")
        return
    
    with open(txt_path, 'rb') as f:
        content = f.read()
    
    result = read_document_content('chemical_solubility_study.txt', content)
    print(f"Reference data extracted: {len(result.get('text', ''))} chars")
    
    # Prepare state for LLM analyst
    state = {
        "df": df,
        "filename": os.path.basename(csv_path),
        "document_text": result.get("text", ""),
        "document_metadata": result.get("metadata", {}),
        "document_filename": result.get("document_filename", ""),
    }
    
    # Get LLM analyst and run analysis with debug
    analyst = get_data_analyst()
    
    # Let's manually call the LLM to see raw response
    print("\n=== Manual LLM Call ===")
    system_prompt = analyst._build_system_prompt()
    print(f"System prompt length: {len(system_prompt)} chars")
    
    # Build user content like in analyze method
    document_text = state.get("document_text", "")
    document_meta = state.get("document_metadata", {})
    
    parts = ["以下のデータについて分析プランを提案してください。"]
    if document_text:
        parts.append(f"\n\n## ドキュメント内容\n{document_text[:2000]}")  # 最初の2000文字
        if document_meta:
            parts.append(f"\n\nドキュメントメタデータ: {document_meta}")
    
    # Add data summary
    current_summary = analyst.build_data_summary(state)
    parts.append(f"\n\n{current_summary}")
    user_content = "\n".join(parts)
    
    print(f"User content length: {len(user_content)} chars")
    print(f"User content preview: {user_content[:500]}...")
    
    # Call LLM
    try:
        raw_reply = analyst._call_llm(system_prompt, [{"role": "user", "content": user_content}])
        print(f"\n=== RAW LLM RESPONSE ===")
        print(f"Length: {len(raw_reply)} chars")
        print(f"First 500 chars: {raw_reply[:500]}")
        print(f"Last 500 chars: {raw_reply[-500:] if len(raw_reply) > 500 else raw_reply}")
        print(f"Full response:\n{raw_reply}")
        
        # Try to parse suggestions
        suggestions = analyst._parse_suggestions(raw_reply)
        print(f"\n=== PARSED SUGGESTIONS ===")
        print(suggestions)
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
