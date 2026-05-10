import sys
import os
sys.path.append('.')

import pandas as pd
from backend.llm.data_analyst import get_data_analyst, read_document_content

def main():
    print("Testing LLM analysis with reference data...")
    
    # Load main data (CSV)
    csv_path = 'debug_samples/solubility_data.csv'
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded main data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    
    # Load reference data (text file)
    txt_path = 'debug_samples/chemical_solubility_study.txt'
    if not os.path.exists(txt_path):
        print(f"Text file not found: {txt_path}")
        return
    
    with open(txt_path, 'rb') as f:
        content = f.read()
    
    result = read_document_content('chemical_solubility_study.txt', content)
    print(f"\nReference data extracted:")
    print(f"  Text length: {len(result.get('text', ''))}")
    print(f"  Tables count: {len(result.get('tables', []))}")
    if result.get('text'):
        print(f"  Text preview: {result['text'][:200]}...")
    
    # Prepare state for LLM analyst
    state = {
        "df": df,
        "filename": os.path.basename(csv_path),
        "document_text": result.get("text", ""),
        "document_metadata": result.get("metadata", {}),
        "document_filename": result.get("document_filename", ""),
    }
    
    # Get LLM analyst and run analysis
    analyst = get_data_analyst()
    print("\nRunning LLM analysis...")
    analysis_result = analyst.analyze(state, reset=True)
    
    print("\nLLM Reply:")
    print(analysis_result.get("reply", ""))
    
    print("\nSuggestions:")
    import json
    print(json.dumps(analysis_result.get("suggestions", {}), indent=2, ensure_ascii=False))
    
    # Apply suggestions to state
    applied = analyst.apply_suggestions(state, analysis_result.get("suggestions", {}))
    print(f"\nApplied suggestions: {applied}")
    
    print("\nUpdated state keys:")
    for key in ["target_col", "task_type", "selected_descriptors", "monotonic_constraints"]:
        if key in state:
            print(f"  {key}: {state[key]}")

if __name__ == "__main__":
    main()
