#!/usr/bin/env python3
"""
サンプルデータ生成スクリプト
- SMILES 系データ：RDKit 記述子から擬似目的変数を生成
- テーブルデータ：scikit-learn の make_regression + 人工的ノイズ・欠損値追加
- 混合物データ：化合物 3 列（SMILES）、回帰、重量 WT％での分率
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from sklearn.datasets import make_regression, make_classification
import os

# ========== SMILES データ生成 ==========
SMILES_POOL = [
    # 医薬品
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Naproxen
    # 溶媒・試薬
    "CCO", "CC(C)O", "CC(C)(C)O", "c1ccccc1", "CC1=CC=CC=C1",
    # 天然物・複雑構造
    "CC1=CC2=C(C=C1C=O)C(=O)C3=C(O2)C=CC(=C3)O",  # Simple flavonoid
    "CCC1=CC(=C(C=C1)O)C=O",  # Vanillin
    # 極端ケース
    "C"*50,  # 長鎖アルカン（エラーテスト）
    "C1CC1C1CC1C1CC1",  # 複雑な環構造
]

def calc_pseudo_targets(smiles: str) -> dict:
    """SMILES から擬似目的変数を計算（物性値の相関を模擬）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"logS": np.nan, "pIC50": np.nan, "class": -1}
    
    # RDKit 記述子
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    
    # 擬似 logS（水溶性）: MW↑・logP↑・TPSA↑ → 溶解度↓
    logS = -0.01*mw - 0.3*logp - 0.005*tpsa + np.random.normal(0, 0.3)
    
    # 擬似 pIC50（活性）: 複雑な構造ほど高い活性を模擬
    pIC50 = 3 + 0.005*mw + 0.2*logp - 0.1*hbd + np.random.normal(0, 0.5)
    
    # 分類ラベル（logP 閾値ベース）
    cls = 1 if logp > 2.5 else 0
    
    return {
        "logS": round(logS, 3),
        "pIC50": round(pIC50, 3),
        "class": cls,
        "MW": round(mw, 2),
        "LogP": round(logp, 2)
    }

def generate_smiles_samples(n: int, filename: str):
    data = []
    for i in range(n):
        smiles = SMILES_POOL[i % len(SMILES_POOL)]
        targets = calc_pseudo_targets(smiles)
        data.append({
            "SMILES": smiles,
            "Compound_Name": f"Compound_{i+1:03d}",
            "logS": targets["logS"],
            "pIC50": targets["pIC50"],
            "Class": targets["class"],
            "Source": "Synthetic",
            "Notes": "Generated for testing"
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 混合物データ生成（数値データなし）==========
def generate_mixture_samples(n: int, filename: str):
    """
    混合物データ生成：化合物 3 列（SMILES）、回帰目的変数、重量％分率（WT%）
    - 3 成分の混合物を想定
    - 各成分の重量％は合計 100% になるように正規化
    - 目的変数は各成分の寄与と相互作用から擬似生成
    """
    # 使用する化合物プール（異なる特性を持つものを選択）
    compound_pool = [
        ("CCO", "Ethanol"),                    # 極性溶媒
        ("CC(C)O", "Isopropanol"),             # 極性溶媒
        ("c1ccccc1", "Benzene"),               # 非極性溶媒
        ("CC1=CC=CC=C1", "Toluene"),           # 非極性溶媒
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),  # 医薬品
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),  # 天然物
        ("CCC1=CC(=C(C=C1)O)C=O", "Vanillin"), # 香料
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),  # 医薬品
        ("CCOC", "Ethyl acetate"),              # エステル溶媒
        ("CCCO", "1-Propanol"),                # アルコール
        ("c1ccc(O)cc1", "Phenol"),            # フェノール
    ]

    data = []
    for i in range(n):
        # 3 成分をランダム選択（重複あり）
        indices = np.random.choice(len(compound_pool), size=3, replace=True)

        # 重量％の生成（合計 100% に正規化）
        raw_weights = np.random.uniform(10, 90, size=3)
        wt_percent = (raw_weights / raw_weights.sum() * 100).round(2)

        # 化合物情報
        smiles_list = [compound_pool[idx][0] for idx in indices]
        names = [compound_pool[idx][1] for idx in indices]

        # 擬似目的変数の生成（各成分の特性と相互作用を考慮）
        mol_props = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                mol_props.append((mw, logp, tpsa))
            else:
                mol_props.append((0, 0, 0))

        # 重量％を係数として使用（0-1 の範囲）
        frac = wt_percent / 100.0

        # 擬似物性値：各成分の寄与の加重平均 + 相互作用項 + ノイズ
        base_property = sum(frac[j] * (mol_props[j][1] * 0.5 + mol_props[j][2] * 0.01) for j in range(3))
        interaction = frac[0] * frac[1] * 5 + frac[1] * frac[2] * 3  # 二元相互作用
        target = base_property + interaction + np.random.normal(0, 2)

        data.append({
            "Compound_1_SMILES": smiles_list[0],
            "Compound_1_Name": names[0],
            "Compound_1_WT%": wt_percent[0],
            "Compound_2_SMILES": smiles_list[1],
            "Compound_2_Name": names[1],
            "Compound_2_WT%": wt_percent[1],
            "Compound_3_SMILES": smiles_list[2],
            "Compound_3_Name": names[2],
            "Compound_3_WT%": wt_percent[2],
            "Target_Property": round(target, 3),
            "Sample_ID": f"MIX{i+1:04d}",
            "Total_WT%": wt_percent.sum(),  # 検証用（常に 100%）
            "Notes": "Synthetic mixture data"
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")


# ========== 混合物データ生成（数値データあり：温度、湿度など）==========
def generate_mixture_with_numerics_samples(n: int, filename: str):
    """
    混合物データ生成：化合物 3 列（SMILES）、回帰目的変数、重量％分率（WT%）＋数値データ（温度、湿度など）
    - 3 成分の混合物を想定
    - 各成分の重量％は合計 100% になるように正規化
    - 数値条件：温度（℃）、湿度（%）、圧力（MPa）、攪拌速度（rpm）
    - 目的変数は各成分の寄与・相互作用・数値条件から擬似生成
    """
    # 使用する化合物プール（異なる特性を持つものを選択）
    compound_pool = [
        ("CCO", "Ethanol"),                    # 極性溶媒
        ("CC(C)O", "Isopropanol"),             # 極性溶媒
        ("c1ccccc1", "Benzene"),               # 非極性溶媒
        ("CC1=CC=CC=C1", "Toluene"),           # 非極性溶媒
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),  # 医薬品
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),  # 天然物
        ("CCC1=CC(=C(C=C1)O)C=O", "Vanillin"), # 香料
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),  # 医薬品
        ("CCOC", "Ethyl acetate"),              # エステル溶媒
        ("CCCO", "1-Propanol"),                # アルコール
        ("c1ccc(O)cc1", "Phenol"),            # フェノール
    ]
    
    data = []
    for i in range(n):
        # 3 成分をランダム選択（重複あり）
        indices = np.random.choice(len(compound_pool), size=3, replace=True)
        
        # 重量％の生成（合計 100% に正規化）
        raw_weights = np.random.uniform(10, 90, size=3)
        wt_percent = (raw_weights / raw_weights.sum() * 100).round(2)
        
        # 化合物情報
        smiles_list = [compound_pool[idx][0] for idx in indices]
        names = [compound_pool[idx][1] for idx in indices]
        
        # 数値条件の生成（実験条件を模擬）
        temperature = np.random.uniform(15, 85)      # 温度（℃）
        humidity = np.random.uniform(20, 80)         # 湿度（%）
        pressure = np.random.uniform(0.1, 2.0)       # 圧力（MPa）
        stirring_speed = np.random.uniform(100, 1000)  # 攪拌速度（rpm）
        
        # 擬似目的変数の生成（各成分の特性と相互作用・数値条件を考慮）
        mol_props = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                mol_props.append((mw, logp, tpsa))
            else:
                mol_props.append((0, 0, 0))
        
        # 重量％を係数として使用（0-1 の範囲）
        frac = wt_percent / 100.0
        
        # 擬似物性値：各成分の寄与の加重平均 + 相互作用項 + 数値条件の影響 + ノイズ
        base_property = sum(frac[j] * (mol_props[j][1] * 0.5 + mol_props[j][2] * 0.01) for j in range(3))
        interaction = frac[0] * frac[1] * 5 + frac[1] * frac[2] * 3  # 二元相互作用
        
        # 数値条件の影響（温度・湿度・圧力・攪拌）
        temp_effect = (temperature - 50) * 0.05  # 温度の影響
        humidity_effect = (humidity - 50) * 0.02  # 湿度の影響
        pressure_effect = pressure * 1.5  # 圧力の影響
        stirring_effect = np.log(stirring_speed / 100) * 2  # 攪拌速度の影響（対数スケール）
        
        target = base_property + interaction + temp_effect + humidity_effect + pressure_effect + stirring_effect + np.random.normal(0, 1.5)
        
        data.append({
            "Compound_1_SMILES": smiles_list[0],
            "Compound_1_Name": names[0],
            "Compound_1_WT%": wt_percent[0],
            "Compound_2_SMILES": smiles_list[1],
            "Compound_2_Name": names[1],
            "Compound_2_WT%": wt_percent[1],
            "Compound_3_SMILES": smiles_list[2],
            "Compound_3_Name": names[2],
            "Compound_3_WT%": wt_percent[2],
            "Temperature_C": round(temperature, 2),
            "Humidity_%": round(humidity, 2),
            "Pressure_MPa": round(pressure, 3),
            "Stirring_rpm": round(stirring_speed, 1),
            "Target_Property": round(target, 3),
            "Sample_ID": f"MIX_NUM{i+1:04d}",
            "Total_WT%": wt_percent.sum(),  # 検証用（常に 100%）
            "Notes": "Synthetic mixture data with numeric conditions"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 通常テーブルデータ生成 ==========
def generate_tabular_samples(n: int, filename: str, task: str = "regression"):
    if task == "regression":
        X, y = make_regression(
            n_samples=n, n_features=8, n_informative=5,
            noise=10, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]
        
        # カテゴリカル特徴量の追加
        df["Category"] = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n)
        
        # 欠損値の人工的追加（5%）
        mask = np.random.random((n, 3)) < 0.05
        for col_idx in range(3):
            df.iloc[mask[:, col_idx], col_idx] = np.nan
            
    else:  # classification
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=7,
            n_redundant=2, n_classes=3, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target_Class"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]
    
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 実行 ==========
if __name__ == "__main__":
    os.makedirs("data/samples", exist_ok=True)
    
    # SMILES データ
    generate_smiles_samples(25, "data/samples/smiles_25_quick.csv")
    generate_smiles_samples(100, "data/samples/smiles_100_ml.csv")
    generate_smiles_samples(500, "data/samples/smiles_500_stress.csv")
    
    # 通常テーブルデータ
    generate_tabular_samples(50, "data/samples/tabular_50_simple.csv", "regression")
    generate_tabular_samples(200, "data/samples/tabular_200_complex.csv", "regression")
    generate_tabular_samples(1000, "data/samples/tabular_1000_large.csv", "regression")
    
    # 混合物データ（数値条件なし）
    generate_mixture_samples(30, "data/samples/mixture_30_simple.csv")
    generate_mixture_samples(100, "data/samples/mixture_100_ml.csv")
    
    # 混合物データ（数値条件あり：温度、湿度など）
    generate_mixture_with_numerics_samples(30, "data/samples/mixture_with_numerics_30_simple.csv")
    generate_mixture_with_numerics_samples(100, "data/samples/mixture_with_numerics_100_ml.csv")
    
    print("\n🎉 All sample data generated successfully!")
