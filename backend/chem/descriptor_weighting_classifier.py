"""
backend/chem/descriptor_weighting_classifier.py

ChemAI2の全記述子に対する加重方法（重量分率/モル分率/文脈依存）の
明示的分類マッピング。

全RDKit 217記述子 + xTB基本10記述子 + xTB ML派生8記述子 +
アンサンブル特徴量 + 信頼度スコア + Gasteiger電荷 を
**1件ずつ**物理化学的根拠に基づいて手動設定。

さらに正規表現フォールバックで未知記述子にも対応。

既存モジュールへの影響: なし（完全新規）
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

WeightingType = Literal["weight", "mole", "context"]

# ============================================================
# 全記述子の明示的マッピング（1件ずつ手動設定）
# ============================================================

# weight = 重量分率で加重平均が物理的に適切
# mole   = モル分率で加重平均が物理的に適切
# context = 文脈依存（線形加重が不適切な可能性）

EXPLICIT_DESCRIPTOR_MAP: dict[str, tuple[WeightingType, str]] = {
    # ────────────────────────────────────────────────────────
    # RDKit 217記述子 — 完全リスト
    # ────────────────────────────────────────────────────────

    # --- E-State / 電子状態インデックス (電子状態→モル比) ---
    "MaxAbsEStateIndex":  ("mole", "電子状態インデックス: 分子内電子分布"),
    "MaxEStateIndex":     ("mole", "電子状態インデックス: 分子内電子分布"),
    "MinAbsEStateIndex":  ("mole", "電子状態インデックス: 分子内電子分布"),
    "MinEStateIndex":     ("mole", "電子状態インデックス: 分子内電子分布"),

    # --- QED / SPS ---
    "qed":  ("weight", "薬物らしさスコア: 組成比に比例"),
    "SPS":  ("weight", "合成容易性スコア: 構造組成比"),

    # --- 質量関連 (→重量比) ---
    "MolWt":               ("weight", "分子量: 質量保存則"),
    "HeavyAtomMolWt":      ("weight", "重原子分子量: 質量保存則"),
    "ExactMolWt":          ("weight", "精密質量: 質量保存則"),

    # --- 電子数 (→モル比) ---
    "NumValenceElectrons": ("mole", "価電子数: 分子あたり電子数"),
    "NumRadicalElectrons": ("mole", "ラジカル電子数: 分子あたり電子数"),

    # --- 部分電荷 (→モル比) ---
    "MaxPartialCharge":    ("mole", "Gasteiger最大部分電荷: 分子内電子分布"),
    "MinPartialCharge":    ("mole", "Gasteiger最小部分電荷: 分子内電子分布"),
    "MaxAbsPartialCharge": ("mole", "Gasteiger最大絶対部分電荷"),
    "MinAbsPartialCharge": ("mole", "Gasteiger最小絶対部分電荷"),

    # --- FP密度 (→context: ビット系) ---
    "FpDensityMorgan1": ("context", "Morgan FP密度: ビットベクトル系"),
    "FpDensityMorgan2": ("context", "Morgan FP密度: ビットベクトル系"),
    "FpDensityMorgan3": ("context", "Morgan FP密度: ビットベクトル系"),

    # --- BCUT2D (→重量比: 巨視的物性の代理変数) ---
    "BCUT2D_MWHI":   ("weight", "BCUT2D分子量上限: 質量ベース"),
    "BCUT2D_MWLOW":  ("weight", "BCUT2D分子量下限: 質量ベース"),
    "BCUT2D_CHGHI":  ("mole", "BCUT2D電荷上限: 電子状態"),
    "BCUT2D_CHGLO":  ("mole", "BCUT2D電荷下限: 電子状態"),
    "BCUT2D_LOGPHI": ("weight", "BCUT2DLogP上限: 分配係数ベース"),
    "BCUT2D_LOGPLOW":("weight", "BCUT2DLogP下限: 分配係数ベース"),
    "BCUT2D_MRHI":   ("weight", "BCUT2D屈折率上限: 巨視的光学物性"),
    "BCUT2D_MRLOW":  ("weight", "BCUT2D屈折率下限: 巨視的光学物性"),

    # --- トポロジカルインデックス (→重量比: 構造組成) ---
    "BalabanJ":    ("weight", "Balaban J: トポロジカル構造指標"),
    "BertzCT":     ("weight", "Bertz複雑性: 構造複雑性指標"),
    "AvgIpc":      ("weight", "平均情報量: トポロジカル"),
    "Ipc":         ("weight", "情報量: トポロジカル"),
    "HallKierAlpha":("weight", "Hall-Kier α: 分子形状"),
    "Kappa1":      ("weight", "Kappa1: 分子形状"),
    "Kappa2":      ("weight", "Kappa2: 分子形状"),
    "Kappa3":      ("weight", "Kappa3: 分子形状"),
    "Phi":         ("weight", "Phi: 柔軟性指標"),

    # --- 接続性インデックス Chi (→重量比: 構造トポロジー) ---
    "Chi0":  ("weight", "Chi0: 接続性インデックス"),
    "Chi0n": ("weight", "Chi0n: 接続性インデックス"),
    "Chi0v": ("weight", "Chi0v: 接続性インデックス"),
    "Chi1":  ("weight", "Chi1: 接続性インデックス"),
    "Chi1n": ("weight", "Chi1n: 接続性インデックス"),
    "Chi1v": ("weight", "Chi1v: 接続性インデックス"),
    "Chi2n": ("weight", "Chi2n: 接続性インデックス"),
    "Chi2v": ("weight", "Chi2v: 接続性インデックス"),
    "Chi3n": ("weight", "Chi3n: 接続性インデックス"),
    "Chi3v": ("weight", "Chi3v: 接続性インデックス"),
    "Chi4n": ("weight", "Chi4n: 接続性インデックス"),
    "Chi4v": ("weight", "Chi4v: 接続性インデックス"),

    # --- 表面積 VSA系 (→重量比: 面積は質量スケール加算的) ---
    "LabuteASA":   ("weight", "Labute ASA: 表面積"),
    "TPSA":        ("weight", "極性表面積: 質量スケール加算"),
    "EState_VSA1": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA2": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA3": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA4": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA5": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA6": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA7": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA8": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA9": ("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA10":("weight", "EState VSA: 面積×電子状態"),
    "EState_VSA11":("weight", "EState VSA: 面積×電子状態"),
    "VSA_EState1": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState2": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState3": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState4": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState5": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState6": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState7": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState8": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState9": ("weight", "VSA EState: 面積×電子状態"),
    "VSA_EState10":("weight", "VSA EState: 面積×電子状態"),
    "PEOE_VSA1":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA2":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA3":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA4":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA5":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA6":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA7":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA8":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA9":   ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA10":  ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA11":  ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA12":  ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA13":  ("weight", "PEOE VSA: 面積×部分電荷"),
    "PEOE_VSA14":  ("weight", "PEOE VSA: 面積×部分電荷"),
    "SMR_VSA1":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA2":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA3":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA4":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA5":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA6":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA7":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA8":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA9":    ("weight", "SMR VSA: 面積×屈折率"),
    "SMR_VSA10":   ("weight", "SMR VSA: 面積×屈折率"),
    "SlogP_VSA1":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA2":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA3":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA4":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA5":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA6":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA7":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA8":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA9":  ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA10": ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA11": ("weight", "SlogP VSA: 面積×LogP"),
    "SlogP_VSA12": ("weight", "SlogP VSA: 面積×LogP"),

    # --- 物性 LogP / MR (→重量比: 分配・屈折率) ---
    "MolLogP": ("weight", "LogP: 分配係数は質量濃度定義"),
    "MolMR":   ("weight", "分子屈折率: 巨視的光学物性"),

    # --- 原子カウント・環構造 (→重量比: 構造組成) ---
    "HeavyAtomCount":             ("weight", "重原子数: 構造組成"),
    "NHOHCount":                  ("weight", "NH/OH数: 官能基組成"),
    "NOCount":                    ("weight", "NO数: 官能基組成"),
    "NumAliphaticCarbocycles":    ("weight", "脂肪族炭素環数: 構造組成"),
    "NumAliphaticHeterocycles":   ("weight", "脂肪族複素環数: 構造組成"),
    "NumAliphaticRings":          ("weight", "脂肪族環数: 構造組成"),
    "NumAmideBonds":              ("weight", "アミド結合数: 構造組成"),
    "NumAromaticCarbocycles":     ("weight", "芳香族炭素環数: 構造組成"),
    "NumAromaticHeterocycles":    ("weight", "芳香族複素環数: 構造組成"),
    "NumAromaticRings":           ("weight", "芳香族環数: 構造組成"),
    "NumAtomStereoCenters":       ("context", "不斉中心数: 光学非加算性"),
    "NumBridgeheadAtoms":         ("weight", "橋頭原子数: 構造組成"),
    "NumHAcceptors":              ("weight", "水素結合受容体数: 官能基組成"),
    "NumHDonors":                 ("weight", "水素結合供与体数: 官能基組成"),
    "NumHeteroatoms":             ("weight", "ヘテロ原子数: 構造組成"),
    "NumHeterocycles":            ("weight", "複素環数: 構造組成"),
    "NumRotatableBonds":          ("weight", "回転可能結合数: 構造組成"),
    "NumSaturatedCarbocycles":    ("weight", "飽和炭素環数: 構造組成"),
    "NumSaturatedHeterocycles":   ("weight", "飽和複素環数: 構造組成"),
    "NumSaturatedRings":          ("weight", "飽和環数: 構造組成"),
    "NumSpiroAtoms":              ("weight", "スピロ原子数: 構造組成"),
    "NumUnspecifiedAtomStereoCenters": ("context", "未指定不斉中心: 光学非加算性"),
    "RingCount":                  ("weight", "環数: 構造組成"),
    "FractionCSP3":               ("weight", "SP3炭素割合: 構造組成比"),

    # --- 官能基フラグメントカウント fr_* (→重量比: 構造組成) ---
    "fr_Al_COO":               ("weight", "脂肪族カルボン酸数"),
    "fr_Al_OH":                ("weight", "脂肪族OH数"),
    "fr_Al_OH_noTert":         ("weight", "脂肪族OH(3級除く)数"),
    "fr_ArN":                  ("weight", "芳香族N数"),
    "fr_Ar_COO":               ("weight", "芳香族カルボン酸数"),
    "fr_Ar_N":                 ("weight", "芳香族N数"),
    "fr_Ar_NH":                ("weight", "芳香族NH数"),
    "fr_Ar_OH":                ("weight", "芳香族OH数"),
    "fr_COO":                  ("weight", "カルボン酸数"),
    "fr_COO2":                 ("weight", "カルボン酸エステル数"),
    "fr_C_O":                  ("weight", "C=O数"),
    "fr_C_O_noCOO":            ("weight", "C=O(COO除く)数"),
    "fr_C_S":                  ("weight", "C=S数"),
    "fr_HOCCN":                ("weight", "HOCCN数"),
    "fr_Imine":                ("weight", "イミン数"),
    "fr_NH0":                  ("weight", "3級アミン数"),
    "fr_NH1":                  ("weight", "2級アミン数"),
    "fr_NH2":                  ("weight", "1級アミン数"),
    "fr_N_O":                  ("weight", "N-O結合数"),
    "fr_Ndealkylation1":       ("weight", "N-脱アルキル化1"),
    "fr_Ndealkylation2":       ("weight", "N-脱アルキル化2"),
    "fr_Nhpyrrole":            ("weight", "N-Hピロール数"),
    "fr_SH":                   ("weight", "チオール数"),
    "fr_aldehyde":             ("weight", "アルデヒド数"),
    "fr_alkyl_carbamate":      ("weight", "アルキルカルバメート数"),
    "fr_alkyl_halide":         ("weight", "アルキルハライド数"),
    "fr_allylic_oxid":         ("weight", "アリル酸化数"),
    "fr_amide":                ("weight", "アミド数"),
    "fr_amidine":              ("weight", "アミジン数"),
    "fr_aniline":              ("weight", "アニリン数"),
    "fr_aryl_methyl":          ("weight", "アリールメチル数"),
    "fr_azide":                ("weight", "アジド数"),
    "fr_azo":                  ("weight", "アゾ数"),
    "fr_barbitur":             ("weight", "バルビツール酸数"),
    "fr_benzene":              ("weight", "ベンゼン環数"),
    "fr_benzodiazepine":       ("weight", "ベンゾジアゼピン数"),
    "fr_bicyclic":             ("weight", "二環式構造数"),
    "fr_diazo":                ("weight", "ジアゾ数"),
    "fr_dihydropyridine":      ("weight", "ジヒドロピリジン数"),
    "fr_epoxide":              ("weight", "エポキシド数"),
    "fr_ester":                ("weight", "エステル数"),
    "fr_ether":                ("weight", "エーテル数"),
    "fr_furan":                ("weight", "フラン数"),
    "fr_guanido":              ("weight", "グアニジン数"),
    "fr_halogen":              ("weight", "ハロゲン数"),
    "fr_hdrzine":              ("weight", "ヒドラジン数"),
    "fr_hdrzone":              ("weight", "ヒドラゾン数"),
    "fr_imidazole":            ("weight", "イミダゾール数"),
    "fr_imide":                ("weight", "イミド数"),
    "fr_isocyan":              ("weight", "イソシアネート数"),
    "fr_isothiocyan":          ("weight", "イソチオシアネート数"),
    "fr_ketone":               ("weight", "ケトン数"),
    "fr_ketone_Topliss":       ("weight", "ケトン(Topliss)数"),
    "fr_lactam":               ("weight", "ラクタム数"),
    "fr_lactone":              ("weight", "ラクトン数"),
    "fr_methoxy":              ("weight", "メトキシ数"),
    "fr_morpholine":           ("weight", "モルホリン数"),
    "fr_nitrile":              ("weight", "ニトリル数"),
    "fr_nitro":                ("weight", "ニトロ数"),
    "fr_nitro_arom":           ("weight", "芳香族ニトロ数"),
    "fr_nitro_arom_nonortho":  ("weight", "芳香族ニトロ(非オルト)数"),
    "fr_nitroso":              ("weight", "ニトロソ数"),
    "fr_oxazole":              ("weight", "オキサゾール数"),
    "fr_oxime":                ("weight", "オキシム数"),
    "fr_para_hydroxylation":   ("weight", "パラ位水酸化数"),
    "fr_phenol":               ("weight", "フェノール数"),
    "fr_phenol_noOrthoHbond":  ("weight", "フェノール(オルト水素結合なし)数"),
    "fr_phos_acid":            ("weight", "リン酸数"),
    "fr_phos_ester":           ("weight", "リン酸エステル数"),
    "fr_piperdine":            ("weight", "ピペリジン数"),
    "fr_piperzine":            ("weight", "ピペラジン数"),
    "fr_priamide":             ("weight", "1級アミド数"),
    "fr_prisulfonamd":         ("weight", "1級スルホンアミド数"),
    "fr_pyridine":             ("weight", "ピリジン数"),
    "fr_quatN":                ("weight", "4級窒素数"),
    "fr_sulfide":              ("weight", "スルフィド数"),
    "fr_sulfonamd":            ("weight", "スルホンアミド数"),
    "fr_sulfone":              ("weight", "スルホン数"),
    "fr_term_acetylene":       ("weight", "末端アセチレン数"),
    "fr_tetrazole":            ("weight", "テトラゾール数"),
    "fr_thiazole":             ("weight", "チアゾール数"),
    "fr_thiocyan":             ("weight", "チオシアン酸数"),
    "fr_thiophene":            ("weight", "チオフェン数"),
    "fr_unbrch_alkane":        ("weight", "直鎖アルカン数"),
    "fr_urea":                 ("weight", "ウレア数"),

    # ────────────────────────────────────────────────────────
    # xTB 基本記述子 (10件)
    # ────────────────────────────────────────────────────────
    "xtb_TotalEnergy":       ("mole", "全エネルギー: 1分子あたり電子エネルギー"),
    "xtb_HomoEnergy":        ("mole", "HOMO: 1分子軌道エネルギー"),
    "xtb_LumoEnergy":        ("mole", "LUMO: 1分子軌道エネルギー"),
    "xtb_HomoLumoGap":       ("mole", "HLG: 1分子軌道エネルギー差"),
    "xtb_DipoleMoment":      ("mole", "双極子モーメント: 分子分極"),
    "xtb_Polarizability":    ("mole", "分極率: 分子応答"),
    "xtb_MullikenChargeMax": ("mole", "Mulliken電荷最大: 分子内電子分布"),
    "xtb_MullikenChargeMin": ("mole", "Mulliken電荷最小: 分子内電子分布"),
    "xtb_MullikenChargeMean":("mole", "Mulliken電荷平均: 分子内電子分布"),
    "xtb_MullikenChargeStd": ("mole", "Mulliken電荷標準偏差: 分子内電子分布"),

    # ────────────────────────────────────────────────────────
    # xTB ML派生特徴量 (8件)
    # ────────────────────────────────────────────────────────
    "xtb_ml_TotalEnergy":       ("mole", "派生: 全エネルギー"),
    "xtb_ml_HomoEnergy":        ("mole", "派生: HOMO"),
    "xtb_ml_LumoEnergy":        ("mole", "派生: LUMO"),
    "xtb_ml_Gap":               ("mole", "派生: HLG"),
    "xtb_ml_Dipole":            ("mole", "派生: 双極子"),
    "xtb_ml_Hardness":          ("mole", "化学的硬さ: (IP-EA)/2"),
    "xtb_ml_Softness":          ("mole", "化学的軟らかさ: 1/硬さ"),
    "xtb_ml_Electrophilicity":  ("mole", "親電子性指数: μ²/2η"),

    # ────────────────────────────────────────────────────────
    # 信頼度スコア (6件)
    # ────────────────────────────────────────────────────────
    "conf_convergence":            ("context", "収束性スコア: 計算品質指標"),
    "conf_electronic_stability":   ("context", "電子安定性スコア: 計算品質指標"),
    "conf_charge_consistency":     ("context", "電荷一貫性スコア: 計算品質指標"),
    "conf_descriptor_completeness":("context", "記述子完全性: 計算品質指標"),
    "conf_conformer_repr":         ("context", "配座代表性: 計算品質指標"),
    "conf_overall":                ("context", "総合信頼度: 計算品質指標"),

    # ────────────────────────────────────────────────────────
    # アンサンブル特徴量 (ens_*)
    # ────────────────────────────────────────────────────────
    "ens_n_conformers":       ("context", "conformer数: 統計量"),
    "ens_boltzmann_entropy":  ("mole", "Boltzmannエントロピー: 統計力学"),
    "ens_energy_mean":        ("mole", "エネルギー平均: 分子エネルギー"),
    "ens_energy_std":         ("context", "エネルギー標準偏差: 柔軟性指標"),
    "ens_energy_range":       ("context", "エネルギー範囲: 柔軟性指標"),
    "ens_energy_min":         ("mole", "最低エネルギー: 安定構造"),
    "ens_energy_boltz_mean":  ("mole", "Boltzmann加重エネルギー"),
    "ens_energy_range_kcal":  ("context", "エネルギー範囲(kcal): 柔軟性"),
    "ens_homo_mean":          ("mole", "HOMO平均: 分子軌道"),
    "ens_homo_std":           ("context", "HOMO標準偏差: 構造依存性"),
    "ens_homo_range":         ("context", "HOMO範囲: 構造依存性"),
    "ens_homo_boltz_mean":    ("mole", "HOMO Boltzmann加重"),
    "ens_lumo_mean":          ("mole", "LUMO平均: 分子軌道"),
    "ens_lumo_std":           ("context", "LUMO標準偏差: 構造依存性"),
    "ens_lumo_range":         ("context", "LUMO範囲: 構造依存性"),
    "ens_lumo_boltz_mean":    ("mole", "LUMO Boltzmann加重"),
    "ens_gap_mean":           ("mole", "HLG平均"),
    "ens_gap_std":            ("context", "HLG標準偏差: 構造依存性"),
    "ens_gap_range":          ("context", "HLG範囲: 構造依存性"),
    "ens_gap_boltz_mean":     ("mole", "HLG Boltzmann加重"),
    "ens_qmax_mean":          ("mole", "電荷最大平均"),
    "ens_qmax_std":           ("context", "電荷最大標準偏差"),
    "ens_qmin_mean":          ("mole", "電荷最小平均"),
    "ens_qmin_std":           ("context", "電荷最小標準偏差"),
    "ens_qstd_mean":          ("mole", "電荷STD平均"),
    "ens_qstd_std":           ("context", "電荷STD標準偏差"),
    "ens_dipole_mean":        ("mole", "双極子平均"),
    "ens_dipole_std":         ("context", "双極子標準偏差: 構造依存性"),
    "ens_dipole_cv":          ("context", "双極子変動係数"),
    "ens_polar_mean":         ("mole", "分極率平均"),
    "ens_polar_std":          ("context", "分極率標準偏差"),
    "ens_polar_cv":           ("context", "分極率変動係数"),

    # ────────────────────────────────────────────────────────
    # 3D幾何特徴量 (xtb_ml_features.py から)
    # ────────────────────────────────────────────────────────
    "xtb_ml_3D_MaxDistance":   ("context", "最大原子間距離: 3D構造依存"),
    "xtb_ml_3D_Asphericity":  ("context", "非球面性: 3D構造依存"),
    "xtb_ml_3D_Eccentricity": ("context", "離心率: 3D構造依存"),
    "xtb_ml_3D_InertiaX":    ("context", "慣性モーメントX: 3D構造依存"),
    "xtb_ml_3D_InertiaY":    ("context", "慣性モーメントY: 3D構造依存"),
    "xtb_ml_3D_InertiaZ":    ("context", "慣性モーメントZ: 3D構造依存"),
    "xtb_ml_3D_RadiusOfGyration": ("context", "回転半径: 3D構造依存"),

    # ────────────────────────────────────────────────────────
    # Gasteiger電荷 (RDKit adapter から追加される場合)
    # ────────────────────────────────────────────────────────
    "GasteigerChargeMax":  ("mole", "Gasteiger最大電荷"),
    "GasteigerChargeMin":  ("mole", "Gasteiger最小電荷"),
    "GasteigerChargeMean": ("mole", "Gasteiger平均電荷"),
    "GasteigerChargeStd":  ("mole", "Gasteiger電荷標準偏差"),
}

# 4000件の手動マッピング（JSON）の動的ロード
import json
import os
import logging

_logger = logging.getLogger(__name__)

def _load_4000_mappings():
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chemai2_4000_manual_weighting.json")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                EXPLICIT_DESCRIPTOR_MAP[k] = (v.get("weighting", "context"), v.get("rationale", "Loaded from JSON"))
    except Exception as e:
        _logger.warning(f"Failed to load user descriptor mappings from {json_path}: {e}")

_load_4000_mappings()

# ============================================================
# 正規表現フォールバック（未知記述子用）
# ============================================================

_FALLBACK_RULES: list[tuple[str, WeightingType, str]] = [
    # 質量・体積・密度
    (r"(?i).*(MolWt|Weight|Mass|Volume|Density).*", "weight", "質量関連"),
    # 表面積
    (r"(?i).*(TPSA|VSA|ASA|Surface|PSA).*", "weight", "表面積関連"),
    # LogP / 溶解度
    (r"(?i).*(LogP|LogS|Solub|Partition).*", "weight", "分配係数関連"),
    # 官能基カウント
    (r"(?i)^fr_.*", "weight", "官能基フラグメント"),
    # 原子カウント
    (r"(?i).*(Count|Num[A-Z]).*", "weight", "原子/構造カウント"),
    # 接続性 / トポロジカル
    (r"(?i).*(Chi\d|Kappa|Balaban|Bertz|Ipc).*", "weight", "トポロジカル"),
    # 軌道エネルギー
    (r"(?i).*(HOMO|LUMO|Gap|Orbital|Eigenval).*", "mole", "軌道エネルギー"),
    # 電荷
    (r"(?i).*(Charge|Mulliken|Gasteiger|ESP|CM5).*", "mole", "電荷"),
    # 双極子 / 分極率
    (r"(?i).*(Dipole|Polar|Quadrupole).*", "mole", "分極"),
    # 反応性
    (r"(?i).*(Fukui|Softness|Hardness|Electrophil|Nucleophil).*", "mole", "反応性"),
    # 熱力学
    (r"(?i).*(Entropy|Enthalpy|Gibbs|Heat_Cap|Thermo|ZPE).*", "mole", "熱力学"),
    # 振動
    (r"(?i).*(Freq|Vibrat|IR_|Raman).*", "mole", "振動"),
    # エネルギー
    (r"(?i).*(Energy|TotalE).*", "mole", "エネルギー"),
    # フィンガープリント / 埋め込み
    (r"(?i).*(FP|Fingerprint|Morgan|MACCS|Embed|BERT|GNN).*", "context", "ビット/埋め込み"),
    # 3D
    (r"(?i).*(3D|Distance|Inertia|Aspher|Eccentric|Gyration).*", "context", "3D幾何"),
    # アンサンブル
    (r"(?i)^ens_.*", "context", "アンサンブル統計"),
    # 信頼度
    (r"(?i)^conf_.*", "context", "信頼度スコア"),
]


def classify_descriptor(descriptor_name: str) -> tuple[WeightingType, str]:
    """
    記述子名から加重方法を判定する。

    1. EXPLICIT_DESCRIPTOR_MAP に明示的エントリがあればそれを使用
    2. なければ正規表現フォールバックルールで分類
    3. それでもマッチしなければ "context" を返す

    Returns:
        (weighting_type, rationale)
    """
    # 1. 明示的マッピング
    if descriptor_name in EXPLICIT_DESCRIPTOR_MAP:
        return EXPLICIT_DESCRIPTOR_MAP[descriptor_name]

    # 2. 正規表現フォールバック
    for pattern, wtype, rationale in _FALLBACK_RULES:
        if re.match(pattern, descriptor_name):
            return wtype, f"自動分類({rationale})"

    # 3. デフォルト
    return "context", "分類不能: 安全側で文脈依存扱い"


def classify_all(
    descriptor_names: list[str],
) -> dict[str, tuple[WeightingType, str]]:
    """
    記述子名リストを一括分類する。

    Returns:
        {name: (weighting_type, rationale)}
    """
    return {name: classify_descriptor(name) for name in descriptor_names}


def get_weighting_summary(
    descriptor_names: list[str],
) -> dict[str, int]:
    """
    分類結果のサマリーを返す。

    Returns:
        {"weight": N, "mole": M, "context": K}
    """
    classified = classify_all(descriptor_names)
    counts = {"weight": 0, "mole": 0, "context": 0}
    for wtype, _ in classified.values():
        counts[wtype] = counts.get(wtype, 0) + 1
    return counts
