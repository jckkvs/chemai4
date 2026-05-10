"""
backend/chem/rdkit_adapter.py

RDKit を使った化合物特徴量化アダプタ。
Descriptors.descList の全217記述子 + Gasteiger電荷 + フィンガープリントを計算。
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorResult
from backend.utils.optional_import import safe_import

_rdkit = safe_import("rdkit", "rdkit")
logger = logging.getLogger(__name__)


# ─── 主要記述子の日本語説明・カテゴリ辞書 ──────────────────────
# format: {RDKit内部名: (日本語名, カテゴリ, 物理化学的意味)}
_DESCRIPTOR_JP_META: dict[str, tuple[str, str, str]] = {
    # ── 分子サイズ・形状 ──
    "MolWt":                ("分子量",          "分子サイズ",   "分子の質量（Da）。大きいほど沸点・粘度が上がる"),
    "HeavyAtomMolWt":      ("重原子分子量",     "分子サイズ",   "水素を除いた骨格部分の質量"),
    "ExactMolWt":           ("精密分子量",       "分子サイズ",   "同位体を考慮した正確な分子量（質量分析に対応）"),
    "HeavyAtomCount":       ("重原子数",         "分子サイズ",   "水素以外の原子の総数。分子骨格の大きさを反映"),
    "NumAtoms":             ("全原子数",         "分子サイズ",   "水素含む全原子数"),
    "LabuteASA":            ("近似溶媒接触面積", "分子サイズ",   "溶媒に接触できる表面積。大きいほど溶媒との相互作用が多い"),

    # ── 極性・溶解性 ──
    "MolLogP":              ("LogP",             "極性・溶解性", "油/水への溶けやすさの指標。正→油に溶けやすい、負→水に溶けやすい"),
    "TPSA":                 ("極性表面積",       "極性・溶解性", "O,N由来の極性表面の広さ。大きいと水に溶けやすく膜を通りにくい"),
    "MolMR":                ("モル屈折率",       "極性・溶解性", "分子が光を曲げる力。分極しやすさ＝分子間力の強さを反映"),

    # ── 電子状態 ──
    "MaxPartialCharge":     ("最大部分電荷",     "電子状態",     "最もプラスに帯電した原子の電荷。求電子反応の起点"),
    "MinPartialCharge":     ("最小部分電荷",     "電子状態",     "最もマイナスに帯電した原子の電荷。求核反応の起点"),
    "MaxAbsPartialCharge":  ("最大|部分電荷|",   "電子状態",     "電荷の偏りが最も大きい原子。反応性の目安"),
    "MinAbsPartialCharge":  ("最小|部分電荷|",   "電子状態",     "電荷が最も均一な原子"),
    "NumRadicalElectrons":  ("ラジカル電子数",   "電子状態",     "ペアを組んでいない電子の数。多いとラジカル反応を起こしやすい"),
    "NumValenceElectrons":  ("価電子数",         "電子状態",     "化学結合に参加できる電子の総数"),

    # ── 水素結合 ──
    "NumHAcceptors":        ("水素結合受容体数", "水素結合",     "水素結合を受け入れるO,N等の数。水溶性・薬品の吸収に影響"),
    "NumHDonors":           ("水素結合供与体数", "水素結合",     "水素結合を提供する-OH,-NH等の数。結晶性・粘度に影響"),
    "NHOHCount":            ("N-OH/N-H数",       "水素結合",     "アルコールやアミンの水素結合可能部位の総数"),
    "NOCount":              ("NO数",              "水素結合",     "窒素と酸素の合計数。極性の簡易指標"),

    # ── トポロジー ──
    "NumRotatableBonds":    ("回転可能結合数",   "トポロジー",   "自由に回転できる結合の数。多いほど分子が柔らかく折りたためる"),
    "RingCount":            ("環の総数",         "トポロジー",   "すべての環の数。剛直性・安定性に寄与"),
    "NumAromaticRings":     ("芳香環数",         "トポロジー",   "π電子共役した平面環の数。光吸収・熱安定性に寄与"),
    "NumAliphaticRings":    ("脂肪族環数",       "トポロジー",   "芳香族でない環の数。立体構造の制約に影響"),
    "NumSaturatedRings":    ("飽和環数",         "トポロジー",   "二重結合を含まない環の数。柔軟性のある立体構造"),
    "NumHeterocycles":      ("ヘテロ環数",       "トポロジー",   "N,O,S等を含む環の数。生理活性に重要"),
    "NumAromaticHeterocycles": ("芳香族ヘテロ環数", "トポロジー", "芳香族性を持つヘテロ環。薬品設計の基本骨格"),
    "NumAromaticCarbocycles": ("芳香族炭素環数", "トポロジー",   "ベンゼン環などの炭素だけの芳香環の数"),
    "NumAliphaticHeterocycles": ("脂肪族ヘテロ環数", "トポロジー", "芳香族でないヘテロ環の数"),
    "NumAliphaticCarbocycles": ("脂肪族炭素環数", "トポロジー",   "シクロヘキサン等の芳香族でない炭素環の数"),
    "NumSaturatedHeterocycles": ("飽和ヘテロ環数", "トポロジー",  "不飽和結合のないヘテロ環の数"),
    "NumSaturatedCarbocycles": ("飽和炭素環数",   "トポロジー",   "不飽和結合のない炭素環の数"),
    "FractionCSP3":         ("sp3炭素割合",      "トポロジー",   "sp3の割合。高い→立体的(薬品向き)、低い→平面的(有機EL向き)"),

    # ── 位相的指標（分子グラフの数学的特徴） ──
    "BalabanJ":             ("Balaban J指数",    "位相的指標",   "分子のグラフ理論的な均一性。高い→対称的な構造"),
    "BertzCT":              ("Bertz複雑度",      "位相的指標",   "分子の構造的複雑さ。分岐・環が多いほど高い"),
    "HallKierAlpha":        ("Hall-Kier α",      "位相的指標",   "原子サイズの効果を考慮した分子形状指標"),
    "Kappa1":               ("κ1",               "位相的指標",   "直線性の指標。直鎖に近いほど大きい"),
    "Kappa2":               ("κ2",               "位相的指標",   "分岐度の指標。分岐が多いほど大きい"),
    "Kappa3":               ("κ3",               "位相的指標",   "空間的な広がりの指標。3次元的な嵩高さ"),
    "Chi0":                 ("χ0",               "位相的指標",   "0次の結合性指数。原子の種類の多様性"),
    "Chi0n":                ("χ0n(原子価)",      "位相的指標",   "原子価を考慮した0次結合性指数"),
    "Chi0v":                ("χ0v(原子価)",      "位相的指標",   "原子価を考慮した0次結合性指数（van der Waals版）"),
    "Chi1":                 ("χ1",               "位相的指標",   "1次の結合性指数。結合の連結パターン"),
    "Chi1n":                ("χ1n(原子価)",      "位相的指標",   "原子価を考慮した1次結合性指数"),
    "Chi1v":                ("χ1v(原子価)",      "位相的指標",   "原子価を考慮した1次結合性指数（van der Waals版）"),
    "Chi2n":                ("χ2n",              "位相的指標",   "2次の結合性指数。2結合先までの結合パターン"),
    "Chi2v":                ("χ2v",              "位相的指標",   "2次結合性指数（van der Waals版）"),
    "Chi3n":                ("χ3n",              "位相的指標",   "3次の結合性指数。3結合先までの分岐パターン"),
    "Chi3v":                ("χ3v",              "位相的指標",   "3次結合性指数（van der Waals版）"),
    "Chi4n":                ("χ4n",              "位相的指標",   "4次の結合性指数。4結合先までの経路"),
    "Chi4v":                ("χ4v",              "位相的指標",   "4次結合性指数（van der Waals版）"),
    "Ipc":                  ("情報含量",          "位相的指標",   "分子グラフの情報量。構造の多様性を反映"),
    "AvgIpc":               ("平均情報含量",      "位相的指標",   "原子あたりの平均情報含量"),

    # ── BCUT記述子（分子全体の固有値分解） ──
    "BCUT2D_MWHI":          ("BCUT MW高",        "BCUT",         "原子量の空間分布パターン（最も偏った方向）"),
    "BCUT2D_MWLOW":         ("BCUT MW低",        "BCUT",         "原子量の空間分布パターン（最も均一な方向）"),
    "BCUT2D_CHGHI":         ("BCUT 電荷高",      "BCUT",         "電荷分布の最も偏った方向のパターン"),
    "BCUT2D_CHGLO":         ("BCUT 電荷低",      "BCUT",         "電荷分布の最も均一な方向のパターン"),
    "BCUT2D_LOGPHI":        ("BCUT LogP高",      "BCUT",         "疎水性の空間的な偏り（最大方向）"),
    "BCUT2D_LOGPLOW":       ("BCUT LogP低",      "BCUT",         "疎水性の空間的な偏り（最小方向）"),
    "BCUT2D_MRHI":          ("BCUT MR高",        "BCUT",         "分極率の空間的偏り（最大方向）"),
    "BCUT2D_MRLOW":         ("BCUT MR低",        "BCUT",         "分極率の空間的偏り（最小方向）"),

    # ── EState指標（原子の電気化学的環境） ──
    "MaxAbsEStateIndex":    ("最大|EState|",     "EState",       "最も電気化学的に影響力のある原子の指標"),
    "MaxEStateIndex":       ("最大EState",        "EState",       "最も電子を受け取りやすい原子の指標"),
    "MinAbsEStateIndex":    ("最小|EState|",     "EState",       "最も電気化学的に中立な原子の指標"),
    "MinEStateIndex":       ("最小EState",        "EState",       "最も電子を放出しやすい原子の指標"),

    # ── VSA系（表面積の物性値別分布） ──
    # PEOE_VSA: 部分電荷別に表面積をビン分割
    "PEOE_VSA1":            ("PEOE_VSA1",        "表面積分布",   "最も負に帯電した原子の表面積(電荷<-0.30)"),
    "PEOE_VSA2":            ("PEOE_VSA2",        "表面積分布",   "やや負に帯電した原子の表面積(-0.30~-0.25)"),
    "PEOE_VSA3":            ("PEOE_VSA3",        "表面積分布",   "弱く負に帯電した原子の表面積(-0.25~-0.20)"),
    "PEOE_VSA4":            ("PEOE_VSA4",        "表面積分布",   "わずかに負の原子の表面積(-0.20~-0.15)"),
    "PEOE_VSA5":            ("PEOE_VSA5",        "表面積分布",   "ほぼ中性の原子の表面積(-0.15~-0.10)"),
    "PEOE_VSA6":            ("PEOE_VSA6",        "表面積分布",   "中性付近の原子の表面積(-0.10~-0.05)"),
    "PEOE_VSA7":            ("PEOE_VSA7",        "表面積分布",   "中性の原子の表面積(-0.05~0.00)"),
    "PEOE_VSA8":            ("PEOE_VSA8",        "表面積分布",   "わずかに正の原子の表面積(0.00~0.05)"),
    "PEOE_VSA9":            ("PEOE_VSA9",        "表面積分布",   "弱く正に帯電した原子の表面積(0.05~0.10)"),
    "PEOE_VSA10":           ("PEOE_VSA10",       "表面積分布",   "やや正に帯電した原子の表面積(0.10~0.15)"),
    "PEOE_VSA11":           ("PEOE_VSA11",       "表面積分布",   "正に帯電した原子の表面積(0.15~0.20)"),
    "PEOE_VSA12":           ("PEOE_VSA12",       "表面積分布",   "強く正に帯電した原子の表面積(0.20~0.25)"),
    "PEOE_VSA13":           ("PEOE_VSA13",       "表面積分布",   "最も正に帯電した原子の表面積(0.25~0.30)"),
    "PEOE_VSA14":           ("PEOE_VSA14",       "表面積分布",   "非常に正に帯電した原子の表面積(>0.30)"),
    # SMR_VSA: 屈折率(分極率)別に表面積をビン分割
    "SMR_VSA1":             ("SMR_VSA1",         "表面積分布",   "屈折率の低い原子の表面積（分極しにくい部分）"),
    "SMR_VSA2":             ("SMR_VSA2",         "表面積分布",   "やや低屈折率の原子の表面積"),
    "SMR_VSA3":             ("SMR_VSA3",         "表面積分布",   "中程度の屈折率の原子の表面積"),
    "SMR_VSA4":             ("SMR_VSA4",         "表面積分布",   "やや高屈折率の原子の表面積"),
    "SMR_VSA5":             ("SMR_VSA5",         "表面積分布",   "高屈折率の原子の表面積"),
    "SMR_VSA6":             ("SMR_VSA6",         "表面積分布",   "非常に高屈折率の原子の表面積"),
    "SMR_VSA7":             ("SMR_VSA7",         "表面積分布",   "屈折率が最大の原子の表面積（最も分極しやすい部分）"),
    "SMR_VSA8":             ("SMR_VSA8",         "表面積分布",   "極度に高い屈折率の原子の表面積"),
    "SMR_VSA9":             ("SMR_VSA9",         "表面積分布",   "最大屈折率領域の表面積"),
    "SMR_VSA10":            ("SMR_VSA10",        "表面積分布",   "超高屈折率領域の表面積"),
    # SlogP_VSA: LogP(疎水性)別に表面積をビン分割
    "SlogP_VSA1":           ("SlogP_VSA1",       "表面積分布",   "最も親水的な原子の表面積"),
    "SlogP_VSA2":           ("SlogP_VSA2",       "表面積分布",   "親水的な原子の表面積"),
    "SlogP_VSA3":           ("SlogP_VSA3",       "表面積分布",   "やや親水的な原子の表面積"),
    "SlogP_VSA4":           ("SlogP_VSA4",       "表面積分布",   "中性付近の原子の表面積"),
    "SlogP_VSA5":           ("SlogP_VSA5",       "表面積分布",   "やや疎水的な原子の表面積"),
    "SlogP_VSA6":           ("SlogP_VSA6",       "表面積分布",   "疎水的な原子の表面積"),
    "SlogP_VSA7":           ("SlogP_VSA7",       "表面積分布",   "強く疎水的な原子の表面積"),
    "SlogP_VSA8":           ("SlogP_VSA8",       "表面積分布",   "非常に疎水的な原子の表面積"),
    "SlogP_VSA9":           ("SlogP_VSA9",       "表面積分布",   "極めて疎水的な原子の表面積"),
    "SlogP_VSA10":          ("SlogP_VSA10",      "表面積分布",   "最も疎水的な原子の表面積"),
    "SlogP_VSA11":          ("SlogP_VSA11",      "表面積分布",   "超疎水的な原子の表面積"),
    "SlogP_VSA12":          ("SlogP_VSA12",      "表面積分布",   "超疎水的領域の追加表面積"),
    # EState_VSA: 電気化学的状態別に表面積をビン分割
    "EState_VSA1":          ("EState_VSA1",      "表面積分布",   "最も電子リッチな原子の表面積"),
    "EState_VSA2":          ("EState_VSA2",      "表面積分布",   "電子リッチな原子の表面積"),
    "EState_VSA3":          ("EState_VSA3",      "表面積分布",   "やや電子リッチな原子の表面積"),
    "EState_VSA4":          ("EState_VSA4",      "表面積分布",   "中性付近の原子の表面積"),
    "EState_VSA5":          ("EState_VSA5",      "表面積分布",   "やや電子プアな原子の表面積"),
    "EState_VSA6":          ("EState_VSA6",      "表面積分布",   "電子プアな原子の表面積"),
    "EState_VSA7":          ("EState_VSA7",      "表面積分布",   "かなり電子プアな原子の表面積"),
    "EState_VSA8":          ("EState_VSA8",      "表面積分布",   "強く電子プアな原子の表面積"),
    "EState_VSA9":          ("EState_VSA9",      "表面積分布",   "非常に電子プアな原子の表面積"),
    "EState_VSA10":         ("EState_VSA10",     "表面積分布",   "最も電子プアな原子の表面積"),
    "EState_VSA11":         ("EState_VSA11",     "表面積分布",   "極度に電子プアな領域の表面積"),
    # VSA_EState: EState値の表面積加重統計
    "VSA_EState1":          ("VSA_EState1",      "表面積分布",   "表面積で加重したEState指標ビン1"),
    "VSA_EState2":          ("VSA_EState2",      "表面積分布",   "表面積で加重したEState指標ビン2"),
    "VSA_EState3":          ("VSA_EState3",      "表面積分布",   "表面積で加重したEState指標ビン3"),
    "VSA_EState4":          ("VSA_EState4",      "表面積分布",   "表面積で加重したEState指標ビン4"),
    "VSA_EState5":          ("VSA_EState5",      "表面積分布",   "表面積で加重したEState指標ビン5"),
    "VSA_EState6":          ("VSA_EState6",      "表面積分布",   "表面積で加重したEState指標ビン6"),
    "VSA_EState7":          ("VSA_EState7",      "表面積分布",   "表面積で加重したEState指標ビン7"),
    "VSA_EState8":          ("VSA_EState8",      "表面積分布",   "表面積で加重したEState指標ビン8"),
    "VSA_EState9":          ("VSA_EState9",      "表面積分布",   "表面積で加重したEState指標ビン9"),
    "VSA_EState10":         ("VSA_EState10",     "表面積分布",   "表面積で加重したEState指標ビン10"),

    # ── 薬品適性 ──
    "qed":                  ("QED(薬品適性)",     "薬品適性",     "薬品としての望ましさスコア(0-1)。高い→薬に向いた構造"),
    "SPS":                  ("SPS指数",           "薬品適性",     "合成容易性を考慮した薬品スコア"),

    # ── フィンガープリント密度 ──
    "FpDensityMorgan1":     ("FP密度Morgan1",    "FP密度",       "半径1の化学環境の密度。局所的な多様性"),
    "FpDensityMorgan2":     ("FP密度Morgan2",    "FP密度",       "半径2の化学環境の密度。中距離の多様性"),
    "FpDensityMorgan3":     ("FP密度Morgan3",    "FP密度",       "半径3の化学環境の密度。広域の多様性"),

    # ── その他の重要記述子 ──
    "NumAmideBonds":        ("アミド結合数",     "官能基",       "ペプチド結合に代表されるアミド結合の数"),
    "NumBridgeheadAtoms":   ("橋頭原子数",       "トポロジー",   "二環式以上の橋頭位置にある原子の数"),
    "NumSpiroAtoms":        ("スピロ原子数",     "トポロジー",   "2つの環が1原子を共有するスピロ構造の数"),
}

# fr_系（官能基フラグメントカウント）は自動で日本語名を付与
_FR_DESCRIPTORS_JP: dict[str, str] = {
    "fr_Al_COO": "脂肪族カルボン酸", "fr_Al_OH": "脂肪族ヒドロキシル",
    "fr_Al_OH_noTert": "脂肪族OH(3級以外)", "fr_ArN": "芳香族窒素",
    "fr_Ar_COO": "芳香族カルボン酸", "fr_Ar_N": "芳香族アミン",
    "fr_Ar_NH": "芳香族NH", "fr_Ar_OH": "フェノール性OH",
    "fr_COO": "カルボン酸(エステル含)", "fr_COO2": "カルボン酸",
    "fr_C_O": "カルボニル", "fr_C_O_noCOO": "カルボニル(COO除く)",
    "fr_C_S": "チオカルボニル", "fr_HOCCN": "β-ヒドロキシアミン",
    "fr_Imine": "イミン", "fr_NH0": "3級アミン",
    "fr_NH1": "2級アミン", "fr_NH2": "1級アミン",
    "fr_N_O": "N-酸化物", "fr_Ndealkylation1": "N-脱アルキル化部位1",
    "fr_Ndealkylation2": "N-脱アルキル化部位2",
    "fr_Nhpyrrole": "ピロール型NH", "fr_SH": "チオール",
    "fr_aldehyde": "アルデヒド", "fr_alkyl_carbamate": "アルキルカーバメート",
    "fr_alkyl_halide": "アルキルハライド", "fr_allylic_oxid": "アリル酸化部位",
    "fr_amide": "アミド", "fr_amidine": "アミジン",
    "fr_aniline": "アニリン", "fr_azide": "アジド",
    "fr_azo": "アゾ", "fr_barbitur": "バルビツール酸",
    "fr_benzene": "ベンゼン環", "fr_benzodiazepine": "ベンゾジアゼピン",
    "fr_bicyclic": "二環式", "fr_diazo": "ジアゾ",
    "fr_dihydropyridine": "ジヒドロピリジン", "fr_epoxide": "エポキシド",
    "fr_ester": "エステル", "fr_ether": "エーテル",
    "fr_furan": "フラン", "fr_guanido": "グアニジン",
    "fr_halogen": "ハロゲン", "fr_hdrzine": "ヒドラジン",
    "fr_hdrzone": "ヒドラゾン", "fr_imidazole": "イミダゾール",
    "fr_imide": "イミド", "fr_isocyan": "イソシアネート",
    "fr_isothiocyan": "イソチオシアネート", "fr_ketone": "ケトン",
    "fr_ketone_Topliss": "ケトン(Topliss)", "fr_lactam": "ラクタム",
    "fr_lactone": "ラクトン", "fr_methoxy": "メトキシ",
    "fr_morpholine": "モルホリン", "fr_nitrile": "ニトリル",
    "fr_nitro": "ニトロ", "fr_nitro_arom": "芳香族ニトロ",
    "fr_nitro_arom_nonortho": "芳香族ニトロ(非オルト)",
    "fr_nitroso": "ニトロソ", "fr_oxazole": "オキサゾール",
    "fr_oxime": "オキシム", "fr_para_hydroxylation": "パラ水酸化部位",
    "fr_phenol": "フェノール", "fr_phenol_noOrthoHbond": "フェノール(オルトHB無)",
    "fr_phos_acid": "リン酸", "fr_phos_ester": "リン酸エステル",
    "fr_piperdine": "ピペリジン", "fr_piperzine": "ピペラジン",
    "fr_priamide": "1級アミド", "fr_prisulfonamd": "1級スルホンアミド",
    "fr_pyridine": "ピリジン", "fr_quatN": "4級窒素",
    "fr_sulfide": "スルフィド", "fr_sulfonamd": "スルホンアミド",
    "fr_sulfone": "スルホン", "fr_term_acetylene": "末端アセチレン",
    "fr_tetrazole": "テトラゾール", "fr_thiazole": "チアゾール",
    "fr_thiocyan": "チオシアネート", "fr_thiophene": "チオフェン",
    "fr_unbrch_alkane": "非分岐アルカン", "fr_urea": "ウレア",
}


class RDKitAdapter(BaseChemAdapter):
    """
    RDKit による化合物記述子計算アダプタ。

    計算内容:
    - RDKit Descriptors.descList の全記述子（217個）
    - Gasteiger 部分電荷統計量（5個）
    - Morgan フィンガープリント（ECFP4: radius=2）
    - RDKit フィンガープリント（2048 bit）
    - MACCS Keys

    Args:
        compute_fp: フィンガープリントも計算するか
        morgan_radius: Morgan FPの半径（デフォルト2 = ECFP4）
        morgan_bits: Morgan FPのビット数
        rdkit_fp_bits: RDKit FPのビット数
        include_maccs: MACCS keysを含めるか
    """

    def __init__(
        self,
        compute_fp: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
        morgan_count: bool = False,
        morgan_order_by_appearance: bool = False,
        rdkit_fp_bits: int = 2048,
        include_maccs: bool = False,
        compute_gasteiger: bool = True,
    ) -> None:
        self.compute_fp = compute_fp
        self.morgan_radius = morgan_radius
        self.morgan_bits = morgan_bits
        self.morgan_count = morgan_count
        self.morgan_order_by_appearance = morgan_order_by_appearance
        self.rdkit_fp_bits = rdkit_fp_bits
        self.include_maccs = include_maccs
        self.compute_gasteiger = compute_gasteiger

    @property
    def name(self) -> str:
        return "rdkit"

    @property
    def description(self) -> str:
        return "RDKit 全記述子（217個）+ 部分電荷 + フィンガープリント計算"

    def is_available(self) -> bool:
        return bool(_rdkit)

    def compute(
        self,
        smiles_list: list[str],
        charge_config_store: Any | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES リストから RDKit 全記述子を計算する。
        """
        self._require_available()
        logger.info(f"🔬 RDKit記述子計算を開始します（{len(smiles_list)}分子）...")

        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, MACCSkeys
        from rdkit.Chem import rdPartialCharges

        # 全記述子リストを取得
        desc_list = Descriptors.descList  # list of (name, function)

        rows: list[dict[str, float]] = []
        failed: list[int] = []
        bit_first_appearance: dict[int, int] = {} if self.morgan_order_by_appearance else None
        all_morgan_bits: set[int] = set() if self.morgan_order_by_appearance else None

        for idx, smi in enumerate(smiles_list):
            try:
                # プロトン化変換を適用
                if charge_config_store is not None:
                    cfg = charge_config_store.get_config(smi)
                    from backend.chem.protonation import apply_protonation
                    smi_to_use = apply_protonation(smi, cfg)
                else:
                    smi_to_use = smi

                mol = Chem.MolFromSmiles(smi_to_use)
                if mol is None:
                    mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"無効なSMILES: {smi!r}")

                row: dict[str, float] = {}

                # ── 全Descriptors.descList記述子を計算 ──
                for desc_name, desc_fn in desc_list:
                    try:
                        val = desc_fn(mol)
                        row[desc_name] = float(val) if val is not None and np.isfinite(float(val)) else np.nan
                    except Exception:
                        row[desc_name] = np.nan

                # ── Gasteiger 部分電荷統計量 ──
                if self.compute_gasteiger:
                    try:
                        mol_h = Chem.AddHs(mol)
                        rdPartialCharges.ComputeGasteigerCharges(mol_h)
                        charges = [
                            float(mol_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
                            for i in range(mol_h.GetNumAtoms())
                        ]
                        charges = [q for q in charges if np.isfinite(q)]
                        if charges:
                            ca = np.array(charges)
                            row["gasteiger_q_max"]      = float(np.max(ca))
                            row["gasteiger_q_min"]      = float(np.min(ca))
                            row["gasteiger_q_range"]    = float(np.max(ca) - np.min(ca))
                            row["gasteiger_q_std"]      = float(np.std(ca))
                            row["gasteiger_q_abs_mean"] = float(np.mean(np.abs(ca)))
                        else:
                            for k in ["gasteiger_q_max", "gasteiger_q_min",
                                      "gasteiger_q_range", "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                                row[k] = np.nan
                    except Exception:
                        for k in ["gasteiger_q_max", "gasteiger_q_min",
                                  "gasteiger_q_range", "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                            row[k] = np.nan

                # ── Morgan フィンガープリント (ECFP4) ──
                if self.compute_fp:
                    try:
                        if self.morgan_count:
                            # 数え上げ系（カウントベース）のMorganフィンガープリント
                            # GetHashedMorganFingerprint は UIntSparseIntVect を返す（スパースカウント）
                            from rdkit.Chem import rdMolDescriptors
                            morgan_fp = rdMolDescriptors.GetHashedMorganFingerprint(
                                mol, self.morgan_radius, nBits=self.morgan_bits
                            )
                            # 非ゼロ要素を取得（bit_id -> count)
                            nonzero = morgan_fp.GetNonzeroElements()
                            for bit_id, count in nonzero.items():
                                if bit_id < self.morgan_bits:
                                    key = f"Morgan_r{self.morgan_radius}_{bit_id}"
                                    row[key] = float(count)
                                    if self.morgan_order_by_appearance:
                                        if bit_id not in bit_first_appearance:
                                            bit_first_appearance[bit_id] = idx
                                        all_morgan_bits.add(bit_id)
                            # ゼロの場合もカラムを確保（順序付けない場合）
                            if not self.morgan_order_by_appearance:
                                for j in range(self.morgan_bits):
                                    key = f"Morgan_r{self.morgan_radius}_{j}"
                                    if key not in row:
                                        row[key] = 0.0
                        else:
                            # 通常のビットベクトル（0/1）
                            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                                mol, self.morgan_radius, nBits=self.morgan_bits
                            )
                            for j, bit in enumerate(morgan_fp):
                                if bit or not self.morgan_order_by_appearance:
                                    row[f"Morgan_r{self.morgan_radius}_{j}"] = float(bit)
                                    if self.morgan_order_by_appearance and bit:
                                        if j not in bit_first_appearance:
                                            bit_first_appearance[j] = idx
                                        all_morgan_bits.add(j)
                    except Exception as e:
                        logger.warning(f"Morgan FP計算エラー (idx={idx}): {e}")
                        for j in range(self.morgan_bits):
                            row[f"Morgan_r{self.morgan_radius}_{j}"] = 0.0

                    # RDKit トポロジカルフィンガープリント
                    try:
                        rdkit_fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_fp_bits)
                        for j, bit in enumerate(rdkit_fp):
                            row[f"RDKitFP_{j}"] = float(bit)
                    except Exception:
                        for j in range(self.rdkit_fp_bits):
                            row[f"RDKitFP_{j}"] = 0.0

                    # MACCS Keys (166 bit)
                    if self.include_maccs:
                        try:
                            maccs = MACCSkeys.GenMACCSKeys(mol)
                            for j, bit in enumerate(maccs):
                                row[f"MACCS_{j}"] = float(bit)
                        except Exception:
                            for j in range(167):
                                row[f"MACCS_{j}"] = 0.0

                rows.append(row)

            except Exception as e:
                logger.warning(f"RDKit: index={idx}, SMILES={smi!r}: {e}")
                failed.append(idx)
                rows.append({})

        df = pd.DataFrame(rows).fillna(0.0)

        # Morganビットをデータセットでの出現順に並べ替え
        if self.morgan_order_by_appearance and bit_first_appearance:
            # 出現順でビットをソート
            sorted_bits = sorted(bit_first_appearance.keys(),
                               key=lambda j: bit_first_appearance[j])
            morgan_prefix = f"Morgan_r{self.morgan_radius}_"
            morgan_cols = [f"{morgan_prefix}{j}" for j in sorted_bits]
            other_cols = [c for c in df.columns if not c.startswith(morgan_prefix)]
            # 新しい列順序でDataFrameを再構築
            df = df[other_cols + morgan_cols]
            logger.info(
                f"Morganビットを出現順に並べ替えました（{len(sorted_bits)}ビット）"
            )

        logger.info(
            f"RDKit計算完了: {len(smiles_list)}件 / "
            f"失敗={len(failed)} / 記述子={df.shape[1]}"
        )
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
            metadata={
                "morgan_radius": self.morgan_radius,
                "morgan_bits": self.morgan_bits if self.compute_fp else 0,
                "total_descriptors": df.shape[1],
            },
        )

    def get_descriptors_metadata(self) -> list:
        """RDKit全記述子の詳細メタデータを返す。"""
        from backend.chem.base import DescriptorMetadata

        try:
            from rdkit.Chem import Descriptors
            desc_names = [name for name, _ in Descriptors.descList]
        except Exception:
            desc_names = list(_DESCRIPTOR_JP_META.keys())

        meta = []
        for name in desc_names:
            jp_info = _DESCRIPTOR_JP_META.get(name)
            if jp_info:
                jp_name, category, meaning = jp_info
                is_count = name.startswith("fr_") or "Count" in name or "Num" in name
                meta.append(DescriptorMetadata(name, f"{jp_name}：{meaning}", is_count=is_count))
            elif name.startswith("fr_"):
                jp_name = _FR_DESCRIPTORS_JP.get(name, name.replace("fr_", "").replace("_", " "))
                meta.append(DescriptorMetadata(name, f"官能基：{jp_name}", is_count=True))
            elif name.startswith("VSA_EState"):
                meta.append(DescriptorMetadata(name, f"EState表面積：{name}", is_count=False))
            elif name.startswith("PEOE_VSA"):
                meta.append(DescriptorMetadata(name, f"部分電荷表面積：{name}", is_count=False))
            elif name.startswith("SMR_VSA"):
                meta.append(DescriptorMetadata(name, f"MR表面積：{name}", is_count=False))
            elif name.startswith("SlogP_VSA"):
                meta.append(DescriptorMetadata(name, f"LogP表面積：{name}", is_count=False))
            elif name.startswith("EState_VSA"):
                meta.append(DescriptorMetadata(name, f"EState-VSA：{name}", is_count=False))
            elif name.startswith("Chi") or name.startswith("Kappa"):
                meta.append(DescriptorMetadata(name, f"位相的指標：{name}", is_count=False))
            else:
                meta.append(DescriptorMetadata(name, name, is_count=False))

        # Morgan フィンガープリントのメタデータ
        if self.compute_fp:
            fp_type = "数え上げ系（カウント）" if self.morgan_count else "有無系（ビット）"
            for j in range(self.morgan_bits):
                bit_name = f"Morgan_r{self.morgan_radius}_{j}"
                if self.morgan_count:
                    desc = (
                        f"Morgan FP (r={self.morgan_radius}) ビット{j}: "
                        f"半径{self.morgan_radius}の部分構造の出現回数。{fp_type}。"
                        f"部分構造がデータセットで出現した順に並ぶ。"
                    )
                else:
                    desc = (
                        f"Morgan FP (r={self.morgan_radius}) ビット{j}: "
                        f"半径{self.morgan_radius}の部分構造の有無（0/1）。{fp_type}。"
                    )
                meta.append(DescriptorMetadata(
                    name=bit_name,
                    meaning=desc,
                    is_count=self.morgan_count,
                ))

        # Gasteiger
        for gn, gd in [
            ("gasteiger_q_max", "Gasteiger最大部分電荷"),
            ("gasteiger_q_min", "Gasteiger最小部分電荷"),
            ("gasteiger_q_range", "Gasteiger電荷レンジ"),
            ("gasteiger_q_std", "Gasteiger電荷標準偏差"),
            ("gasteiger_q_abs_mean", "Gasteiger|電荷|平均"),
        ]:
            meta.append(DescriptorMetadata(gn, gd, is_count=False))

        return meta

    def get_descriptor_names(self) -> list[str]:
        """
        計算可能な記述子名のリストを返す。
        """
        names = [m.name for m in self.get_descriptors_metadata()]
        if self.compute_fp:
            names += [f"Morgan_r{self.morgan_radius}_{j}" for j in range(self.morgan_bits)]
            names += [f"RDKitFP_{j}" for j in range(self.rdkit_fp_bits)]
            if self.include_maccs:
                names += [f"MACCS_{j}" for j in range(167)]
        return names

    @staticmethod
    def get_descriptor_jp_info(name: str) -> tuple[str, str, str] | None:
        """記述子名から (日本語名, カテゴリ, 意味) のタプルを返す。"""
        if name in _DESCRIPTOR_JP_META:
            return _DESCRIPTOR_JP_META[name]
        if name in _FR_DESCRIPTORS_JP:
            return (f"官能基：{_FR_DESCRIPTORS_JP[name]}", "官能基カウント", f"{_FR_DESCRIPTORS_JP[name]}の数")
        return None
