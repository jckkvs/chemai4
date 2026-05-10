"""
backend/chem/recommender.py

目的変数の予測に対する「推奨説明変数」を提供するモジュール。

1. ルールベース推奨: RDKit, XTB, COSMO-RS等のライブラリごとに、
   事前に理にかなった記述子を8個定義（既存のまま）。
2. LLMベース推奨: 4000+ の全特徴量から、LLMを使って
   目的変数に適した特徴量を選択（新規追加）。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
import os
import re

@dataclass
class DescriptorInfo:
    """推奨される説明変数のメタデータ"""
    name: str              # 記述子名（プログラム上でのカラム名等）
    library: str           # 計算ライブラリ (RDKit, XTB, COSMO-RS, Uni-pKa, GroupContribution)
    meaning: str           # 物理・化学的な意味付け
    source: str            # 関連論文や根拠となる出典
    category: str = "その他" # 記述子の物理的意味の分類（例：立体・形状系、量子化学系など）

@dataclass
class TargetRecommendations:
    """ある目的変数に関する推奨セット"""
    target_name: str       # 目的変数の表示名
    summary: str           # 目的変数の予測に関する事前知識（どのような変数が支配的か）
    category: str          # 目的変数の系統分類（例：光・電磁気系、力学・強度系など）
    descriptors: List[DescriptorInfo]

# 各目的変数の事前知識データベース
_RECOMMENDATION_DATA = [
    TargetRecommendations(
        target_name="屈折率 (Refractive Index)",
        summary="ローレンツ・ローレンツの式に基づき、材料の密度と分子の分極率が支配的なパラメータとなります。π電子系（芳香環）の多さや密なパッキングが屈折率を高めます。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("MolMR", "RDKit", "モル屈折（分極率の直接的な指標）", "Lorentz-Lorenz equation", "量子化学・電子状態系"),
            DescriptorInfo("Polarizability", "XTB", "分極率（電場に対する電子雲の応答性）", "GFN2-xTB / QC calculation", "量子化学・電子状態系"),
            DescriptorInfo("Density", "COSMO-RS", "密度（単位体積あたりの質量、屈折率に直結）", "COSMOTherm / OpenCOSMO-RS density models", "熱力学・相互作用系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "General Physical Chemistry", "立体・形状系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数（π電子の非局在化による高分極化）", "Bicerano (2002) Prediction of Polymer Properties", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素の割合（値が低いほどπ電子系が多く屈折率が高い）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "ファンデルワールス体積（パッキング密度の指標）", "McGowan Volume / Bicerano", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="吸収率 (Absorption / UV-Vis)",
        summary="光の吸収は電子遷移によるため、HOMO-LUMOギャップや共役系の長さが最も影響します。部分電荷や特定の発色団の存在も重要です。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("HomoLumoGap", "XTB", "HOMO-LUMOエネルギーギャップ（吸収極大波長に反比例）", "sTDA / GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MaxConjugatedChain", "RDKit", "最大共役長（共役が長いほど長波長シフト・高吸収）", "Physical Organic Chemistry", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数", "General Spectroscopy", "トポロジー系"),
            DescriptorInfo("IonizationPotential", "XTB", "イオン化ポテンシャル", "GFN2-xTB / Koopmans' theorem", "量子化学・電子状態系"),
            DescriptorInfo("ElectronAffinity", "XTB", "電子親和力", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("OscillatorStrength", "XTB", "振動子強度（吸収ピークの強度）", "sTDA / sTD-DFT", "量子化学・電子状態系"),
            DescriptorInfo("TPSA", "RDKit", "トポロジカル極性表面積（極性相互作用の影響）", "Ertl et al. (2000)", "極性・官能基系"),
            DescriptorInfo("PartialCharges_Max", "XTB", "最大部分電荷（分極・遷移モーメントの指標）", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="シャルピー強度 (Charpy Impact Strength)",
        summary="衝撃に対するタフネスを示します。ポリマーの絡み合い分子量、主鎖の剛直性、分子間力（凝集エネルギー）や柔軟性が鍵となります。",
        category="力学・強度系",
        descriptors=[
            DescriptorInfo("EntanglementMW", "GroupContribution", "絡み合い分子量（値が小さいほど衝撃に強い）", "Wu (1989) / Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("CohesiveEnergy", "COSMO-RS", "凝集エネルギー密度（分子間力の強さ、降伏挙動に影響）", "Hansen / Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（鎖の柔軟性の指標）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移点（脆性破壊か延性破壊かの境界温度）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素の比率（主鎖の立体障害・柔軟性）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("MolWt", "RDKit", "分子量 / モノマー分子量", "Polymer Physics", "立体・形状系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（強力な分子間ネットワーク）", "RDKit Descriptors", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "RDKit Descriptors", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="破壊強度 (Fracture Strength)",
        summary="降伏または破断に至るまでの最大応力。分子間凝集力や架橋、剛直性が高く寄与します。",
        category="力学・強度系",
        descriptors=[
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度（CED）", "Bicerano (2002) / Polymer Physics", "熱力学・相互作用系"),
            DescriptorInfo("BackboneRigidity", "GroupContribution", "主鎖の剛直性パラメータ（Kuhn長等）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("VanDerWaalsVolume", "RDKit", "モノマーのVan der Waals体積", "McGowan mapping", "立体・形状系"),
            DescriptorInfo("HydrogenBonds", "RDKit", "水素結合能力 (HBA + HBD)", "Polymer mechanics", "極性・官能基系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数（スタッキングによる剛性向上）", "Polymer mechanics", "トポロジー系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "General rule of mixtures", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（極性相互作用による引き合い）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("RingCount", "RDKit", "環の総数（立体障害による塑性変形の抑制）", "Bicerano (2002)", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="弾性率 (Elastic Modulus)",
        summary="変形しにくさ（剛性）を示します。ファンデルワールス体積あたりの凝集エネルギー密度が高く、分子鎖が剛直であるほど高くなります。",
        category="力学・強度系",
        descriptors=[
            DescriptorInfo("CohesiveEnergyDensity", "COSMO-RS", "凝集エネルギー密度（剛性の最大の支配要因）", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "Van der Waals体積", "Bicerano (2002)", "立体・形状系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積", "Ertl et al. (2000)", "極性・官能基系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（低いほど弾性率が高い）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("RingCount", "RDKit", "環の数（主鎖の硬さに比例）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（強い分子間力）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "RDKit", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="Tg (ガラス転移点)",
        summary="ポリマー鎖のミクロなブラウン運動が開始する温度。鎖の剛直性（かさ高い側鎖や主鎖の環）、極性相互作用、自由体積が影響します。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("BackboneFlexibility", "GroupContribution", "主鎖の柔軟性パラメーター", "Bicerano (2002) Prediction of Tg", "トポロジー系"),
            DescriptorInfo("CohesiveEnergy", "COSMO-RS", "凝集エネルギー（分子間チェーンの拘束力）", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（自由体積と運動性に直結）", "RDKit", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（主鎖に含まれるとTg劇的向上）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("FreeVolume", "GroupContribution", "分子的自由体積", "VdW Volume derivations", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（双極子間力による拘束）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積", "RDKit", "極性・官能基系"),
            DescriptorInfo("MolWt", "RDKit", "分子量（Flory-Foxの式による連鎖末端効果）", "Flory-Fox Equation", "立体・形状系"),
        ]
    ),
    TargetRecommendations(
        target_name="pKa (酸解離定数)",
        summary="酸・塩基の強さを表します。脱プロトン化後の共役塩基の安定化（共役による非局在化、電子求引基による誘起効果、溶媒和エネルギー）が重要です。",
        category="界面・溶液系",
        descriptors=[
            DescriptorInfo("pKa_pred", "Uni-pKa", "Uni-pKa等の専用ライブラリによる基本予測値", "Uni-pKa (Deep Learning Model)", "量子化学・電子状態系"),
            DescriptorInfo("PartialCharge_Acidic", "XTB", "酸性プロトン/塩基性原子のGasteiger/XTB部分電荷", "GFN2-xTB / RDKit", "極性・官能基系"),
            DescriptorInfo("SolvationFreeEnergy", "COSMO-RS", "水和自由エネルギー（水溶液中でのイオン安定性）", "COSMO-RS theory (Klamt)", "熱力学・相互作用系"),
            DescriptorInfo("HomoEnergy", "XTB", "HOMOエネルギー（塩基性の指標）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("LumoEnergy", "XTB", "LUMOエネルギー（酸・電子受容性の指標）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("Polarizability", "XTB", "分極率（電荷の分散能力）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MaxConjugatedChain", "RDKit", "最大共役長（共役に基づくアニオンの安定化）", "Organic Chemistry", "トポロジー系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "RDKit", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="毒性 (Toxicity)",
        summary="受容体との結合や体内動態（脂溶性）など。トクシコフォア（毒性発現基）の存在、LogPによる細胞膜透過性などが決定打になり得ます。",
        category="環境・安全性",
        descriptors=[
            DescriptorInfo("MolLogP", "RDKit", "オクタノール/水分配係数（生体バリア透過性）", "Lipinski's Rule of Five", "熱力学・相互作用系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "Lipinski's Rule of Five", "立体・形状系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積", "Lipinski's Rule of Five", "極性・官能基系"),
            DescriptorInfo("Toxicophores", "RDKit", "既知の毒性アラート構造（SMARTSマッチング等）", "Tox21 / Structural Alerts", "トポロジー系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Rule of Five", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "Rule of Five", "極性・官能基系"),
            DescriptorInfo("Electrophilicity", "XTB", "親電子性インデックス（DNAやタンパク質への反応性）", "Density Functional Reactivity Theory", "量子化学・電子状態系"),
            DescriptorInfo("AqueousSolubility", "COSMO-RS", "水溶性 (LogS)", "COSMOTherm", "界面・溶液系"),
        ]
    ),
    TargetRecommendations(
        target_name="粘度 (Viscosity)",
        summary="液体の流れにくさ。分子量、形状（球状か鎖状か）、および分子間相互作用（水素結合、双極子相互作用）によって決定されます。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("MolWt", "RDKit", "分子量（大きいほど絡み合いや慣性が増加）", "General Fluid Mechanics", "立体・形状系"),
            DescriptorInfo("SpherocityIndex", "RDKit", "球形度（球状に近いほど粘度が低くなりやすい）", "RDKit 3D descriptors", "立体・形状系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（鎖の柔軟性と絡み合い）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("CohesiveEnergyDensity", "COSMO-RS", "凝集エネルギー密度（分子を引き離すための抵抗）", "Hansen Solubility Parameters", "熱力学・相互作用系"),
            DescriptorInfo("HydrogenBonds", "RDKit", "水素結合能力 (HBA * HBD) （強いネットワーク形成）", "Intermolecular forces", "極性・官能基系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("MolMR", "RDKit", "モル体積のプロキシ", "RDKit", "立体・形状系"),
            DescriptorInfo("ChainEntanglement", "GroupContribution", "分子鎖絡み合いパラメーター", "Bicerano (2002)", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="密度 (Density)",
        summary="単位体積あたりの質量。重原子（特にハロゲン）の存在や、分子のパッキング効率（凝集エネルギーや立体形状）で決まります。",
        category="熱力学・相転移系",
        descriptors=[
            DescriptorInfo("MolWt", "RDKit", "分子量（質量の直接要因）", "Basic Physics", "立体・形状系"),
            DescriptorInfo("VanDerWaalsVolume", "COSMO-RS", "Van der Waals体積（体積の直接要因）", "COSMO volume / McGowan", "立体・形状系"),
            DescriptorInfo("NumHalogens", "RDKit", "重ハロゲン（Cl, Br, I）の数（著しく密度を上げる）", "RDKit", "極性・官能基系"),
            DescriptorInfo("CohesiveEnergy", "GroupContribution", "凝集エネルギー（パッキングの強さ）", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数（平面構造による密なパッキング）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("SpherocityIndex", "RDKit", "球形度", "RDKit 3D", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（配向極性によるパッキング）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素の比率（立体障害による密度低下）", "Bicerano (2002)", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="誘電率 (Dielectric Constant)",
        summary="電場に対する応答性。永久双極子モーメントと、分子の分極率（電子分極）、および有効体積のバランスからローレンツ・オンサーガー等の式に従います。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（配向分極に支配的）", "GFN2-xTB / Onsager theory", "極性・官能基系"),
            DescriptorInfo("Polarizability", "XTB", "分極率（電子分極に寄与）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MolMR", "RDKit", "モル屈折（分極率の別指標）", "Lorentz-Lorenz derivation", "量子化学・電子状態系"),
            DescriptorInfo("VanDerWaalsVolume", "COSMO-RS", "分子体積（単位体積あたりの双極子密度の分母）", "COSMO-RS", "立体・形状系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積", "RDKit", "極性・官能基系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Polymer Data Handbook", "熱力学・相互作用系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体", "Dielectric behavior of H-bond networks", "極性・官能基系"),
            DescriptorInfo("NumHeteroatoms", "RDKit", "ヘテロ原子(O, N, F等)の数（局所双極子の要因）", "RDKit", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="誘電正接 (Dissipation Factor / Tan Delta)",
        summary="高周波電場におけるエネルギー損失。永久双極子の緩和（動きやすさ）や、不純物（吸水）、主鎖のTgなどが損失に寄与します。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("DipoleMoment", "XTB", "永久双極子モーメント", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（双極子部位の局所運動性）", "Polymer dielectric relaxation", "トポロジー系"),
            DescriptorInfo("Polarizability", "XTB", "分極率", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MolLogP", "RDKit", "脂溶性（吸水＝水分子による巨大な誘電損失を防ぐ指標）", "Dielectric properties of polymers", "熱力学・相互作用系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積（吸水性を示す）", "RDKit", "極性・官能基系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移温度（測定温度とのマージン）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（分子鎖の運動性）", "Bicerano (2002)", "立体・形状系"),
            DescriptorInfo("MolWt", "RDKit", "極性末端基の比率を下げる効果としての分子量", "General Knowledge", "立体・形状系"),
        ]
    ),
    TargetRecommendations(
        target_name="耐湿性 / 吸水率 (Moisture Resistance / Water Absorption)",
        summary="水分子の取り込みやすさ。極性基（-OH, -NH等）の数や疎水性（LogP）、自由体積などが強く影響します。（耐湿性が高い＝吸水率が低い）",
        category="環境・安全性",
        descriptors=[
            DescriptorInfo("MolLogP", "RDKit", "脂溶性（高いほど耐湿性が高い）", "Lipinski / ADMET", "熱力学・相互作用系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積（小さいほど耐湿性が高い）", "Ertl et al. (2000)", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（水分子を捕捉）", "Water absorption theory", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Water absorption theory", "極性・官能基系"),
            DescriptorInfo("HLB", "GroupContribution", "親水性疎水性バランス", "Griffin's method", "熱力学・相互作用系"),
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（水分子が入り込むスペース）", "Bicerano (2002)", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（水との配向相互作用）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数（疎水性パッキング）", "RDKit", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="耐光性 (Light Resistance / UV Stability)",
        summary="紫外線による結合解離や酸化劣化への耐性。HOMO-LUMOギャップ（吸収端）、結合解離エネルギー(BDE)、抗酸化機能の有無が関連します。",
        category="環境・安全性",
        descriptors=[
            DescriptorInfo("HomoLumoGap", "XTB", "HOMO-LUMOギャップ（UV光による励起のされやすさ）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MinBDE", "XTB", "最小結合解離エネルギー（最も切れやすい結合の強度）", "DFT / GFN2-xTB", "熱力学・相互作用系"),
            DescriptorInfo("IonizationPotential", "XTB", "イオン化電位（光酸化のしやすさ）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（UVエネルギーの吸収と熱散逸能力）", "Polymer Photochemistry", "トポロジー系"),
            DescriptorInfo("ConjugatedDoubleBonds", "RDKit", "共役二重結合の数", "RDKit", "トポロジー系"),
            DescriptorInfo("MaxAbsorptionWavelength", "XTB", "最大吸収波長", "sTDA / sTD-DFT", "量子化学・電子状態系"),
            DescriptorInfo("Electrophilicity", "XTB", "親電子性インデックス", "Conceptual DFT", "量子化学・電子状態系"),
            DescriptorInfo("UVStabilizerAlerts", "RDKit", "HALSやフェノール系OH構造の有無(SMARTS)", "Domain Knowledge", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="耐熱性 (Heat Resistance / Thermal Decomposition, Td)",
        summary="熱分解開始温度(Td)など。主鎖の結合エネルギー、芳香環による連鎖硬直化、架橋点、ハロゲン（フッ素など）の難燃性寄与などが影響します。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("MinBDE", "XTB", "最小結合解離エネルギー（熱開裂のトリーガー）", "Polymer Degradation Theory", "熱力学・相互作用系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環の数（主鎖の耐熱安定性を著しく高める）", "Van Krevelen (1990)", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率（低い方が耐熱性が高い）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（少ないほど剛性・耐熱増大）", "RDKit", "トポロジー系"),
            DescriptorInfo("NumFluorine", "RDKit", "フッ素原子の数（C-F結合の強さ、耐熱・難燃性寄与）", "Halogenated polymers literature", "極性・官能基系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "Polymer science", "立体・形状系"),
            DescriptorInfo("CrosslinkDensity", "GroupContribution", "架橋点密度（熱変形温度向上）", "Polymer Data Handbook", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="相溶性 (Compatibility / Miscibility)",
        summary="二種類の物質が混ざり合うかどうかの指標。ハンセン溶解度パラメータ（分散力、極性、水素結合）の距離や凝集エネルギー密度から推定されます。",
        category="界面・溶液系",
        descriptors=[
            DescriptorInfo("HSP_Dispersion", "COSMO-RS", "ハンセン溶解度パラメータ(HSP) 分散力項", "Hansen Solubility Parameters (1967)", "熱力学・相互作用系"),
            DescriptorInfo("HSP_Polar", "COSMO-RS", "ハンセン溶解度パラメータ(HSP) 極性項", "Hansen Solubility Parameters / COSMO-RS", "極性・官能基系"),
            DescriptorInfo("HSP_Hbond", "COSMO-RS", "ハンセン溶解度パラメータ(HSP) 水素結合項", "Hansen Solubility Parameters / COSMO-RS", "極性・官能基系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Hildebrand Solubility Parameter", "熱力学・相互作用系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（極性相互作用）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("MolLogP", "RDKit", "オクタノール/水分配係数", "RDKit", "熱力学・相互作用系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積", "RDKit", "極性・官能基系"),
            DescriptorInfo("MolVolume", "COSMO-RS", "モル体積（Flory-Huggins理論におけるエントロピー項）", "Flory-Huggins solution theory", "立体・形状系"),
        ]
    ),
    TargetRecommendations(
        target_name="アッベ数 (Abbe Number)",
        summary="色分散の小ささを示す。分極率の波長依存性（異常分散）に関係し、電子分極が小さくπ電子系が少ないほどアッべ数が高い。屈折率と逆相関の傾向。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("MolMR", "RDKit", "モル屈折（分極率の指標、アッベ数と負の相関）", "Lorentz-Lorenz", "量子化学・電子状態系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（π電子→異常分散→低アッベ数）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率（高いほど分散低→高アッベ数）", "Polymer Optics", "トポロジー系"),
            DescriptorInfo("Polarizability", "XTB", "分極率", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("HomoLumoGap", "XTB", "HOMO-LUMOギャップ（大きいほど分散が少ない）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "General", "立体・形状系"),
            DescriptorInfo("NumHalogens", "RDKit", "ハロゲン原子数（重原子は分散を増大）", "RDKit", "極性・官能基系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "ファンデルワールス体積", "McGowan", "立体・形状系"),
        ]
    ),
    TargetRecommendations(
        target_name="融点 (Tm / Melting Temperature)",
        summary="結晶の秩序が崩壊する温度。格子エネルギー（分子間力の強さ）、分子の対称性、水素結合ネットワーク、分子量が支配的。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("CohesiveEnergy", "COSMO-RS", "凝集エネルギー（格子エネルギーの指標）", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（極性分子間力）", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "Crystal packing theory", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Crystal packing theory", "極性・官能基系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（π-πスタッキング）", "Crystal Engineering", "トポロジー系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "General thermodynamics", "立体・形状系"),
            DescriptorInfo("SpherocityIndex", "RDKit", "球形度（対称性が高いほど高Tm）", "RDKit 3D", "立体・形状系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（柔軟性→低Tm）", "RDKit", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="熱膨張係数 (CTE / Coefficient of Thermal Expansion)",
        summary="温度変化に対する体積膨張率。自由体積の大きさ、分子間力の弱さ、鎖の柔軟性が高いほどCTEが大きい。Tgとは負の相関。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（大きいほどCTE大）", "Bicerano (2002)", "立体・形状系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度（強いほどCTE小）", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（柔軟→高CTE）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移温度（Tg以上でCTE急増）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（剛直→低CTE）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "ファンデルワールス体積", "McGowan", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Bicerano (2002)", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="溶解度 (Solubility / LogS)",
        summary="溶媒への溶けやすさ。ハンセン溶解度パラメータの距離、分子の極性、水素結合能力、分子サイズが決定因子。",
        category="界面・溶液系",
        descriptors=[
            DescriptorInfo("MolLogP", "RDKit", "オクタノール/水分配係数（水溶性と強い負相関）", "ADMET / Yalkowsky", "熱力学・相互作用系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積（大きいほど水溶性）", "Ertl et al. (2000)", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "Lipinski", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Lipinski", "極性・官能基系"),
            DescriptorInfo("AqueousSolubility", "COSMO-RS", "COSMO-RS水溶性予測値", "COSMOTherm", "界面・溶液系"),
            DescriptorInfo("MolWt", "RDKit", "分子量（大きいほど溶解エントロピー小）", "General", "立体・形状系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数", "RDKit", "トポロジー系"),
            DescriptorInfo("SolvationFreeEnergy", "COSMO-RS", "溶媒和自由エネルギー", "COSMO-RS", "熱力学・相互作用系"),
        ]
    ),
    TargetRecommendations(
        target_name="引張強度 (Tensile Strength)",
        summary="引張破壊に至るまでの最大応力。凝集エネルギー密度、水素結合ネットワーク、主鎖の剛直性、結晶化度が影響。",
        category="力学・強度系",
        descriptors=[
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（強い分子間ネットワーク）", "Polymer Mechanics", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Polymer Mechanics", "極性・官能基系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（剛直性と配向結晶化）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（柔軟性）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "Polymer science", "立体・形状系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Bicerano (2002)", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="靭性 (Toughness)",
        summary="破壊までに吸収するエネルギー量。強度と伸びの積。絡み合い分子量、凝集力と柔軟性のバランス、衝撃吸収能力が重要。",
        category="力学・強度系",
        descriptors=[
            DescriptorInfo("EntanglementMW", "GroupContribution", "絡み合い分子量", "Wu (1989) / Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（延性に寄与）", "Polymer Physics", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率（柔軟性）", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移温度（使用温度との差）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（エネルギー吸収）", "Polymer Mechanics", "極性・官能基系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "Polymer science", "立体・形状系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="バンドギャップ (Band Gap)",
        summary="電子のHOMO-LUMOギャップに対応。共役長、電子供与・吸引基のバランス、分子の平面性が支配的。有機半導体・OLEDの設計指標。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("HomoLumoGap", "XTB", "HOMO-LUMOギャップ（バンドギャップの直接近似）", "GFN2-xTB / Koopmans", "量子化学・電子状態系"),
            DescriptorInfo("HomoEnergy", "XTB", "HOMOエネルギー", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("LumoEnergy", "XTB", "LUMOエネルギー", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MaxConjugatedChain", "RDKit", "最大共役長（長い→小ギャップ）", "Physical Organic Chemistry", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（π非局在化）", "Organic Electronics", "トポロジー系"),
            DescriptorInfo("Polarizability", "XTB", "分極率", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("IonizationPotential", "XTB", "イオン化ポテンシャル", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("ElectronAffinity", "XTB", "電子親和力", "GFN2-xTB", "量子化学・電子状態系"),
        ]
    ),
    TargetRecommendations(
        target_name="導電率 (Electrical Conductivity)",
        summary="電荷キャリアの移動度に依存。HOMO-LUMOギャップ（小さいほど導電性）、共役系、ドーパント相互作用、ホッピング経路の有無が支配的。",
        category="光・電磁気系",
        descriptors=[
            DescriptorInfo("HomoLumoGap", "XTB", "HOMO-LUMOギャップ（小さいほど高導電性）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("MaxConjugatedChain", "RDKit", "最大共役長（キャリア移動経路）", "Organic Electronics", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（π軌道の非局在化）", "Organic Electronics", "トポロジー系"),
            DescriptorInfo("ElectronAffinity", "XTB", "電子親和力（n型導電性）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("IonizationPotential", "XTB", "イオン化ポテンシャル（p型導電性）", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("Polarizability", "XTB", "分極率", "GFN2-xTB", "量子化学・電子状態系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率（低いほど共役→高導電）", "Organic Electronics", "トポロジー系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="親水性 / 撥水性 (Hydrophilicity / Hydrophobicity)",
        summary="水との親和性。接触角に対応。LogP、極性表面積、水素結合基の数、表面自由エネルギーが決定因子。",
        category="界面・溶液系",
        descriptors=[
            DescriptorInfo("MolLogP", "RDKit", "オクタノール/水分配係数（疎水性の直接指標）", "Lipinski", "熱力学・相互作用系"),
            DescriptorInfo("TPSA", "RDKit", "極性表面積（大きいほど親水性）", "Ertl et al. (2000)", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数", "Surface chemistry", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Surface chemistry", "極性・官能基系"),
            DescriptorInfo("SolvationFreeEnergy", "COSMO-RS", "水和自由エネルギー", "COSMO-RS", "熱力学・相互作用系"),
            DescriptorInfo("HLB", "GroupContribution", "親水性疎水性バランス(HLB)", "Griffin's method", "熱力学・相互作用系"),
            DescriptorInfo("NumHeteroatoms", "RDKit", "ヘテロ原子数（親水基の数）", "RDKit", "極性・官能基系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Surface chemistry", "トポロジー系"),
        ]
    ),
    TargetRecommendations(
        target_name="酸素透過性 (Oxygen Permeability / OTR)",
        summary="酸素ガスの透過量。自由体積と拡散係数、凝集力と溶解度係数のバランスで決まる。結晶化度が高いほどバリア性が高い。",
        category="輸送・透過性系",
        descriptors=[
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（大きいほど透過性大）", "Bicerano (2002) / Freeman theory", "立体・形状系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度（高いほどバリア性大）", "Freeman (1999)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（鎖可動性→拡散促進）", "Membrane science", "トポロジー系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移温度（Tg以上で透過急増）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "ファンデルワールス体積", "McGowan", "立体・形状系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（パッキング密度→バリア性）", "Membrane science", "トポロジー系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント（O2との相互作用）", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="水蒸気透過性 (Water Vapor Permeability / WVTR)",
        summary="水蒸気の透過量。親水基の多さ（溶解度係数）と自由体積（拡散係数）の積。極性基が多いほど水分の溶解度が上がり透過性増大。",
        category="輸送・透過性系",
        descriptors=[
            DescriptorInfo("TPSA", "RDKit", "極性表面積（水蒸気との親和性）", "Ertl / Membrane science", "極性・官能基系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（水分子の溶解促進）", "Water transport theory", "極性・官能基系"),
            DescriptorInfo("NumHAcceptors", "RDKit", "水素結合受容体数", "Water transport theory", "極性・官能基系"),
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（拡散係数の指標）", "Bicerano (2002)", "立体・形状系"),
            DescriptorInfo("MolLogP", "RDKit", "脂溶性（高いほど水蒸気バリア性）", "ADMET / Membrane science", "熱力学・相互作用系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Freeman (1999)", "熱力学・相互作用系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（パッキング→バリア性）", "Membrane science", "トポロジー系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
        ]
    ),
    TargetRecommendations(
        target_name="水素透過性 (Hydrogen Permeability)",
        summary="水素ガスの透過量。H2は最小分子のため自由体積による拡散が支配的。凝集力や結晶性は溶解度を低下させバリア性を向上。",
        category="輸送・透過性系",
        descriptors=[
            DescriptorInfo("FreeVolume", "GroupContribution", "自由体積（H2拡散に最も支配的）", "Freeman (1999) / Robeson upper bound", "立体・形状系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度（バリア性）", "Freeman (1999)", "熱力学・相互作用系"),
            DescriptorInfo("Tg_estimated", "GroupContribution", "ガラス転移温度（ゴム状態で透過急増）", "Bicerano (2002)", "熱・相転移系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（鎖運動性→拡散促進）", "Membrane science", "トポロジー系"),
            DescriptorInfo("VanDerWaalsVolume", "GroupContribution", "ファンデルワールス体積", "McGowan / Bondi", "立体・形状系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数", "Membrane science", "トポロジー系"),
            DescriptorInfo("MolWt", "RDKit", "分子量", "General", "立体・形状系"),
        ]
    ),
    TargetRecommendations(
        target_name="溶融粘度 (Melt Viscosity)",
        summary="溶融状態での流れにくさ。分子量（絡み合い）、分子間力、鎖の剛直性が支配的。加工性に直結。",
        category="熱・相転移系",
        descriptors=[
            DescriptorInfo("MolWt", "RDKit", "分子量（絡み合い分子量超で粘度急増）", "Rouse-Zimm / Reptation theory", "立体・形状系"),
            DescriptorInfo("EntanglementMW", "GroupContribution", "絡み合い分子量", "Wu (1989) / Bicerano (2002)", "トポロジー系"),
            DescriptorInfo("CohesiveEnergyDensity", "GroupContribution", "凝集エネルギー密度", "Bicerano (2002)", "熱力学・相互作用系"),
            DescriptorInfo("NumRotatableBonds", "RDKit", "回転可能結合数（柔軟→低粘度）", "Polymer Rheology", "トポロジー系"),
            DescriptorInfo("NumAromaticRings", "RDKit", "芳香環数（剛直→高粘度）", "Polymer Rheology", "トポロジー系"),
            DescriptorInfo("NumHDonors", "RDKit", "水素結合供与体数（分子間拘束）", "Polymer Rheology", "極性・官能基系"),
            DescriptorInfo("DipoleMoment", "XTB", "双極子モーメント", "GFN2-xTB", "極性・官能基系"),
            DescriptorInfo("FractionCSP3", "RDKit", "sp3炭素比率", "Polymer Physics", "トポロジー系"),
        ]
    ),
]

def get_all_target_recommendations() -> List[TargetRecommendations]:
    """登録されているすべての目的変数推奨セットのリストを返す"""
    return list(_RECOMMENDATION_DATA)

def get_target_recommendation_by_name(name: str) -> TargetRecommendations | None:
    """指定された名前のターゲット変数の推奨情報を取得する（部分一致対応）"""
    for rec in _RECOMMENDATION_DATA:
        if name.lower() in rec.target_name.lower():
            return rec
    return None

def get_target_names() -> List[str]:
    """登録されているすべての目的変数名(表示用)のリストを返す"""
    return [rec.target_name for rec in _RECOMMENDATION_DATA]

def get_target_categories() -> List[str]:
    """登録されている目的変数のカテゴリ（系統）のユニークなリストを返す"""
    categories = [rec.category for rec in _RECOMMENDATION_DATA]
    # 順序を保持したまま重複排除
    return list(dict.fromkeys(categories))

def get_targets_by_category(category: str) -> List[TargetRecommendations]:
    """指定したカテゴリに属する目的変数のリストを返す"""
    return [rec for rec in _RECOMMENDATION_DATA if rec.category == category]

def get_all_descriptor_categories() -> List[str]:
    """全ての説明変数カテゴリ（物理的意味の分類）のユニークなリストを返す"""
    categories = []
    for rec in _RECOMMENDATION_DATA:
        for desc in rec.descriptors:
            if desc.category not in categories:
                categories.append(desc.category)
    return categories


def get_descriptors_by_category(category: str) -> List[DescriptorInfo]:
    """指定した意味カテゴリに属する説明変数のリストを重複なく返す"""
    seen_names = set()
    descriptors = []
    for rec in _RECOMMENDATION_DATA:
        for desc in rec.descriptors:
            if desc.category == category and desc.name not in seen_names:
                seen_names.add(desc.name)
                descriptors.append(desc)
    return descriptors


# ── LLMベース推奨機能 ───────────────────────────────────────────────────────────
# 4000+ の全特徴量からLLMを使って目的変数に適したものを選択する。

_FEARTURE_WEIGHTING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "chemai2_4000_manual_weighting.json"
)


def _load_all_descriptors() -> Dict[str, Dict]:
    """chemai2_4000_manual_weighting.json から全特徴量を読み込む。"""
    path = os.path.normpath(_FEARTURE_WEIGHTING_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _categorize_descriptors(desc_dict: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    全特徴量をカテゴリごとに分類する。
    大きなグループはLLMトークン制限を考慮してサブグループに分割する。
    戻り値: {カテゴリ名: [特徴量名のリスト]} の辞書。
    """
    categories: Dict[str, List[str]] = {}

    for name in desc_dict.keys():
        lower = name.lower()

        # Fingerprint系: 2048個 — サブグループに分割
        if lower.startswith("fingerprint"):
            # ビット位置で分割（1-512, 513-1024, 1025-1536, 1537-2048）
            m = re.search(r'(\d+)', name)
            if m:
                idx = int(m.group(1))
                if idx <= 512:
                    key = "Fingerprint_1_512"
                elif idx <= 1024:
                    key = "Fingerprint_513_1024"
                elif idx <= 1536:
                    key = "Fingerprint_1025_1536"
                else:
                    key = "Fingerprint_1537_2048"
            else:
                key = "Fingerprint_Other"
            categories.setdefault(key, []).append(name)

        # Quantum Chemical系: Orbital (1001) + Mulliken (1001) = 2002個
        elif lower.startswith("orbital"):
            # Orbital特徴量をサブグループに分割
            m = re.search(r'(\d+)', name)
            if m:
                idx = int(m.group(1))
                if idx <= 334:
                    key = "Quantum_Orbital_1_334"
                elif idx <= 667:
                    key = "Quantum_Orbital_335_667"
                else:
                    key = "Quantum_Orbital_668_1001"
            else:
                key = "Quantum_Orbital_Other"
            categories.setdefault(key, []).append(name)

        elif lower.startswith("mulliken"):
            # Mulliken特徴量をサブグループに分割
            m = re.search(r'(\d+)', name)
            if m:
                idx = int(m.group(1))
                if idx <= 334:
                    key = "Quantum_Mulliken_1_334"
                elif idx <= 667:
                    key = "Quantum_Mulliken_335_667"
                else:
                    key = "Quantum_Mulliken_668_1001"
            else:
                key = "Quantum_Mulliken_Other"
            categories.setdefault(key, []).append(name)

        # Vibrational系: 999個 — サブグループに分割
        elif lower.startswith("freq"):
            m = re.search(r'(\d+)', name)
            if m:
                idx = int(m.group(1))
                if idx <= 333:
                    key = "Vibrational_1_333"
                elif idx <= 666:
                    key = "Vibrational_334_666"
                else:
                    key = "Vibrational_667_999"
            else:
                key = "Vibrational_Other"
            categories.setdefault(key, []).append(name)

        # RDKit 2D系
        elif any(lower.startswith(p) for p in ["rdkit_", "molwt", "logp", "tpsa", "num_", "frac_", "heavy"]):
            categories.setdefault("RDKit_2D", []).append(name)

        # Mordred系
        elif any(lower.startswith(p) for p in ["mordred_", "abc", "ats", "bcu", "c2d", "c3d", "charge", "eig", "grav", "jgt", "k_", "m_", "md", "ml", "mor", "mp_", "mw_", "p_", "r_", "s_", "t_", "v_", "w_", "z_"]):
            categories.setdefault("Mordred", []).append(name)

        # Physical/Chemical系
        elif any(x in lower for x in ["octanol", "surface", "viscosity", "refractive", "dielectric", "flash", "boiling", "melting", "vapor", "henry", "bioaccumulation", "toxicity", "mutagenicity", "carcinogenicity", "endocrine", "biodegradation"]):
            categories.setdefault("Physical_Chemical", []).append(name)

        else:
            categories.setdefault("Other", []).append(name)

    return categories


def _chunk_list(lst: List[str], chunk_size: int = 200) -> List[List[str]]:
    """リストを指定サイズのチャンクに分割（LLMトークン制限対策）。"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


_CATEGORY_JA = {
    # Fingerprint (subdivided)
    "Fingerprint_1_512": "フィンガープリント（ビット1-512、分子構造の部分的ハッシュ表現）",
    "Fingerprint_513_1024": "フィンガープリント（ビット513-1024）",
    "Fingerprint_1025_1536": "フィンガープリント（ビット1025-1536）",
    "Fingerprint_1537_2048": "フィンガープリント（ビット1537-2048）",
    "Fingerprint_Other": "フィンガープリント（その他）",
    # Quantum Chemical (subdivided)
    "Quantum_Orbital_1_334": "Orbital特徴量（ビット1-334、量子化学計算の軌道エネルギー等）",
    "Quantum_Orbital_335_667": "Orbital特徴量（ビット335-667）",
    "Quantum_Orbital_668_1001": "Orbital特徴量（ビット668-1001）",
    "Quantum_Orbital_Other": "Orbital特徴量（その他）",
    "Quantum_Mulliken_1_334": "Mulliken特徴量（ビット1-334、部分電荷・電子状態等）",
    "Quantum_Mulliken_335_667": "Mulliken特徴量（ビット335-667）",
    "Quantum_Mulliken_668_1001": "Mulliken特徴量（ビット668-1001）",
    "Quantum_Mulliken_Other": "Mulliken特徴量（その他）",
    # Vibrational (subdivided)
    "Vibrational_1_333": "振動モード特徴量（モード1-333、赤外・ラマン分光）",
    "Vibrational_334_666": "振動モード特徴量（モード334-666）",
    "Vibrational_667_999": "振動モード特徴量（モード667-999）",
    "Vibrational_Other": "振動モード特徴量（その他）",
    # Others
    "RDKit_2D": "RDKit 2次元記述子（分子量、LogP、TPSAなどの基本記述子）",
    "Mordred": "Mordred記述子（多様な構造記述子の集合）",
    "Physical_Chemical": "物理化学特性（粘度、屈折率、誘電率などの実測値予測用）",
    "Other": "その他の特徴量",
}

_TARGET_CATEGORY_GUIDE = {
    "屈折率 (Refractive Index)": "光・電磁気系",
    "吸収率 (Absorption / UV-Vis)": "光・電磁気系",
    "シャルピー強度 (Charpy Impact Strength)": "力学・強度系",
    "破壊強度 (Fracture Strength)": "力学・強度系",
    "弾性率 (Elastic Modulus)": "力学・強度系",
    "Tg (ガラス転移点)": "熱・相転移系",
    "pKa (酸解離定数)": "界面・溶液系",
    "毒性 (Toxicity)": "環境・安全性",
    "粘度 (Viscosity)": "熱・相転移系",
    "密度 (Density)": "熱力学・相転移系",
    "誘電率 (Dielectric Constant)": "光・電磁気系",
    "誘電正接 (Dissipation Factor / Tan Delta)": "光・電磁気系",
    "耐湿性 / 吸水率 (Moisture Resistance / Water Absorption)": "環境・安全性",
    "耐光性 (Light Resistance / UV Stability)": "環境・安全性",
    "耐熱性 (Heat Resistance / Thermal Decomposition, Td)": "熱・相転移系",
    "相溶性 (Compatibility / Miscibility)": "界面・溶液系",
    "アッベ数 (Abbe Number)": "光・電磁気系",
    "融点 (Tm / Melting Temperature)": "熱・相転移系",
    "熱膨張係数 (CTE / Coefficient of Thermal Expansion)": "熱・相転移系",
    "溶解度 (Solubility / LogS)": "界面・溶液系",
    "引張強度 (Tensile Strength)": "力学・強度系",
    "靭性 (Toughness)": "力学・強度系",
    "バンドギャップ (Band Gap)": "光・電磁気系",
    "導電率 (Electrical Conductivity)": "光・電磁気系",
    "親水性 / 撥水性 (Hydrophilicity / Hydrophobicity)": "界面・溶液系",
    "酸素透過性 (Oxygen Permeability / OTR)": "輸送・透過性系",
    "水蒸気透過性 (Water Vapor Permeability / WVTR)": "輸送・透過性系",
    "水素透過性 (Hydrogen Permeability)": "輸送・透過性系",
    "溶融粘度 (Melt Viscosity)": "熱・相転移系",
}


def _build_llm_prompt_for_category(
    target_name: str,
    category: str,
    descriptors: List[str],
    max_select: int = 30,
) -> str:
    """
    1カテゴリに含まれる特徴量リストからLLMに選択させるプロンプトを構築する。
    トークン数節約のため、特徴量名のみを提示し、化学的知見に基づいた選択を依頼する。
    """
    category_desc = _CATEGORY_JA.get(category, category)
    target_category = _TARGET_CATEGORY_GUIDE.get(target_name, "不明")

    # 特徴量リストを整形（トークン節約のため、番号付きリストで簡潔に）
    desc_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(descriptors))

    prompt = f"""\
あなたは化学情報学・QSAR/QSPRの専門家です。

## 課題
目的変数「{target_name}」（カテゴリ: {target_category}）を予測する機械学習モデルを構築します。
以下の「{category}」に含まれる特徴量（記述子）の中から、この目的変数の予測に**最も有用な上位{max_select}個**を選択してください。

## カテゴリの説明
{category_desc}

## 特徴量リスト（{len(descriptors)}個）
{desc_list}

## 選択の指針
1. **化学的因果関係**: その特徴量が目的変数に物理的・化学的にどう寄与するかを考慮する
2. **情報の重複回避**: 似たような意味を持つ特徴量を複数選ばない（例: 複数の分子量変数）
3. **カテゴリの特性**:
   - フィンガープリント: 構造キーであるため、重要なのは「使用する」か「しないか」。使用する場合は "fingerprint" として1つにまとめる
   - 量子化学: HOMO、LUMO、ギャップ、分極率、双極子モーメントなど電子状態が重要
   - 振動モード: 目的変数が熱・相転移・力学系の場合に関連するモードを選ぶ
   - RDKit・Mordred: トポロジー、表面積、水素結合などの基本記述子
   - 物理化学: 目的変数が当該特性自体の場合は積極的に選ぶ
4. **最大{max_select}個**まで選択可能（少なくてもよい）

## 出力形式
以下の形式で、選択した特徴量名をカンマ区切りで出力してください。説明や理由は不要です。

選択特徴量: 特徴量名1, 特徴量名2, 特徴量名3, ...

※フィンガープリント系は "fingerprint" とだけ書いてください（2048個すべてを列挙しない）。
"""
    return prompt


def _parse_llm_response(response_text: str) -> List[str]:
    """
    LLMのレスポンスから特徴量名のリストを抽出する。
    「選択特徴量: xxx, yyy, zzz」の形式をパースする。
    """
    # "選択特徴量:" の後のテキストを抽出
    match = re.search(r'選択特徴量[:：]\s*(.+)', response_text, re.IGNORECASE)
    if not match:
        # カンマ区切りのリストがそのまま書かれている場合のフォールバック
        text = response_text.strip()
    else:
        text = match.group(1).strip()

    # カンマ、改行、スペースで分割
    names = re.split(r'[,、\n\r]+', text)
    result = []
    for name in names:
        name = name.strip().strip('"\'[]{}').strip()
        if name and not name.startswith('選択') and not name.startswith('特徴量'):
            result.append(name)
    return result


def get_llm_recommendations(
    target_name: str,
    provider_name: str = "stub",
    max_per_chunk: int = 10,
    chunk_size: int = 150,
    include_rule_based: bool = True,
) -> Dict:
    """
    LLMを使って4000+特徴量から目的変数に適したものを選択する。
    大きなカテゴリはチャンク分割して処理する。

    Args:
        target_name: 目的変数名（例: "屈折率", "Tg"）
        provider_name: LLMプロバイダー名（"stub", "openai", "anthropic"など）
        max_per_chunk: 1チャンクあたりの最大選択数
        chunk_size: 1チャンクの特徴量数（トークン制限対策）
        include_rule_based: 既存のルールベース推奨結果も含めるか

    Returns:
        {
            "target_name": str,
            "llm_selected": [{"name": str, "category": str}],
            "rule_based": [DescriptorInfoのdict],
            "total_candidates": int,
            "errors": list or None,
        }
    """
    from backend.llm import get_llm_provider
    from backend.llm.provider import LLMRequest, LLMProviderError

    # 1. 全特徴量を読み込んでカテゴリ分け
    all_desc = _load_all_descriptors()
    if not all_desc:
        return {
            "target_name": target_name,
            "llm_selected": [],
            "rule_based": [],
            "total_candidates": 0,
            "error": "特徴量定義ファイル(chemai2_4000_manual_weighting.json)が見つかりません",
        }

    categories = _categorize_descriptors(all_desc)

    # 2. LLMプロバイダーを取得
    try:
        provider = get_llm_provider(provider_name)
    except Exception:
        from backend.llm.provider import StubLLMProvider
        provider = StubLLMProvider()

    # スタブプロバイダーの場合は選択モードに設定
    if hasattr(provider, 'set_mode'):
        provider.set_mode('select')

    # 3. 各カテゴリをチャンク分割してLLMで選択
    selected: List[Dict] = []
    errors: List[str] = []

    for cat_name, desc_list in categories.items():
        if not desc_list:
            continue

        # チャンク分割
        chunks = _chunk_list(desc_list, chunk_size)
        if len(chunks) > 1:
            # 大きなカテゴリの場合は最初の数チャンクのみ処理（全体の代表性を考慮）
            # 全チャンク処理するとAPIコストがかさむため、最大5チャンクまで
            chunks = chunks[:5]

        for chunk_idx, chunk in enumerate(chunks):
            prompt = _build_llm_prompt_for_category(
                target_name=target_name,
                category=f"{cat_name} (part {chunk_idx + 1}/{len(chunks)})",
                descriptors=chunk,
                max_select=max_per_chunk,
            )

            try:
                request = LLMRequest(
                    user_prompt=prompt,
                    system_prompt='あなたは化学情報学・QSAR/QSPRの専門家です。簡潔に特徴量名を出力してください。',
                    max_tokens=512,
                    temperature=0.1,
                )
                response = provider.generate(request)
                names = _parse_llm_response(response.content)

                for n in names:
                    n_lower = n.lower()
                    if n_lower == "fingerprint" or "fingerprint" in n_lower:
                        # フィンガープリントは該当チャンク内の全てを追加
                        selected.extend(
                            {"name": d, "category": cat_name}
                            for d in chunk
                            if d.lower().startswith("fingerprint")
                        )
                    else:
                        # 名前一致（厳密・部分）
                        matches = [
                            d for d in chunk
                            if d.lower() == n_lower or n_lower in d.lower() or d.lower().startswith(n_lower[:20])
                        ]
                        if matches:
                            selected.append({"name": matches[0], "category": cat_name})

            except Exception as e:
                errors.append(f"{cat_name}(chunk {chunk_idx + 1}): {e}")

    # 重複除去
    seen = set()
    unique_selected = []
    for item in selected:
        if item["name"] not in seen:
            seen.add(item["name"])
            unique_selected.append(item)

    # 4. ルールベースの結果も取得
    rule_based = []
    if include_rule_based:
        rec = get_target_recommendation_by_name(target_name)
        if rec:
            rule_based = [
                {
                    "name": d.name,
                    "library": d.library,
                    "meaning": d.meaning,
                    "source": d.source,
                    "category": d.category,
                }
                for d in rec.descriptors
            ]

    return {
        "target_name": target_name,
        "llm_selected": unique_selected,
        "rule_based": rule_based,
        "total_candidates": len(all_desc),
        "errors": errors if errors else None,
    }


def get_combined_recommendations(
    target_name: str,
    provider_name: str = "stub",
    llm_weight: float = 0.7,
) -> Dict:
    """
    LLM推奨とルールベース推奨を統合した最終推奨リストを返す。

    Args:
        target_name: 目的変数名
        provider_name: LLMプロバイダー名
        llm_weight: LLM推奨の重み（0.0-1.0、残りはルールベース）

    Returns:
        {
            "target_name": str,
            "descriptors": [{"name": str, "source": "llm"|"rule", "category": str}],
            "summary": str,
            "llm_selected_count": int,
            "rule_based_count": int,
        }
    """
    result = get_llm_recommendations(target_name, provider_name, include_rule_based=True)

    # LLM選択結果とルールベース結果を統合
    combined = []
    seen = set()

    # LLM選択を優先（重みが高い）
    llm_items = result.get("llm_selected", [])
    rule_items = result.get("rule_based", [])
    llm_count = int(len(llm_items) * llm_weight)

    for item in llm_items[:llm_count]:
        name = item.get("name", "")
        if name and name not in seen:
            seen.add(name)
            combined.append({
                "name": name,
                "source": "llm",
                "category": item.get("category", "不明"),
            })

    # ルールベースを追加
    for item in rule_items:
        name = item.get("name", "")
        if name and name not in seen:
            seen.add(name)
            combined.append({
                "name": name,
                "source": "rule",
                "category": item.get("category", "不明"),
                "meaning": item.get("meaning", ""),
                "library": item.get("library", ""),
            })

    # 残りのLLM選択を追加
    for item in llm_items[llm_count:]:
        name = item.get("name", "")
        if name and name not in seen:
            seen.add(name)
            combined.append({
                "name": name,
                "source": "llm",
                "category": item.get("category", "不明"),
            })

    # 要約
    llm_rec = get_target_recommendation_by_name(target_name)
    summary = llm_rec.summary if llm_rec else "LLMによる特徴量選択"

    return {
        "target_name": target_name,
        "descriptors": combined,
        "llm_selected_count": len(llm_items),
        "rule_based_count": len(rule_items),
        "summary": summary,
        "errors": result.get("errors"),
    }

