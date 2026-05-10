"""
frontend_nicegui/components/descriptor_catalog.py

全SMILES記述子エンジンの記述子をカテゴリ別にグループ分けし、
化学的意味（10-20文字）を付与したカタログモジュール。

各アダプタのソースコードを精読し、記述子の化学的本質を理解した上で
説明文を作成している。ライブラリの説明の単純コピーではない。
"""
from __future__ import annotations


def get_rdkit_catalog() -> dict[str, list[dict]]:
    """RDKit記述子をカテゴリ別に返す。ソース: rdkit_adapter.py _DESCRIPTOR_JP_META"""
    return {
        "分子サイズ・形状": [
            {"name": "MolWt", "short": "分子全体の質量(Da)", "cat": "分子サイズ"},
            {"name": "HeavyAtomMolWt", "short": "水素除外の骨格質量", "cat": "分子サイズ"},
            {"name": "ExactMolWt", "short": "同位体考慮の精密質量", "cat": "分子サイズ"},
            {"name": "HeavyAtomCount", "short": "水素以外の原子の総数", "cat": "分子サイズ"},
            {"name": "NumAtoms", "short": "水素含む全原子の数", "cat": "分子サイズ"},
            {"name": "LabuteASA", "short": "溶媒が接触できる表面積", "cat": "分子サイズ"},
        ],
        "極性・溶解性": [
            {"name": "MolLogP", "short": "油水間の溶けやすさ比率", "cat": "極性・溶解性"},
            {"name": "TPSA", "short": "N,O由来の極性面の広さ", "cat": "極性・溶解性"},
            {"name": "MolMR", "short": "光を曲げる力＝分極率", "cat": "極性・溶解性"},
        ],
        "電子状態": [
            {"name": "MaxPartialCharge", "short": "最も正に偏った原子電荷", "cat": "電子状態"},
            {"name": "MinPartialCharge", "short": "最も負に偏った原子電荷", "cat": "電子状態"},
            {"name": "MaxAbsPartialCharge", "short": "電荷偏りが最大の原子", "cat": "電子状態"},
            {"name": "MinAbsPartialCharge", "short": "電荷が最も均一な原子", "cat": "電子状態"},
            {"name": "NumRadicalElectrons", "short": "不対電子の数＝反応性", "cat": "電子状態"},
            {"name": "NumValenceElectrons", "short": "結合に参加できる電子数", "cat": "電子状態"},
        ],
        "水素結合": [
            {"name": "NumHAcceptors", "short": "H結合を受ける部位の数", "cat": "水素結合"},
            {"name": "NumHDonors", "short": "H結合を与える部位の数", "cat": "水素結合"},
            {"name": "NHOHCount", "short": "OH/NHの水素結合部位数", "cat": "水素結合"},
            {"name": "NOCount", "short": "窒素と酸素の合計原子数", "cat": "水素結合"},
        ],
        "トポロジー（環・結合）": [
            {"name": "NumRotatableBonds", "short": "自由回転できる結合の数", "cat": "トポロジー"},
            {"name": "RingCount", "short": "分子中の環構造の総数", "cat": "トポロジー"},
            {"name": "NumAromaticRings", "short": "π共役した芳香環の数", "cat": "トポロジー"},
            {"name": "NumAliphaticRings", "short": "芳香族でない環の総数", "cat": "トポロジー"},
            {"name": "NumSaturatedRings", "short": "二重結合のない環の数", "cat": "トポロジー"},
            {"name": "NumHeterocycles", "short": "N,O,S含む環の総数", "cat": "トポロジー"},
            {"name": "NumAromaticHeterocycles", "short": "芳香族性のヘテロ環数", "cat": "トポロジー"},
            {"name": "NumAromaticCarbocycles", "short": "ベンゼン型の炭素環数", "cat": "トポロジー"},
            {"name": "NumAliphaticHeterocycles", "short": "非芳香族のヘテロ環数", "cat": "トポロジー"},
            {"name": "NumAliphaticCarbocycles", "short": "非芳香族の炭素環数", "cat": "トポロジー"},
            {"name": "NumSaturatedHeterocycles", "short": "飽和のヘテロ環の数", "cat": "トポロジー"},
            {"name": "NumSaturatedCarbocycles", "short": "飽和の炭素環の数", "cat": "トポロジー"},
            {"name": "FractionCSP3", "short": "sp3炭素の割合＝立体性", "cat": "トポロジー"},
            {"name": "NumAmideBonds", "short": "ペプチド型アミド結合数", "cat": "トポロジー"},
            {"name": "NumBridgeheadAtoms", "short": "二環式の橋頭位原子数", "cat": "トポロジー"},
            {"name": "NumSpiroAtoms", "short": "環が1原子共有する箇所", "cat": "トポロジー"},
        ],
        "位相的指標（グラフ理論）": [
            {"name": "BalabanJ", "short": "分子グラフの対称均一性", "cat": "位相的指標"},
            {"name": "BertzCT", "short": "構造の分岐・環の複雑度", "cat": "位相的指標"},
            {"name": "HallKierAlpha", "short": "原子サイズ補正の形状値", "cat": "位相的指標"},
            {"name": "Kappa1", "short": "直鎖らしさの形状指標", "cat": "位相的指標"},
            {"name": "Kappa2", "short": "分岐の多さの形状指標", "cat": "位相的指標"},
            {"name": "Kappa3", "short": "空間的広がりの形状指標", "cat": "位相的指標"},
            {"name": "Chi0", "short": "原子種の多様性(0次)", "cat": "位相的指標"},
            {"name": "Chi0n", "short": "原子価考慮の0次接続性", "cat": "位相的指標"},
            {"name": "Chi0v", "short": "VdW版の0次接続性指数", "cat": "位相的指標"},
            {"name": "Chi1", "short": "隣接原子の結合パターン", "cat": "位相的指標"},
            {"name": "Chi1n", "short": "原子価考慮の1次接続性", "cat": "位相的指標"},
            {"name": "Chi1v", "short": "VdW版の1次接続性指数", "cat": "位相的指標"},
            {"name": "Chi2n", "short": "2結合先の経路パターン", "cat": "位相的指標"},
            {"name": "Chi2v", "short": "VdW版の2次接続性指数", "cat": "位相的指標"},
            {"name": "Chi3n", "short": "3結合先の分岐パターン", "cat": "位相的指標"},
            {"name": "Chi3v", "short": "VdW版の3次接続性指数", "cat": "位相的指標"},
            {"name": "Chi4n", "short": "4結合先の経路パターン", "cat": "位相的指標"},
            {"name": "Chi4v", "short": "VdW版の4次接続性指数", "cat": "位相的指標"},
            {"name": "Ipc", "short": "分子グラフの情報エントロピー", "cat": "位相的指標"},
            {"name": "AvgIpc", "short": "原子あたり平均情報含量", "cat": "位相的指標"},
        ],
        "BCUT（空間分布パターン）": [
            {"name": "BCUT2D_MWHI", "short": "原子量分布の最大偏り", "cat": "BCUT"},
            {"name": "BCUT2D_MWLOW", "short": "原子量分布の最小偏り", "cat": "BCUT"},
            {"name": "BCUT2D_CHGHI", "short": "電荷分布の最も偏る方向", "cat": "BCUT"},
            {"name": "BCUT2D_CHGLO", "short": "電荷分布の最も均一方向", "cat": "BCUT"},
            {"name": "BCUT2D_LOGPHI", "short": "疎水性の空間的偏り最大", "cat": "BCUT"},
            {"name": "BCUT2D_LOGPLOW", "short": "疎水性の空間的偏り最小", "cat": "BCUT"},
            {"name": "BCUT2D_MRHI", "short": "分極率の空間偏り最大", "cat": "BCUT"},
            {"name": "BCUT2D_MRLOW", "short": "分極率の空間偏り最小", "cat": "BCUT"},
        ],
        "EState（電気化学的環境）": [
            {"name": "MaxAbsEStateIndex", "short": "電荷影響力が最大の原子", "cat": "EState"},
            {"name": "MaxEStateIndex", "short": "電子受容しやすい原子値", "cat": "EState"},
            {"name": "MinAbsEStateIndex", "short": "電荷的に最も中立な原子", "cat": "EState"},
            {"name": "MinEStateIndex", "short": "電子供与しやすい原子値", "cat": "EState"},
        ],
        "表面積分布（VSA系）": [
            # PEOE_VSA: 部分電荷別
            {"name": "PEOE_VSA1", "short": "強い負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA2", "short": "やや負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA3", "short": "弱い負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA4", "short": "微弱な負電荷原子の面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA5", "short": "ほぼ中性の原子の表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA6", "short": "わずかに負の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA7", "short": "中性帯の原子の表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA8", "short": "微弱な正電荷原子の面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA9", "short": "弱い正電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA10", "short": "やや正電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA11", "short": "正電荷帯の原子の表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA12", "short": "強い正電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA13", "short": "かなり正電荷原子の面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA14", "short": "最も正電荷の原子表面積", "cat": "表面積分布"},
            # SMR_VSA: 屈折率別
            {"name": "SMR_VSA1", "short": "低分極率原子の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA2", "short": "やや低分極率原子の面積", "cat": "表面積分布"},
            {"name": "SMR_VSA3", "short": "中程度分極率原子の面積", "cat": "表面積分布"},
            {"name": "SMR_VSA4", "short": "やや高分極率原子の面積", "cat": "表面積分布"},
            {"name": "SMR_VSA5", "short": "高分極率の原子の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA6", "short": "非常に高分極率原子面積", "cat": "表面積分布"},
            {"name": "SMR_VSA7", "short": "分極率最大の原子表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA8", "short": "極高分極率域の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA9", "short": "超高分極率域の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA10", "short": "最大分極率域の表面積", "cat": "表面積分布"},
            # SlogP_VSA: 疎水性別
            {"name": "SlogP_VSA1", "short": "最も親水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA2", "short": "親水的な原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA3", "short": "やや親水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA4", "short": "中性付近の原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA5", "short": "やや疎水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA6", "short": "疎水的な原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA7", "short": "強い疎水性原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA8", "short": "非常に疎水的原子の面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA9", "short": "極めて疎水的原子の面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA10", "short": "最も疎水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA11", "short": "超疎水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA12", "short": "疎水性極大域の表面積", "cat": "表面積分布"},
            # EState_VSA
            {"name": "EState_VSA1", "short": "電子リッチ原子の表面積", "cat": "表面積分布"},
            {"name": "EState_VSA2", "short": "やや電子リッチ原子面積", "cat": "表面積分布"},
            {"name": "EState_VSA3", "short": "弱電子リッチ原子の面積", "cat": "表面積分布"},
            {"name": "EState_VSA4", "short": "電子的中性原子の表面積", "cat": "表面積分布"},
            {"name": "EState_VSA5", "short": "弱電子プア原子の表面積", "cat": "表面積分布"},
            {"name": "EState_VSA6", "short": "電子プア原子の表面積", "cat": "表面積分布"},
            {"name": "EState_VSA7", "short": "かなり電子プア原子面積", "cat": "表面積分布"},
            {"name": "EState_VSA8", "short": "強い電子プア原子の面積", "cat": "表面積分布"},
            {"name": "EState_VSA9", "short": "非常に電子プア原子面積", "cat": "表面積分布"},
            {"name": "EState_VSA10", "short": "最も電子プア原子の面積", "cat": "表面積分布"},
            {"name": "EState_VSA11", "short": "極度電子プア域の表面積", "cat": "表面積分布"},
            # VSA_EState
            {"name": "VSA_EState1", "short": "面積加重EState区間1", "cat": "表面積分布"},
            {"name": "VSA_EState2", "short": "面積加重EState区間2", "cat": "表面積分布"},
            {"name": "VSA_EState3", "short": "面積加重EState区間3", "cat": "表面積分布"},
            {"name": "VSA_EState4", "short": "面積加重EState区間4", "cat": "表面積分布"},
            {"name": "VSA_EState5", "short": "面積加重EState区間5", "cat": "表面積分布"},
            {"name": "VSA_EState6", "short": "面積加重EState区間6", "cat": "表面積分布"},
            {"name": "VSA_EState7", "short": "面積加重EState区間7", "cat": "表面積分布"},
            {"name": "VSA_EState8", "short": "面積加重EState区間8", "cat": "表面積分布"},
            {"name": "VSA_EState9", "short": "面積加重EState区間9", "cat": "表面積分布"},
            {"name": "VSA_EState10", "short": "面積加重EState区間10", "cat": "表面積分布"},
        ],
        "薬品適性": [
            {"name": "qed", "short": "薬品としての望ましさ(0-1)", "cat": "薬品適性"},
            {"name": "SPS", "short": "合成容易性考慮の薬品性", "cat": "薬品適性"},
        ],
        "FP密度": [
            {"name": "FpDensityMorgan1", "short": "半径1の局所環境の密度", "cat": "FP密度"},
            {"name": "FpDensityMorgan2", "short": "半径2の中距離環境密度", "cat": "FP密度"},
            {"name": "FpDensityMorgan3", "short": "半径3の広域環境の密度", "cat": "FP密度"},
        ],
        "Gasteiger電荷": [
            {"name": "gasteiger_q_max", "short": "Gasteiger法の最大電荷", "cat": "Gasteiger電荷"},
            {"name": "gasteiger_q_min", "short": "Gasteiger法の最小電荷", "cat": "Gasteiger電荷"},
            {"name": "gasteiger_q_range", "short": "電荷の最大-最小の幅", "cat": "Gasteiger電荷"},
            {"name": "gasteiger_q_std", "short": "部分電荷のばらつき度", "cat": "Gasteiger電荷"},
            {"name": "gasteiger_q_abs_mean", "short": "電荷絶対値の平均偏り", "cat": "Gasteiger電荷"},
        ],
        "フィンガープリント（RDKit内蔵）": [
            {"name": "_fp_morgan", "short": "円形環境FP(ECFP4) 2048bit [数え上げ系対応]", "cat": "FP",
             "desc": "部分構造をビットにハッシュ。数え上げ系（カウント）対応。出現順並べ替え可"},
            {"name": "_fp_morgan_count", "short": "Morgan FP（数え上げ系）", "cat": "FP",
             "desc": "部分構造の出現回数をカウント。出現順にビットを並べ、自然言語説明付き"},
            {"name": "_fp_rdkit", "short": "経路ベースFP 2048bit", "cat": "FP"},
            {"name": "_fp_maccs", "short": "部分構造キー 166bit", "cat": "FP"},
        ],
        "官能基カウント（fr_系）": [
            {"name": "fr_Al_COO", "short": "脂肪族カルボン酸の数", "cat": "官能基"},
            {"name": "fr_Al_OH", "short": "脂肪族ヒドロキシルの数", "cat": "官能基"},
            {"name": "fr_Al_OH_noTert", "short": "3級以外の脂肪族OHの数", "cat": "官能基"},
            {"name": "fr_ArN", "short": "芳香族窒素原子の数", "cat": "官能基"},
            {"name": "fr_Ar_COO", "short": "芳香族カルボン酸の数", "cat": "官能基"},
            {"name": "fr_Ar_N", "short": "芳香族アミンの数", "cat": "官能基"},
            {"name": "fr_Ar_NH", "short": "芳香環上のNH基の数", "cat": "官能基"},
            {"name": "fr_Ar_OH", "short": "フェノール性OHの数", "cat": "官能基"},
            {"name": "fr_COO", "short": "カルボン酸(エステル含)数", "cat": "官能基"},
            {"name": "fr_COO2", "short": "カルボン酸基の数", "cat": "官能基"},
            {"name": "fr_C_O", "short": "カルボニル基(C=O)の数", "cat": "官能基"},
            {"name": "fr_C_O_noCOO", "short": "COO以外のカルボニル数", "cat": "官能基"},
            {"name": "fr_C_S", "short": "チオカルボニル(C=S)の数", "cat": "官能基"},
            {"name": "fr_HOCCN", "short": "β-ヒドロキシアミンの数", "cat": "官能基"},
            {"name": "fr_Imine", "short": "イミン(C=N)結合の数", "cat": "官能基"},
            {"name": "fr_NH0", "short": "3級アミン(R3N)の数", "cat": "官能基"},
            {"name": "fr_NH1", "short": "2級アミン(R2NH)の数", "cat": "官能基"},
            {"name": "fr_NH2", "short": "1級アミン(RNH2)の数", "cat": "官能基"},
            {"name": "fr_N_O", "short": "N-酸化物の数", "cat": "官能基"},
            {"name": "fr_Ndealkylation1", "short": "N脱アルキル化部位1の数", "cat": "官能基"},
            {"name": "fr_Ndealkylation2", "short": "N脱アルキル化部位2の数", "cat": "官能基"},
            {"name": "fr_Nhpyrrole", "short": "ピロール型NHの数", "cat": "官能基"},
            {"name": "fr_SH", "short": "チオール(-SH)基の数", "cat": "官能基"},
            {"name": "fr_aldehyde", "short": "アルデヒド(-CHO)の数", "cat": "官能基"},
            {"name": "fr_alkyl_carbamate", "short": "アルキルカーバメート数", "cat": "官能基"},
            {"name": "fr_alkyl_halide", "short": "アルキルハライドの数", "cat": "官能基"},
            {"name": "fr_allylic_oxid", "short": "アリル酸化反応部位の数", "cat": "官能基"},
            {"name": "fr_amide", "short": "アミド結合(-CONH-)数", "cat": "官能基"},
            {"name": "fr_amidine", "short": "アミジン基の数", "cat": "官能基"},
            {"name": "fr_aniline", "short": "アニリン構造の数", "cat": "官能基"},
            {"name": "fr_azide", "short": "アジド(-N3)基の数", "cat": "官能基"},
            {"name": "fr_azo", "short": "アゾ(-N=N-)基の数", "cat": "官能基"},
            {"name": "fr_barbitur", "short": "バルビツール酸骨格の数", "cat": "官能基"},
            {"name": "fr_benzene", "short": "ベンゼン環構造の数", "cat": "官能基"},
            {"name": "fr_benzodiazepine", "short": "ベンゾジアゼピン骨格数", "cat": "官能基"},
            {"name": "fr_bicyclic", "short": "二環式構造の数", "cat": "官能基"},
            {"name": "fr_diazo", "short": "ジアゾ基の数", "cat": "官能基"},
            {"name": "fr_dihydropyridine", "short": "ジヒドロピリジン骨格数", "cat": "官能基"},
            {"name": "fr_epoxide", "short": "エポキシド三員環の数", "cat": "官能基"},
            {"name": "fr_ester", "short": "エステル(-COOR)の数", "cat": "官能基"},
            {"name": "fr_ether", "short": "エーテル(-O-)結合の数", "cat": "官能基"},
            {"name": "fr_furan", "short": "フラン環の数", "cat": "官能基"},
            {"name": "fr_guanido", "short": "グアニジン基の数", "cat": "官能基"},
            {"name": "fr_halogen", "short": "ハロゲン原子の総数", "cat": "官能基"},
            {"name": "fr_hdrzine", "short": "ヒドラジン構造の数", "cat": "官能基"},
            {"name": "fr_hdrzone", "short": "ヒドラゾン構造の数", "cat": "官能基"},
            {"name": "fr_imidazole", "short": "イミダゾール環の数", "cat": "官能基"},
            {"name": "fr_imide", "short": "イミド基の数", "cat": "官能基"},
            {"name": "fr_isocyan", "short": "イソシアネート(-NCO)数", "cat": "官能基"},
            {"name": "fr_isothiocyan", "short": "イソチオシアネート数", "cat": "官能基"},
            {"name": "fr_ketone", "short": "ケトン(R-CO-R')の数", "cat": "官能基"},
            {"name": "fr_ketone_Topliss", "short": "Topliss分類ケトンの数", "cat": "官能基"},
            {"name": "fr_lactam", "short": "ラクタム(環状アミド)数", "cat": "官能基"},
            {"name": "fr_lactone", "short": "ラクトン(環状エステル)数", "cat": "官能基"},
            {"name": "fr_methoxy", "short": "メトキシ(-OCH3)基の数", "cat": "官能基"},
            {"name": "fr_morpholine", "short": "モルホリン環の数", "cat": "官能基"},
            {"name": "fr_nitrile", "short": "ニトリル(-CN)基の数", "cat": "官能基"},
            {"name": "fr_nitro", "short": "ニトロ(-NO2)基の数", "cat": "官能基"},
            {"name": "fr_nitro_arom", "short": "芳香族ニトロ基の数", "cat": "官能基"},
            {"name": "fr_nitro_arom_nonortho", "short": "非オルトの芳香族ニトロ数", "cat": "官能基"},
            {"name": "fr_nitroso", "short": "ニトロソ(-NO)基の数", "cat": "官能基"},
            {"name": "fr_oxazole", "short": "オキサゾール環の数", "cat": "官能基"},
            {"name": "fr_oxime", "short": "オキシム(=NOH)基の数", "cat": "官能基"},
            {"name": "fr_para_hydroxylation", "short": "パラ位水酸化反応部位数", "cat": "官能基"},
            {"name": "fr_phenol", "short": "フェノール性OHの数", "cat": "官能基"},
            {"name": "fr_phenol_noOrthoHbond", "short": "オルトHB無しフェノール数", "cat": "官能基"},
            {"name": "fr_phos_acid", "short": "リン酸基の数", "cat": "官能基"},
            {"name": "fr_phos_ester", "short": "リン酸エステルの数", "cat": "官能基"},
            {"name": "fr_piperdine", "short": "ピペリジン環の数", "cat": "官能基"},
            {"name": "fr_piperzine", "short": "ピペラジン環の数", "cat": "官能基"},
            {"name": "fr_priamide", "short": "1級アミド(-CONH2)数", "cat": "官能基"},
            {"name": "fr_prisulfonamd", "short": "1級スルホンアミドの数", "cat": "官能基"},
            {"name": "fr_pyridine", "short": "ピリジン環の数", "cat": "官能基"},
            {"name": "fr_quatN", "short": "4級窒素(R4N+)の数", "cat": "官能基"},
            {"name": "fr_sulfide", "short": "スルフィド(-S-)の数", "cat": "官能基"},
            {"name": "fr_sulfonamd", "short": "スルホンアミドの数", "cat": "官能基"},
            {"name": "fr_sulfone", "short": "スルホン(-SO2-)の数", "cat": "官能基"},
            {"name": "fr_term_acetylene", "short": "末端アセチレンの数", "cat": "官能基"},
            {"name": "fr_tetrazole", "short": "テトラゾール環の数", "cat": "官能基"},
            {"name": "fr_thiazole", "short": "チアゾール環の数", "cat": "官能基"},
            {"name": "fr_thiocyan", "short": "チオシアネート(-SCN)数", "cat": "官能基"},
            {"name": "fr_thiophene", "short": "チオフェン環の数", "cat": "官能基"},
            {"name": "fr_unbrch_alkane", "short": "非分岐アルカン鎖の数", "cat": "官能基"},
            {"name": "fr_urea", "short": "ウレア(-NHCONH-)の数", "cat": "官能基"},
        ],
    }


def get_xtb_catalog() -> dict[str, list[dict]]:
    """XTB量子化学記述子カタログ。ソース: xtb_adapter.py _XTB_DESCRIPTORS"""
    return {
        "軌道エネルギー": [
            {"name": "xtb_HomoLumoGap", "short": "光吸収・反応性の目安値", "cat": "軌道エネルギー"},
            {"name": "xtb_HomoEnergy", "short": "電子が飛び出す境界準位", "cat": "軌道エネルギー"},
            {"name": "xtb_LumoEnergy", "short": "電子を受け取る空き準位", "cat": "軌道エネルギー"},
            {"name": "xtb_TotalEnergy", "short": "全電子系の安定度(Eh)", "cat": "軌道エネルギー"},
        ],
        "電気的性質": [
            {"name": "xtb_DipoleMoment", "short": "電荷分布の偏り(Debye)", "cat": "電気的性質"},
            {"name": "xtb_Polarizability", "short": "外部電場への変形しやすさ", "cat": "電気的性質"},
        ],
        "反応性指標": [
            {"name": "xtb_IonizationPotential", "short": "電子を奪うのに必要な力", "cat": "反応性指標"},
            {"name": "xtb_ElectronAffinity", "short": "電子を受取る際の安定化", "cat": "反応性指標"},
            {"name": "xtb_Electrophilicity", "short": "求電子剤としての攻撃力", "cat": "反応性指標"},
        ],
        "Mulliken部分電荷統計": [
            {"name": "xtb_MullikenChargeMax", "short": "最も正帯電した原子の値", "cat": "Mulliken電荷"},
            {"name": "xtb_MullikenChargeMin", "short": "最も負帯電した原子の値", "cat": "Mulliken電荷"},
            {"name": "xtb_MullikenChargeMean", "short": "原子電荷の平均的偏り", "cat": "Mulliken電荷"},
            {"name": "xtb_MullikenChargeStd", "short": "原子間の電荷ばらつき", "cat": "Mulliken電荷"},
        ],
    }


def get_group_contrib_catalog() -> dict[str, list[dict]]:
    """原子団寄与法(Joback法)カタログ。ソース: group_contrib_adapter.py"""
    return {
        "相転移温度": [
            {"name": "joback_Tb", "short": "Joback推定の沸点[K]", "cat": "相転移温度"},
            {"name": "joback_Tm", "short": "Joback推定の融点[K]", "cat": "相転移温度"},
        ],
        "臨界定数": [
            {"name": "joback_Tc", "short": "気液が区別不能になる温度", "cat": "臨界定数"},
            {"name": "joback_Pc", "short": "臨界点での圧力[bar]", "cat": "臨界定数"},
            {"name": "joback_Vc", "short": "臨界点でのモル体積", "cat": "臨界定数"},
        ],
        "熱力学的エネルギー": [
            {"name": "joback_Hf", "short": "元素から生成する際の熱量", "cat": "エネルギー"},
            {"name": "joback_Gf", "short": "自発性を決めるエネルギー", "cat": "エネルギー"},
            {"name": "joback_Cp298", "short": "温度1K上昇に要する熱量", "cat": "エネルギー"},
        ],
        "構造情報": [
            {"name": "joback_n_groups", "short": "SMARTSで一致した原子団数", "cat": "構造情報"},
        ],
    }


def get_skfp_catalog() -> dict[str, list[dict]]:
    """scikit-fingerprints カタログ。ソース: skfp_adapter.py _FP_CONFIGS"""
    return {
        "円形フィンガープリント": [
            {"name": "ECFP", "short": "原子環境の円形探索FP", "cat": "円形FP", "bits": 2048},
            {"name": "FCFP", "short": "薬理学的特徴付き円形FP", "cat": "円形FP", "bits": 2048},
        ],
        "構造キー型": [
            {"name": "MACCS", "short": "166種の部分構造有無判定", "cat": "構造キー", "bits": 167},
            {"name": "Klekota-Roth", "short": "4860種の詳細部分構造", "cat": "構造キー", "bits": 4860},
        ],
        "経路ベース型": [
            {"name": "RDKit", "short": "分子グラフの経路パターン", "cat": "経路型", "bits": 2048},
            {"name": "Avalon", "short": "部分構造の高速符号化", "cat": "経路型", "bits": 512},
            {"name": "Layered", "short": "階層的なグラフ経路情報", "cat": "経路型", "bits": 2048},
            {"name": "Pattern", "short": "SMARTSパターンの一致判定", "cat": "経路型", "bits": 2048},
            {"name": "LINGO", "short": "SMILES部分文字列の出現", "cat": "経路型", "bits": 1024},
        ],
        "原子ペア・トーション型": [
            {"name": "Atom Pair", "short": "原子対の距離と種類の符号", "cat": "ペア型", "bits": 2048},
            {"name": "TopologicalTorsion", "short": "4原子結合パスの回転型", "cat": "トーション", "bits": 2048},
            {"name": "MAP", "short": "MinHash原子ペアFP", "cat": "ペア型", "bits": 2048},
        ],
        "薬理学的特徴型": [
            {"name": "ERG", "short": "薬理学的特徴の距離分布", "cat": "薬理学", "bits": 0},
            {"name": "PhysiochemicalProperties", "short": "物理化学的特徴の符号化", "cat": "薬理学", "bits": 0},
        ],
        "3D形状型": [
            {"name": "GETAWAY", "short": "分子の3D幾何学的特徴量", "cat": "3D", "bits": 0},
            {"name": "MORSE", "short": "原子間距離の変換符号化", "cat": "3D", "bits": 0},
            {"name": "WHIM", "short": "分子の3D重心拡散パターン", "cat": "3D", "bits": 0},
            {"name": "Autocorrelation", "short": "原子物性の距離相関関数", "cat": "3D", "bits": 0},
        ],
    }


def get_mordred_catalog() -> dict[str, list[dict]]:
    """Mordred厳選記述子カタログ。ソース: mordred_adapter.py SELECTED_DESCRIPTORS"""
    return {
        "分子サイズ・原子数": [
            {"name": "MW", "short": "分子全体の質量(Da)", "cat": "分子サイズ"},
            {"name": "nHeavyAtom", "short": "水素除いた骨格原子数", "cat": "分子サイズ"},
            {"name": "nAtom", "short": "水素含む全原子の総数", "cat": "分子サイズ"},
            {"name": "nBonds", "short": "全化学結合の本数", "cat": "分子サイズ"},
            {"name": "nBondsO", "short": "酸素が関与する結合の数", "cat": "分子サイズ"},
            {"name": "nBondsS", "short": "硫黄が関与する結合の数", "cat": "分子サイズ"},
        ],
        "環構造": [
            {"name": "nRing", "short": "環構造の総数", "cat": "環構造"},
            {"name": "nHRing", "short": "ヘテロ原子を含む環の数", "cat": "環構造"},
            {"name": "nARing", "short": "π共役した芳香環の総数", "cat": "環構造"},
            {"name": "nBRing", "short": "ベンゼン型六員環の数", "cat": "環構造"},
            {"name": "nFARing", "short": "縮合した芳香環の数", "cat": "環構造"},
            {"name": "nFHRing", "short": "縮合したヘテロ環の数", "cat": "環構造"},
            {"name": "nFRing", "short": "辺を共有する縮合環の数", "cat": "環構造"},
            {"name": "nSpiro", "short": "1原子共有のスピロ環数", "cat": "環構造"},
            {"name": "nBridgehead", "short": "架橋構造の橋頭位原子数", "cat": "環構造"},
        ],
        "元素組成": [
            {"name": "nC", "short": "炭素原子の数", "cat": "元素組成"},
            {"name": "nN", "short": "窒素原子の数", "cat": "元素組成"},
            {"name": "nO", "short": "酸素原子の数", "cat": "元素組成"},
            {"name": "nS", "short": "硫黄原子の数", "cat": "元素組成"},
            {"name": "nF", "short": "フッ素原子の数", "cat": "元素組成"},
            {"name": "nCl", "short": "塩素原子の数", "cat": "元素組成"},
            {"name": "nBr", "short": "臭素原子の数", "cat": "元素組成"},
            {"name": "nI", "short": "ヨウ素原子の数", "cat": "元素組成"},
            {"name": "nHet", "short": "C,H以外の原子の総数", "cat": "元素組成"},
            {"name": "nHetero", "short": "N,O,Sなどヘテロ原子数", "cat": "元素組成"},
        ],
        "水素結合・柔軟性": [
            {"name": "nHBAcc", "short": "H結合を受ける部位の数", "cat": "水素結合"},
            {"name": "nHBDon", "short": "H結合を与える部位の数", "cat": "水素結合"},
            {"name": "nHBAcc_Lipin", "short": "Lipinski基準の受容体数", "cat": "水素結合"},
            {"name": "nHBDon_Lipin", "short": "Lipinski基準の供与体数", "cat": "水素結合"},
            {"name": "nRotB", "short": "自由回転できる結合の数", "cat": "柔軟性"},
            {"name": "RotRatio", "short": "全結合中の回転可能割合", "cat": "柔軟性"},
        ],
        "極性・疎水性・表面積": [
            {"name": "TPSA", "short": "N,O由来の極性面の広さ", "cat": "極性"},
            {"name": "TopoPSA", "short": "2D版の極性表面積推定", "cat": "極性"},
            {"name": "LogP", "short": "油水間の溶けやすさ比率", "cat": "疎水性"},
            {"name": "SLogP", "short": "原子寄与法のLogP推定", "cat": "疎水性"},
            {"name": "LabuteASA", "short": "溶媒が接触できる表面積", "cat": "表面積"},
        ],
        "位相的形状・複雑性": [
            {"name": "BertzCT", "short": "構造の分岐・環の複雑度", "cat": "複雑性"},
            {"name": "WPath", "short": "全原子対間の距離合計値", "cat": "位相的"},
            {"name": "WPol", "short": "距離3の原子対数＝分岐度", "cat": "位相的"},
            {"name": "Lop", "short": "分子グラフの対称性指標", "cat": "位相的"},
            {"name": "FragCpx", "short": "フラグメント構成の複雑度", "cat": "複雑性"},
            {"name": "BalabanJ", "short": "グラフの対称均一性指標", "cat": "位相的"},
            {"name": "Ipc", "short": "分子グラフの情報エントロピー", "cat": "位相的"},
        ],
        "Kappa形状指数": [
            {"name": "Kier1", "short": "直鎖らしさの形状指標", "cat": "Kappa"},
            {"name": "Kier2", "short": "分岐の多さの形状指標", "cat": "Kappa"},
            {"name": "Kier3", "short": "空間的広がりの形状指標", "cat": "Kappa"},
            {"name": "KierFlex", "short": "1次と2次κ比の柔軟度", "cat": "Kappa"},
        ],
        "情報理論記述子": [
            {"name": "IC0", "short": "原子種の情報エントロピー", "cat": "情報理論"},
            {"name": "IC1", "short": "結合パターンの多様性", "cat": "情報理論"},
            {"name": "IC2", "short": "2結合先の環境多様性", "cat": "情報理論"},
            {"name": "TIC0", "short": "原子数加重の情報含量", "cat": "情報理論"},
            {"name": "SIC0", "short": "正規化した0次情報含量", "cat": "情報理論"},
            {"name": "SIC1", "short": "正規化した1次情報含量", "cat": "情報理論"},
            {"name": "CIC0", "short": "最大値との差の0次情報", "cat": "情報理論"},
            {"name": "CIC1", "short": "最大値との差の1次情報", "cat": "情報理論"},
        ],
        "電荷表面積分布": [
            {"name": "PEOE_VSA1", "short": "強い負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA2", "short": "やや負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA3", "short": "弱い負電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA4", "short": "中性付近の原子の表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA5", "short": "微弱正電荷の原子表面積", "cat": "表面積分布"},
            {"name": "PEOE_VSA6", "short": "正電荷帯の原子の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA1", "short": "低分極率原子の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA2", "short": "中分極率原子の表面積", "cat": "表面積分布"},
            {"name": "SMR_VSA3", "short": "高分極率原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA1", "short": "最も親水的原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA2", "short": "親水的な原子の表面積", "cat": "表面積分布"},
            {"name": "SlogP_VSA3", "short": "疎水的な原子の表面積", "cat": "表面積分布"},
        ],
        "EState・BCUT・接続性": [
            {"name": "EState_VSA1", "short": "電子リッチ原子の表面積", "cat": "EState"},
            {"name": "EState_VSA2", "short": "中間電子状態原子の面積", "cat": "EState"},
            {"name": "EState_VSA3", "short": "電子プア原子の表面積", "cat": "EState"},
            {"name": "MaxEStateIndex", "short": "電子受容しやすい原子値", "cat": "EState"},
            {"name": "MinEStateIndex", "short": "電子供与しやすい原子値", "cat": "EState"},
            {"name": "MaxAbsEStateIndex", "short": "電荷影響力が最大の原子", "cat": "EState"},
            {"name": "BCUTc-1h", "short": "電荷分布の最大固有値", "cat": "BCUT"},
            {"name": "BCUTc-1l", "short": "電荷分布の最小固有値", "cat": "BCUT"},
            {"name": "BCUTdv-1h", "short": "原子価分布の最大固有値", "cat": "BCUT"},
            {"name": "BCUTdv-1l", "short": "原子価分布の最小固有値", "cat": "BCUT"},
            {"name": "AXp-0dv", "short": "0次の原子価接続性指数", "cat": "接続性"},
            {"name": "AXp-1dv", "short": "1次の原子価接続性指数", "cat": "接続性"},
            {"name": "AXp-2dv", "short": "2次の原子価接続性指数", "cat": "接続性"},
        ],
    }


def get_molfeat_catalog() -> dict[str, list[dict]]:
    """Molfeat計算機タイプカタログ。ソース: molfeat_adapter.py _CALCULATOR_TYPES"""
    return {
        "フィンガープリント系": [
            {"name": "ecfp", "short": "原子環境の円形探索FP", "cat": "FP", "bits": 2048},
            {"name": "fcfp", "short": "薬理学的特徴付き円形FP", "cat": "FP", "bits": 2048},
            {"name": "maccs", "short": "166種部分構造の有無判定", "cat": "FP", "bits": 167},
            {"name": "topological", "short": "分子グラフの経路走査FP", "cat": "FP", "bits": 2048},
            {"name": "avalon", "short": "部分構造の高速符号化FP", "cat": "FP", "bits": 512},
            {"name": "atompair", "short": "原子対の距離と種類符号", "cat": "FP", "bits": 2048},
            {"name": "rdkit", "short": "RDKit経路ベースFP", "cat": "FP", "bits": 2048},
        ],
        "記述子ベース": [
            {"name": "desc2d", "short": "RDKitの2D物性記述子群", "cat": "記述子", "bits": 200},
            {"name": "desc3d", "short": "3D座標からの形状記述子", "cat": "記述子", "bits": 200},
        ],
        "薬理学的特徴": [
            {"name": "cats", "short": "薬理学特徴の距離スペクトル", "cat": "薬理学", "bits": 0},
            {"name": "pharm2d", "short": "薬理学特徴ペアの2Dパターン", "cat": "薬理学", "bits": 0},
        ],
        "構造キー・骨格": [
            {"name": "scaffoldkeys", "short": "分子骨格の分類符号化", "cat": "骨格", "bits": 0},
            {"name": "skeys", "short": "SMARTSパターンの符号化", "cat": "骨格", "bits": 0},
        ],
        "3D形状記述子": [
            {"name": "electroshape", "short": "電荷込み3D形状の符号化", "cat": "3D", "bits": 0},
            {"name": "usr", "short": "3D形状の回転不変表現", "cat": "3D", "bits": 0},
            {"name": "usrcat", "short": "USR+薬理学特徴による形状", "cat": "3D", "bits": 0},
        ],
    }


def get_cosmo_catalog() -> dict[str, list[dict]]:
    """COSMO-RS記述子カタログ。ソース: cosmo_adapter.py _COSMO_DESCRIPTORS"""
    return {
        "溶液熱力学": [
            {"name": "mu_comb", "short": "分子の形状由来の化学的位", "cat": "化学ポテンシャル"},
            {"name": "mu_res", "short": "表面電荷由来の化学的位", "cat": "化学ポテンシャル"},
            {"name": "ln_gamma", "short": "理想溶液からのずれ度合", "cat": "活量係数"},
        ],
    }


def get_descriptastorus_catalog() -> dict[str, list[dict]]:
    """DescriptaStorus（Merck製 200+2D記述子）カタログ。"""
    return {
        "物理化学的性質": [
            {"name": "ds_MolWt", "short": "分子量(Da)", "cat": "物性"},
            {"name": "ds_MolLogP", "short": "油水分配係数LogP", "cat": "物性"},
            {"name": "ds_TPSA", "short": "極性表面積(Å²)", "cat": "物性"},
            {"name": "ds_MolMR", "short": "分子屈折率(分極率)", "cat": "物性"},
            {"name": "ds_HeavyAtomMolWt", "short": "H除外骨格質量", "cat": "物性"},
            {"name": "ds_ExactMolWt", "short": "同位体精密質量", "cat": "物性"},
            {"name": "ds_FractionCSP3", "short": "sp3混成炭素の割合", "cat": "物性"},
        ],
        "電子状態・電荷": [
            {"name": "ds_MaxPartialCharge", "short": "最大部分電荷", "cat": "電荷"},
            {"name": "ds_MinPartialCharge", "short": "最小部分電荷", "cat": "電荷"},
            {"name": "ds_MaxAbsPartialCharge", "short": "電荷偏り最大", "cat": "電荷"},
            {"name": "ds_MinAbsPartialCharge", "short": "電荷偏り最小", "cat": "電荷"},
            {"name": "ds_NumRadicalElectrons", "short": "不対電子数", "cat": "電荷"},
            {"name": "ds_NumValenceElectrons", "short": "価電子の総数", "cat": "電荷"},
        ],
        "水素結合・極性": [
            {"name": "ds_NumHAcceptors", "short": "H結合受容部位数", "cat": "水素結合"},
            {"name": "ds_NumHDonors", "short": "H結合供与部位数", "cat": "水素結合"},
            {"name": "ds_NHOHCount", "short": "NH/OHの数", "cat": "水素結合"},
            {"name": "ds_NOCount", "short": "N+O原子の合計", "cat": "水素結合"},
        ],
        "トポロジー・環構造": [
            {"name": "ds_NumRotatableBonds", "short": "回転可能結合数", "cat": "トポロジー"},
            {"name": "ds_RingCount", "short": "環の総数", "cat": "トポロジー"},
            {"name": "ds_NumAromaticRings", "short": "芳香環の数", "cat": "トポロジー"},
            {"name": "ds_NumAliphaticRings", "short": "非芳香環の数", "cat": "トポロジー"},
            {"name": "ds_NumSaturatedRings", "short": "飽和環の数", "cat": "トポロジー"},
            {"name": "ds_NumHeterocycles", "short": "ヘテロ環の数", "cat": "トポロジー"},
            {"name": "ds_HeavyAtomCount", "short": "非水素原子数", "cat": "トポロジー"},
            {"name": "ds_NumAmideBonds", "short": "アミド結合数", "cat": "トポロジー"},
        ],
        "薬品適性": [
            {"name": "ds_qed", "short": "薬品としての適性(0-1)", "cat": "薬品適性"},
        ],
        "位相的指数": [
            {"name": "ds_BalabanJ", "short": "Balaban対称性指数", "cat": "位相的"},
            {"name": "ds_BertzCT", "short": "Bertz複雑度", "cat": "位相的"},
            {"name": "ds_HallKierAlpha", "short": "Hall-Kier形状値", "cat": "位相的"},
            {"name": "ds_Kappa1", "short": "κ形状指数1次", "cat": "位相的"},
            {"name": "ds_Kappa2", "short": "κ形状指数2次", "cat": "位相的"},
            {"name": "ds_Kappa3", "short": "κ形状指数3次", "cat": "位相的"},
        ],
        "BCUT・表面積分布": [
            {"name": "ds_BCUT2D_MWHI", "short": "原子量分布の最大偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_MWLOW", "short": "原子量分布の最小偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_CHGHI", "short": "電荷分布の最大偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_CHGLO", "short": "電荷分布の最小偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_LOGPHI", "short": "LogP分布の最大偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_LOGPLOW", "short": "LogP分布の最小偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_MRHI", "short": "分極率分布の最大偏り", "cat": "BCUT"},
            {"name": "ds_BCUT2D_MRLOW", "short": "分極率分布の最小偏り", "cat": "BCUT"},
        ],
    }


def get_padel_catalog() -> dict[str, list[dict]]:
    """PaDEL（CDK由来 1600+2D記述子）カタログ。主要グループのみ。"""
    return {
        "原子数・結合数": [
            {"name": "padel_nAtom", "short": "全原子数", "cat": "原子数"},
            {"name": "padel_nHeavyAtom", "short": "H以外の原子数", "cat": "原子数"},
            {"name": "padel_nBonds", "short": "全結合数", "cat": "結合数"},
            {"name": "padel_nRotB", "short": "回転可能結合数", "cat": "結合数"},
        ],
        "電荷・電子状態": [
            {"name": "padel_DPSA1", "short": "正電荷部分表面積の和", "cat": "CPSA"},
            {"name": "padel_DPSA2", "short": "負電荷部分表面積の和", "cat": "CPSA"},
            {"name": "padel_RPSA", "short": "相対的極性表面積", "cat": "CPSA"},
            {"name": "padel_RASA", "short": "相対的非極性表面積", "cat": "CPSA"},
        ],
        "水素結合": [
            {"name": "padel_nHBAcc", "short": "H結合受容体数", "cat": "水素結合"},
            {"name": "padel_nHBDon", "short": "H結合供与体数", "cat": "水素結合"},
        ],
        "トポロジー指数": [
            {"name": "padel_Zagreb", "short": "Zagreb指数(結合度)", "cat": "トポロジー"},
            {"name": "padel_WienerPath", "short": "Wiener経路数(分岐度)", "cat": "トポロジー"},
            {"name": "padel_WienerPol", "short": "Wiener極性(距離3対数)", "cat": "トポロジー"},
            {"name": "padel_MDEC", "short": "分子距離辺カウント", "cat": "トポロジー"},
        ],
        "環構造": [
            {"name": "padel_nRing", "short": "環構造の総数", "cat": "環構造"},
            {"name": "padel_nAromRing", "short": "芳香環の数", "cat": "環構造"},
            {"name": "padel_nFRing", "short": "縮合環の数", "cat": "環構造"},
        ],
        "分子サイズ・物性": [
            {"name": "padel_MW", "short": "分子量(Da)", "cat": "物性"},
            {"name": "padel_TPSA", "short": "極性表面積", "cat": "物性"},
            {"name": "padel_ALogP", "short": "原子寄与法LogP", "cat": "物性"},
            {"name": "padel_AMR", "short": "原子寄与法分子屈折率", "cat": "物性"},
        ],
        "接続性指数": [
            {"name": "padel_Chi0", "short": "0次接続性指数", "cat": "接続性"},
            {"name": "padel_Chi1", "short": "1次接続性指数", "cat": "接続性"},
            {"name": "padel_Chi0v", "short": "0次価数接続性指数", "cat": "接続性"},
            {"name": "padel_Chi1v", "short": "1次価数接続性指数", "cat": "接続性"},
        ],
        "情報理論記述子": [
            {"name": "padel_IC0", "short": "原子種の情報エントロピー", "cat": "情報理論"},
            {"name": "padel_SIC0", "short": "正規化情報含量", "cat": "情報理論"},
        ],
        "Barysz距離行列記述子": [
            {"name": "padel_SpMAD_Dzm", "short": "質量加重距離行列の和", "cat": "Barysz"},
            {"name": "padel_SpDiam_Dzm", "short": "質量加重距離行列の直径", "cat": "Barysz"},
        ],
        "2Dオートコレレーション": [
            {"name": "padel_ATS1m", "short": "Moreau-Broto(1次,質量)", "cat": "自己相関"},
            {"name": "padel_ATS1v", "short": "Moreau-Broto(1次,VdW)", "cat": "自己相関"},
            {"name": "padel_ATS1e", "short": "Moreau-Broto(1次,電気陰性度)", "cat": "自己相関"},
            {"name": "padel_ATS1p", "short": "Moreau-Broto(1次,分極率)", "cat": "自己相関"},
        ],
    }


# ═══════════════════════════════════════════════════════════
# 統合カタログ取得
# ═══════════════════════════════════════════════════════════

ENGINE_CATALOG_MAP = {
    "RDKit": get_rdkit_catalog,
    "XTB": get_xtb_catalog,
    "原子団寄与法": get_group_contrib_catalog,
    "scikit-FP": get_skfp_catalog,
    "Mordred": get_mordred_catalog,
    "Molfeat": get_molfeat_catalog,
    "COSMO-RS": get_cosmo_catalog,
    "DescriptaStorus": get_descriptastorus_catalog,
    "PaDEL": get_padel_catalog,
}

SUPPORTED_ENGINES = list(ENGINE_CATALOG_MAP.keys())


def get_catalog(engine_name: str) -> dict[str, list[dict]] | None:
    """エンジン名からカタログを取得する。"""
    fn = ENGINE_CATALOG_MAP.get(engine_name)
    return fn() if fn else None
