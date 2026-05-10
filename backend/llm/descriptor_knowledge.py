"""
backend/llm/descriptor_knowledge.py

化学物性カテゴリ別の推奨記述子データベース。
LLMが目的変数から適切な記述子を推奨するための知識ベース。
"""


# 物性カテゴリ定義
PROPERTY_CATEGORIES = {
    "光・電磁気系": {
        "keywords": [
            "屈折率", "refractive", "ri", "n_d", "屈折",
            "吸収", "absorption", "uv", "vis", "uv-vis", "吸収率",
            "誘電率", "dielectric", "permittivity", "ε",
            "誘電正接", "dissipation", "tan delta", "loss tangent",
            "アッベ数", "abbe", "abbe number", "ν_d",
            "バンドギャップ", "band gap", "bandgap", "homo", "lumo", "eg",
            "導電率", "conductivity", "conductance", "電気伝導",
        ],
        "properties": {
            "屈折率 (Refractive Index)": {
                "description": "ローレンツ・ローレンツの式に基づき、材料の密度と分子の分極率が支配的。π電子系（芳香環）の多さや密なパッキングが屈折率を高める。",
                "descriptors": [
                    ("MolMR", "RDKit", "モル屈折（分極率の直接的な指標）"),
                    ("MolWt", "RDKit", "分子量"),
                    ("NumAromaticRings", "RDKit", "芳香環の数（π電子の非局在化による高分極化）"),
                    ("FractionCSP3", "RDKit", "sp3炭素の割合（値が低いほどπ電子系が多く屈折率が高い）"),
                ],
            },
            "吸収率 (Absorption / UV-Vis)": {
                "description": "光の吸収は電子遷移によるため、HOMO-LUMOギャップや共役系の長さが最も影響。部分電荷や特定の発色団の存在も重要。",
                "descriptors": [
                    ("NumAromaticRings", "RDKit", "芳香環の数"),
                    ("TPSA", "RDKit", "トポロジカル極性表面積（極性相互作用の影響）"),
                ],
            },
            "誘電率 (Dielectric Constant)": {
                "description": "電場に対する応答性。永久双極子モーメントと、分子の分極率（電子分極）、および有効体積のバランスからローレンツ・ローレンツ・オンサーガー等の式に従う。",
                "descriptors": [
                    ("MolMR", "RDKit", "モル屈折（分極率の別指標）"),
                    ("TPSA", "RDKit", "極性表面積"),
                    ("NumHDonors", "RDKit", "水素結合供与体"),
                    ("NumHeteroatoms", "RDKit", "ヘテロ原子(O, N, F等)の数（局所双極子の要因）"),
                ],
            },
            "誘電正接 (Dissipation Factor / Tan Delta)": {
                "description": "高周波電場におけるエネルギー損失。永久双極子の緩和（動きやすさ）や、不純物（吸水）、主鎖のTgなどが損失に寄与。",
                "descriptors": [
                    ("NumRotatableBonds", "RDKit", "回転可能結合数（双極子部位の局所運動性）"),
                    ("MolLogP", "RDKit", "脂溶性（吸水＝水分子による巨大な誘電損失を防ぐ指標）"),
                    ("TPSA", "RDKit", "極性表面積（吸水性を示す）"),
                    ("MolWt", "RDKit", "分子量（極性末端基の比率を下げる効果）"),
                ],
            },
            "アッベ数 (Abbe Number)": {
                "description": "色分散の小ささを示す。分極率の波長依存性（異常分散）に関係し、電子分極が小さくπ電子系が少ないほどアッべ数が高い。屈折率と逆相関の傾向。",
                "descriptors": [
                    ("MolMR", "RDKit", "モル屈折（分極率の指標、アッベ数と負の相関）"),
                    ("NumAromaticRings", "RDKit", "芳香環数（π電子→異常分散→低アッベ数）"),
                    ("FractionCSP3", "RDKit", "sp3炭素比率（高いほど分散低→高アッベ数）"),
                    ("MolWt", "RDKit", "分子量"),
                ],
            },
            "バンドギャップ (Band Gap)": {
                "description": "電子のHOMO-LUMOギャップに対応。共役長、電子供与・吸引基のバランス、分子の平面性が支配的。有機半導体・OLEDの設計指標。",
                "descriptors": [
                    ("NumAromaticRings", "RDKit", "芳香環数（π非局在化）"),
                ],
            },
            "導電率 (Electrical Conductivity)": {
                "description": "電荷キャリアの移動度に依存。HOMO-LUMOギャップ（小さいほど導電性）、共役系、ドーパント相互作用、ホッピング経路の有無が支配的。",
                "descriptors": [
                    ("NumAromaticRings", "RDKit", "芳香環数（π軌道の非局在化）"),
                    ("FractionCSP3", "RDKit", "sp3炭素比率（低いほど共役→高導電）"),
                ],
            },
        },
    },
    "熱物性系": {
        "keywords": [
            "融点", "melting point", "mp", "m.p.",
            "沸点", "boiling point", "bp", "b.p.",
            "ガラス転移温度", "glass transition", "tg", "t_g",
            "熱伝導率", "thermal conductivity",
            "熱膨張", "thermal expansion",
            "比熱", "specific heat",
            "蒸気圧", "vapor pressure",
            "凝縮性", "condensation",
        ],
        "properties": {
            "融点 (Melting Point)": {
                "description": "分子間相互作用（水素結合、π-π積層、分散力）と分子対称性が支配的。剛直な骨格・高い対称性・強い水素結合が融点を高める。",
                "descriptors": [
                    ("MolWt", "RDKit", "分子量（一般に大きいほど融点が高い）"),
                    ("NumRotatableBonds", "RDKit", "回転可能結合数（多いほど柔軟→融点低下）"),
                    ("TPSA", "RDKit", "極性表面積（水素結合能の指標）"),
                    ("NumHDonors", "RDKit", "水素結合供与体数"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体数"),
                    ("RingCount", "RDKit", "環数（剛直性の指標）"),
                ],
            },
            "沸点 (Boiling Point)": {
                "description": "分子間力（分散力、双極子-双極子相互作用、水素結合）と分子サイズが支配的。極性基や水素結合部位が多いほど高沸点。",
                "descriptors": [
                    ("MolWt", "RDKit", "分子量"),
                    ("MolLogP", "RDKit", "LogP（疎水性と相関）"),
                    ("TPSA", "RDKit", "極性表面積"),
                    ("NumHDonors", "RDKit", "水素結合供与体数"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体数"),
                    ("NumRotatableBonds", "RDKit", "回転可能結合数"),
                ],
            },
            "ガラス転移温度 (Glass Transition Temperature)": {
                "description": "分子の運動性と剛直性のバランス。側鎖の柔軟性、主鎖の剛直性、立体障害がTgを決定。芳香環や剛直骨格がTgを上昇させる。",
                "descriptors": [
                    ("NumRotatableBonds", "RDKit", "回転可能結合数（多いほど柔軟→低Tg）"),
                    ("FractionCSP3", "RDKit", "sp3炭素比率（高いほど柔軟→低Tg）"),
                    ("NumAromaticRings", "RDKit", "芳香環数（剛直→高Tg）"),
                    ("MolWt", "RDKit", "分子量（大きいほどTg上昇）"),
                    ("TPSA", "RDKit", "極性表面積（極性相互作用→高Tg）"),
                ],
            },
            "蒸気圧 (Vapor Pressure)": {
                "description": "分子の大きさと分子間力が支配的。小さくて対称性が高く、極性基が少ないほど蒸気圧が高い（揮発しやすい）。",
                "descriptors": [
                    ("MolWt", "RDKit", "分子量（大きいほど蒸気圧低下）"),
                    ("MolLogP", "RDKit", "LogP（高いほど揮発しやすい）"),
                    ("TPSA", "RDKit", "極性表面積（極性が高いほど蒸気圧低下）"),
                    ("NumHDonors", "RDKit", "水素結合供与体数"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体数"),
                ],
            },
        },
    },
    "溶解・分配系": {
        "keywords": [
            "溶解度", "solubility", "solub",
            "分配係数", "partition", "logp", "log d",
            "浸透性", "permeability",
            "分配", "distribution",
            "水溶解度", "aqueous solubility",
        ],
        "properties": {
            "水溶解度 (Aqueous Solubility)": {
                "description": "分子の極性、水素結合能、疎水性が支配的。LogPが低く、極性基が多く、分子量が小さいほど水溶性が高い。",
                "descriptors": [
                    ("MolLogP", "RDKit", "LogP（低いほど水溶性が高い）"),
                    ("TPSA", "RDKit", "極性表面積（大きいほど水溶性が高い）"),
                    ("MolWt", "RDKit", "分子量（小さいほど水溶性が高い）"),
                    ("NumHDonors", "RDKit", "水素結合供与体数"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体数"),
                    ("NumRotatableBonds", "RDKit", "回転可能結合数（柔軟性と相関）"),
                ],
            },
            "分配係数 (Partition Coefficient / LogP)": {
                "description": "疎水性と親水性のバランス。炭化水素骨格は疎水性、極性基は親水性に寄与。計算値と実測値の比較にも使用。",
                "descriptors": [
                    ("MolLogP", "RDKit", "LogP（自己相関チェック用）"),
                    ("MolMR", "RDKit", "モル屈折（疎水性の寄与）"),
                    ("NumHAcceptors", "RDKit", "ヘテロ原子数（親水性の寄与）"),
                    ("TPSA", "RDKit", "極性表面積"),
                    ("RingCount", "RDKit", "環数（剛直性・疎水性と相関）"),
                ],
            },
        },
    },
    "反応性・安定性": {
        "keywords": [
            "反応性", "reactivity",
            "安定性", "stability",
            "毒性", "toxicity",
            "生分解性", "biodegradation",
            "酸化", "oxidation",
            "加水分解", "hydrolysis",
        ],
        "properties": {
            "反応性 (Reactivity)": {
                "description": "官能基の種類と位置、立体効果、電子効果が支配的。反応部位への電子密度と立体アクセスのしやすさが重要。",
                "descriptors": [
                    ("NumHDonors", "RDKit", "水素結合供与体（反応性部位の指標）"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体"),
                    ("NumHeteroatoms", "RDKit", "ヘテロ原子数（反応性部位）"),
                    ("TPSA", "RDKit", "極性表面積"),
                    ("FractionCSP3", "RDKit", "sp3炭素比率（低いほど反応性が高い傾向）"),
                ],
            },
            "毒性 (Toxicity)": {
                "description": "特定の官能基（警告構造）、分子量、疎水性、および生体内での反応性が関与。QSARモデルで広く使用。",
                "descriptors": [
                    ("MolLogP", "RDKit", "LogP（疎水性と毒性の相関）"),
                    ("MolWt", "RDKit", "分子量"),
                    ("TPSA", "RDKit", "極性表面積"),
                    ("NumHDonors", "RDKit", "水素結合供与体数"),
                    ("NumHAcceptors", "RDKit", "水素結合受容体数"),
                    ("NumAromaticRings", "RDKit", "芳香環数（一部の毒性警告構造）"),
                ],
            },
        },
    },
    "機械的特性": {
        "keywords": [
            "強度", "strength", "tensile",
            "弾性率", "modulus", "elastic",
            "伸び", "elongation",
            "硬度", "hardness",
            "靭性", "toughness",
        ],
        "properties": {
            "弾性率 (Elastic Modulus)": {
                "description": "分子鎖の剛直性、分子間相互作用、架橋密度が支配的。芳香環や剛直骨格、強い分子間力が高弾性率をもたらす。",
                "descriptors": [
                    ("NumAromaticRings", "RDKit", "芳香環数（剛直性→高弾性率）"),
                    ("FractionCSP3", "RDKit", "sp3炭素比率（低いほど剛直）"),
                    ("RingCount", "RDKit", "環数（剛直性の指標）"),
                    ("MolWt", "RDKit", "分子量"),
                    ("TPSA", "RDKit", "極性表面積（分子間力の指標）"),
                ],
            },
        },
    },
}


def find_matching_properties(target_name: str, target_description: str = "") -> list[tuple[str, dict]]:
    """
    目的変数名と説明から、マッチする物性とその推奨記述子を検索する。

    Args:
        target_name: 目的変数名（列名）
        target_description: ユーザーからの追加説明

    Returns:
        [(物性名, 物性情報dict), ...] のリスト（マッチ度順）
    """
    query = f"{target_name} {target_description}".lower()
    results = []

    for category, cat_data in PROPERTY_CATEGORIES.items():
        # カテゴリキーワードチェック
        cat_match = any(kw.lower() in query for kw in cat_data["keywords"])

        for prop_name, prop_data in cat_data["properties"].items():
            score = 0
            prop_short = prop_name.split("(")[0].strip().lower()
            prop_eng = prop_name.split("(")[1].rstrip(")").lower() if "(" in prop_name else ""

            # 物性名（日本語）との一致
            if prop_short in query:
                score += 10
            # 物性名（英語）との一致
            if prop_eng and prop_eng in query:
                score += 8
            # カテゴリキーワードとの一致
            for kw in cat_data["keywords"]:
                if kw.lower() in query:
                    score += 3
            # 説明内のキーワード
            desc_lower = prop_data["description"].lower()
            for word in query.split():
                if len(word) > 3 and word in desc_lower:
                    score += 1

            if score > 0:
                results.append((prop_name, prop_data, score, category))

    # スコア順にソート
    results.sort(key=lambda x: x[2], reverse=True)
    return [(name, {"category": cat, **data}) for name, data, _, cat in results]


def build_descriptor_recommendation_prompt(
    target_col: str,
    target_description: str,
    df_columns: list[str],
    df_sample: str,
    interview_notes: str = "",
) -> str:
    """
    記述子推奨のためのLLMプロンプトを構築する。

    Args:
        target_col: 目的変数名
        target_description: ユーザーからの追加説明
        df_columns: データフレームの列名リスト
        df_sample: データのサンプル（数行）
        interview_notes: ヒヤリングメモ

    Returns:
        LLM用プロンプト文字列
    """
    # マッチする物性を検索
    matches = find_matching_properties(target_col, target_description)

    lines = []
    lines.append("# 記述子推奨タスク")
    lines.append("")
    lines.append("## 目的変数の情報")
    lines.append(f"- 列名: {target_col}")
    if target_description:
        lines.append(f"- ユーザー説明: {target_description}")
    lines.append("")

    lines.append("## データセット情報")
    lines.append(f"- 全列: {', '.join(df_columns)}")
    lines.append("")
    lines.append("## データサンプル")
    lines.append(f"```\n{df_sample}\n```")
    lines.append("")

    if interview_notes:
        lines.append("## ユーザーとのヒヤリングメモ")
        lines.append(interview_notes)
        lines.append("")

    # マッチした物性がある場合
    if matches:
        lines.append("## マッチした物性カテゴリと推奨記述子")
        lines.append("以下の物性が目的変数と一致すると推定されます。これらに基づいて記述子を推奨してください。")
        lines.append("")

        for prop_name, prop_data in matches[:3]:  # 上位3件
            category = prop_data.get("category", "")
            lines.append(f"### {prop_name} [{category}]")
            lines.append(f"説明: {prop_data['description']}")
            lines.append("推奨記述子:")
            for desc_name, desc_lib, desc_reason in prop_data["descriptors"]:
                lines.append(f"  - {desc_name} ({desc_lib}): {desc_reason}")
            lines.append("")
    else:
        lines.append("## 物性カテゴリが特定できませんでした")
        lines.append("ユーザーに以下の質問をしてください：")
        lines.append("- この目的変数はどのような物性を表していますか？（例：屈折率、溶解度、融点など）")
        lines.append("- 化学構造のどのような要因がこの物性に影響すると考えられますか？")
        lines.append("")

    lines.append("## データセットに既存の記述子列がある場合の処理")
    lines.append("以下の列は既に記述子として使用可能です。これらを優先的に考慮してください：")
    existing_descriptors = [c for c in df_columns if c.lower() not in (target_col.lower(), "smiles", "smi", "")]
    if existing_descriptors:
        lines.append(f"{', '.join(existing_descriptors)}")
    else:
        lines.append("（既存の記述子列はありません。SMILESから計算が必要です。）")
    lines.append("")

    lines.append("## 出力形式")
    lines.append("以下のJSON形式で出力してください（コードブロック内）：")
    lines.append("```json")
    lines.append("""{
  "matched_property": "推定される物性名",
  "confidence": "high | medium | low",
  "recommended_descriptors": [
    {
      "name": "記述子名",
      "source": "RDKit | existing_column | ECFP | MACCS",
      "reason": "推奨理由（化学的根拠）",
      "priority": 1
    }
  ],
  "interview_questions": [
    "ユーザーに確認すべき質問（物性が不明な場合など）"
  ],
  "notes": "その他のアドバイス"
}""")
    lines.append("```")
    lines.append("")
    lines.append("## 指示")
    lines.append("1. 目的変数の物性カテゴリを推定し、一致度の高い順に記述子を推奨する。")
    lines.append("2. 推奨記述子は最大10個まで。優先度(priority)は1-5で指定（1が最高）。")
    lines.append("3. 既存の列に同じような記述子がある場合は、それを優先して使用する。")
    lines.append("4. 物性が不明確な場合は、interview_questionsに質問を含める。")
    lines.append("5. 各記述子について、なぜその記述子が適しているかを化学的根拠と共に説明する。")
    lines.append("6. 日本語で回答することを忘れないでください。")

    return "\n".join(lines)


def get_all_descriptor_names() -> list[str]:
    """すべての推奨記述子名のリストを取得する。"""
    names = set()
    for cat_data in PROPERTY_CATEGORIES.values():
        for prop_data in cat_data["properties"].values():
            for desc_name, _, _ in prop_data["descriptors"]:
                names.add(desc_name)
    return sorted(names)


def get_rdkit_descriptor_map() -> dict[str, str]:
    """
    RDKit記述子名からDescriptors関数名へのマッピングを取得する。
    実際のRDKit Descriptorsモジュールの関数名を返す。
    """
    mapping = {
        "MolMR": "MolMR",
        "MolWt": "MolWt",
        "NumAromaticRings": "NumAromaticRings",
        "FractionCSP3": "FractionCSP3",
        "TPSA": "TPSA",
        "NumHDonors": "NumHDonors",
        "NumHAcceptors": "NumHAcceptors",
        "NumHeteroatoms": "NumHeteroatoms",
        "NumRotatableBonds": "NumRotatableBonds",
        "MolLogP": "MolLogP",
        "RingCount": "RingCount",
    }
    return mapping
