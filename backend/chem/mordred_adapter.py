"""
backend/chem/mordred_adapter.py

Mordred を使った化合物記述子計算アダプタ。
RDKit上に構築され、約1800種の2D/3D記述子を計算できる。
QSAR/QSPR研究で広く使用されている包括的な記述子セット。

GitHub: https://github.com/mordred-descriptor/mordred
論文: Moriwaki et al., J Cheminform. 10:4 (2018)
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.chem.base import DescriptorMetadata

import logging

import numpy as np
import pandas as pd

# ── NumPy 2.0 互換パッチ ──
# mordredはnumpy.product等(NumPy 1.x系エイリアス)を使用しているが、
# NumPy 2.0でこれらが削除されたため、モンキーパッチで復元する。
_NP_COMPAT_ALIASES = {
    "product": "prod",
    "cumproduct": "cumprod",
    "sometrue": "any",
    "alltrue": "all",
}
for _alias, _real in _NP_COMPAT_ALIASES.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(np, _real))

from backend.chem.base import BaseChemAdapter, DescriptorResult
from backend.utils.optional_import import safe_import

_rdkit = safe_import("rdkit", "rdkit")
logger = logging.getLogger(__name__)



class MordredAdapter(BaseChemAdapter):
    """
    Mordred による包括的な分子記述子計算アダプタ。

    Mordredは~1800種の2D/3D記述子を計算できるライブラリで、
    QSAR/QSPR（定量的構造活性/物性相関）研究で広く使われる。

    RDKitの上に構築されているため、RDKitが必要。
    インストール: pip install mordred

    Args:
        use_3d: 3D記述子も計算するか（デフォルト: False、高速化のため2Dのみ）
        ignore_3d:  3D座標がない場合の2Dフォールバックを許可するか
        max_descriptors: 上位N個のみ使用（None=全部）
    """

    # 有機化学/ポリマー物性予測に使いやすい主要な2D記述子のサブセット
    # (Mordredは約1800種あるため、デフォルトではこの厳選セットを返す)
    SELECTED_DESCRIPTORS: list[str] = [
        # 分子全体の形状・サイズ
        "MW",           "nHeavyAtom",   "nAtom",        "nBonds",
        "nBondsO",      "nBondsS",
        # 環構造
        "nRing",        "nHRing",       "nARing",       "nBRing",
        "nFARing",      "nFHRing",      "nFRing",       "nSpiro",
        "nBridgehead",
        # 原子タイプ
        "nC",           "nN",           "nO",           "nS",
        "nF",           "nCl",          "nBr",          "nI",
        "nHet",         "nHetero",
        # 水素結合
        "nHBAcc",       "nHBDon",       "nHBAcc_Lipin", "nHBDon_Lipin",
        # 電荷・極性
        "TPSA",
        # 疎水性
        "LogP",         "SLogP",
        # 位相的形状記述子（複雑性）
        "BertzCT",      "TopoPSA",
        # Wiener / Randic系トポロジー
        "WPath",        "WPol",         "Lop",
        # 電子状態 (2D)
        "PEOE_VSA1",    "PEOE_VSA2",    "PEOE_VSA3",    "PEOE_VSA4",
        "PEOE_VSA5",    "PEOE_VSA6",
        "SMR_VSA1",     "SMR_VSA2",     "SMR_VSA3",
        "SlogP_VSA1",   "SlogP_VSA2",   "SlogP_VSA3",
        # Kappa形状指数
        "Kier1",        "Kier2",        "Kier3",        "KierFlex",
        # 情報理論的記述子
        "IC0",          "IC1",          "IC2",          "TIC0",
        # 分子電子グラフ
        "EState_VSA1",  "EState_VSA2",  "EState_VSA3",
        "MaxEStateIndex","MinEStateIndex","MaxAbsEStateIndex",
        # BCUT記述子
        "BCUTc-1h",     "BCUTc-1l",     "BCUTdv-1h",    "BCUTdv-1l",
        # ABCガスケ/ガウジエ
        "AXp-0dv",      "AXp-1dv",      "AXp-2dv",
        # 接触面積
        "LabuteASA",
        # フラグメントベース
        "SIC0",         "SIC1",         "CIC0",         "CIC1",
        # その他
        "Ipc",          "BalabanJ",     "nRotB",        "RotRatio",
        "FragCpx",
    ]

    def __init__(
        self,
        use_3d: bool = False,
        selected_only: bool = True,
    ):
        self.use_3d = use_3d
        self.selected_only = selected_only
        self._descriptor_names: list[str] | None = None

    @property
    def name(self) -> str:
        return "mordred"

    @property
    def description(self) -> str:
        return (
            "Mordred: 約1800種の2D分子記述子を計算できる包括的ライブラリ。"
            "QSAR/QSPR研究で広く使用。RDKit上に構築。"
            "参考: Moriwaki et al., J Cheminform. 10:4 (2018)"
        )

    def is_available(self) -> bool:
        try:
            import mordred  # noqa: F401
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError as e:
            logger.debug(f"MordredAdapter: インポート失敗 - {e}")
            return False
        except Exception as e:
            logger.warning(f"MordredAdapter: Python {'.'.join(str(v) for v in __import__('sys').version_info[:3])} での互換性チェック中にエラー: {e}")
            return False

    def _build_calculator(self):
        """Mordredのcalculatorを構築する（遅延初期化）"""
        from mordred import Calculator, descriptors as mordred_desc
        calc = Calculator(mordred_desc, ignore_3D=not self.use_3d)
        return calc

    def _get_all_mordred_names(self) -> list[str]:
        """Mordredで計算できる全記述子名を返す"""
        if self._descriptor_names is None:
            calc = self._build_calculator()
            self._descriptor_names = [str(d) for d in calc.descriptors]
        return self._descriptor_names

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        """
        SMILES文字列のリストからMordred記述子を計算する。

        Args:
            smiles_list: 入力SMILESのリスト
            **kwargs: 追加オプション（ignore_errors など）

        Returns:
            DescriptorResult インスタンス
        """
        self._require_available()
        logger.info(f"📊 Mordred記述子計算を開始します（{len(smiles_list)}分子）...")

        from rdkit import Chem
        from mordred import Calculator, descriptors as mordred_desc

        calc = Calculator(mordred_desc, ignore_3D=not self.use_3d)

        mols = []
        failed_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                logger.warning(f"MordredAdapter: SMILES parse failed at index {i}: {smi!r}")
                failed_indices.append(i)
                mols.append(None)
            else:
                mols.append(mol)

        try:
            # nproc=1: NumPy 2.0互換パッチがサブプロセスに届かないため、
            # シングルプロセスで計算する
            df_all = calc.pandas([m for m in mols if m is not None], nproc=1)
        except Exception as e:
            logger.error(f"MordredAdapter: 計算中にエラーが発生しました: {e}")
            # フォールバック: 全てNaN
            desc_names = self.get_descriptor_names()
            df_all = pd.DataFrame(
                index=range(len(smiles_list) - len(failed_indices)),
                columns=desc_names,
                dtype=float
            )

        # Mordredの数値以外（エラーオブジェクト等）をNaNに変換
        df_all = df_all.apply(pd.to_numeric, errors="coerce")

        # NaN・無限大を0で補完（機械学習に渡すため）
        df_all = df_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 必要に応じてサブセットのみに絞る
        if self.selected_only:
            available_cols = [c for c in self.SELECTED_DESCRIPTORS if c in df_all.columns]
            df_all = df_all[available_cols] if available_cols else df_all

        # 失敗した行を0行で挿入（全体と行数を一致させる）
        df_result = pd.DataFrame(
            index=range(len(smiles_list)),
            columns=df_all.columns,
            dtype=float
        ).fillna(0.0)

        valid_idx = [i for i in range(len(smiles_list)) if i not in failed_indices]
        if len(df_all) == len(valid_idx):
            df_all.index = valid_idx
            # 明示的にfloat型にキャストしてFutureWarningを防止
            df_all_float = df_all.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            df_result.loc[valid_idx] = df_all_float.values

        return DescriptorResult(
            descriptors=df_result,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={
                "use_3d": self.use_3d,
                "n_descriptors": df_result.shape[1],
                "selected_only": self.selected_only,
            }
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """Mordred記述子の詳細メタデータを返す。"""
        from backend.chem.base import DescriptorMetadata
        
        # 全SELECTED_DESCRIPTORSの化学的意味を定義
        # (記述子名, 化学的意味, is_count)
        raw_meta = [
            # ── 分子サイズ・原子数 ──
            ("MW", "分子量 (Da)。沸点・粘度・蒸気圧に直結する基本物性", False),
            ("nHeavyAtom", "水素以外の原子数。分子骨格の大きさの指標", True),
            ("nAtom", "水素を含む全原子数", True),
            ("nBonds", "全化学結合の数", True),
            ("nBondsO", "酸素原子が関与する結合の数。エーテル・エステル等の極性結合", True),
            ("nBondsS", "硫黄原子が関与する結合の数。チオール・スルホン等", True),

            # ── 環構造 ──
            ("nRing", "環構造の総数。剛直性・安定性に寄与", True),
            ("nHRing", "ヘテロ原子(N,O,S等)を含む環の数。生理活性・反応性に重要", True),
            ("nARing", "芳香環の数。π電子共役による光吸収・熱安定性", True),
            ("nBRing", "ベンゼン環(炭素のみの六員芳香環)の数", True),
            ("nFARing", "縮環構造中の芳香環の数。ナフタレン・アントラセン等", True),
            ("nFHRing", "縮環構造中のヘテロ環の数", True),
            ("nFRing", "縮環(融合環)の数。二環以上が辺を共有する構造", True),
            ("nSpiro", "スピロ環の数。2つの環が1原子を共有する特殊構造", True),
            ("nBridgehead", "橋頭位原子の数。二環式以上の架橋構造", True),

            # ── 原子タイプ ──
            ("nC", "炭素原子数。有機化合物の骨格元素", True),
            ("nN", "窒素原子数。アミン・アミド・ヘテロ環の存在", True),
            ("nO", "酸素原子数。ヒドロキシル・カルボニル・エーテルの存在", True),
            ("nS", "硫黄原子数。チオール・スルフィド・スルホンの存在", True),
            ("nF", "フッ素原子数。高い電気陰性度による極性・代謝安定性", True),
            ("nCl", "塩素原子数。疎水性の増大・生理活性への影響", True),
            ("nBr", "臭素原子数。重い置換基・反応性ハロゲン", True),
            ("nI", "ヨウ素原子数。最も重いハロゲン・造影剤に利用", True),
            ("nHet", "ヘテロ原子(C,H以外)の数。極性・反応性の指標", True),
            ("nHetero", "ヘテロ原子数(別定義)。N,O,S等を含む", True),

            # ── 水素結合・官能基 ──
            ("nHBAcc", "水素結合受容体の数。O,N等の孤立電子対を持つ原子", True),
            ("nHBDon", "水素結合供与体の数。-OH,-NH等", True),
            ("nHBAcc_Lipin", "Lipinski定義の水素結合受容体数(N+O)", True),
            ("nHBDon_Lipin", "Lipinski定義の水素結合供与体数(NH+OH)", True),
            ("nRotB", "回転可能結合数。分子の柔軟性を反映", True),
            ("RotRatio", "回転可能結合比率。全結合中の回転可能結合の割合", False),

            # ── 極性・疎水性 ──
            ("TPSA", "位相的極性表面積 (Å²)。N,O由来の極性表面。水溶性・膜透過性の指標", False),
            ("LogP", "LogP (Wildman-Crippen法)。油/水分配係数の対数。疎水性の基本指標", False),
            ("SLogP", "SLogP (Wildman-Crippen法)。原子ベースLogP推定値", False),
            ("LabuteASA", "Labute近似溶媒接触表面積。溶媒との相互作用面積", False),

            # ── 位相的記述子 ──
            ("BertzCT", "Bertz複雑度。分子グラフの構造的複雑さ。分岐・環が多いほど高い", False),
            ("TopoPSA", "位相的極性表面積(2D版)。3D座標不要のPSA推定", False),
            ("WPath", "Wiener Path Number。分子グラフの全頂点対間距離の合計。分子サイズの指標", False),
            ("WPol", "Wiener Polarity Number。距離3の頂点対の数。分岐度を反映", False),
            ("Lop", "Lopping Index。分子グラフの対称性を反映する指標", False),

            # ── 電荷表面積分布 ──
            ("PEOE_VSA1", "PEOE部分電荷ビン1の表面積。最も負に帯電した原子の表面積", False),
            ("PEOE_VSA2", "PEOE部分電荷ビン2の表面積。負電荷領域", False),
            ("PEOE_VSA3", "PEOE部分電荷ビン3の表面積。やや負の領域", False),
            ("PEOE_VSA4", "PEOE部分電荷ビン4の表面積。中性付近の領域", False),
            ("PEOE_VSA5", "PEOE部分電荷ビン5の表面積。やや正の領域", False),
            ("PEOE_VSA6", "PEOE部分電荷ビン6の表面積。正電荷領域", False),
            ("SMR_VSA1", "屈折率ビン1の表面積。最も分極しにくい原子の表面積", False),
            ("SMR_VSA2", "屈折率ビン2の表面積。低分極率領域", False),
            ("SMR_VSA3", "屈折率ビン3の表面積。高分極率領域", False),
            ("SlogP_VSA1", "LogPビン1の表面積。最も親水的な原子の表面積", False),
            ("SlogP_VSA2", "LogPビン2の表面積。親水的領域", False),
            ("SlogP_VSA3", "LogPビン3の表面積。疎水的領域", False),

            # ── Kappa形状指数 ──
            ("Kier1", "Kier κ1形状指数。分子の直線性を反映。直鎖に近いほど大きい", False),
            ("Kier2", "Kier κ2形状指数。分子の分岐度を反映。分岐が多いほど大きい", False),
            ("Kier3", "Kier κ3形状指数。分子の空間的広がりを反映", False),
            ("KierFlex", "Kier柔軟性指数 (φ)。1次と2次κの比から柔軟性を推定", False),

            # ── 情報理論記述子 ──
            ("IC0", "0次情報含量。原子種の多様性。Shannon情報エントロピーベース", False),
            ("IC1", "1次情報含量。隣接原子の結合パターンの多様性", False),
            ("IC2", "2次情報含量。2結合先までの環境の多様性", False),
            ("TIC0", "0次全情報含量。IC0を原子数で重み付けした値", False),
            ("SIC0", "0次構造情報含量。IC0を正規化した値", False),
            ("SIC1", "1次構造情報含量。IC1を正規化した値", False),
            ("CIC0", "0次相補情報含量。最大エントロピーとIC0の差", False),
            ("CIC1", "1次相補情報含量。最大エントロピーとIC1の差", False),

            # ── 電子状態 (EState) ──
            ("EState_VSA1", "EState表面積ビン1。電子リッチな原子の表面積", False),
            ("EState_VSA2", "EState表面積ビン2。やや電子リッチな原子の表面積", False),
            ("EState_VSA3", "EState表面積ビン3。電子プアな原子の表面積", False),
            ("MaxEStateIndex", "最大EState指数。最も電子受容しやすい原子の値", False),
            ("MinEStateIndex", "最小EState指数。最も電子供与しやすい原子の値", False),
            ("MaxAbsEStateIndex", "最大|EState|指数。電荷偏りが最大の原子", False),

            # ── BCUT記述子（Burden-CAS-University of Texas） ──
            ("BCUTc-1h", "BCUT電荷-高値。電荷分布の最大固有値。分子の電荷の偏りパターン", False),
            ("BCUTc-1l", "BCUT電荷-低値。電荷分布の最小固有値。電荷の均一性", False),
            ("BCUTdv-1h", "BCUT原子価-高値。原子価分布の最大固有値。結合パターンの偏り", False),
            ("BCUTdv-1l", "BCUT原子価-低値。原子価分布の最小固有値", False),

            # ── AXp記述子（原子価連結性指数） ──
            ("AXp-0dv", "0次原子価連結性指数。原子の孤立した性質(原子価ベース)", False),
            ("AXp-1dv", "1次原子価連結性指数。隣接原子との結合パターン(原子価ベース)", False),
            ("AXp-2dv", "2次原子価連結性指数。2結合先までの経路(原子価ベース)", False),

            # ── その他の位相的指標 ──
            ("Ipc", "Bonchev-Trinajstić情報含量。分子グラフの情報量", False),
            ("BalabanJ", "Balaban J指数。分子グラフの均一性。高い→対称的構造", False),
            ("FragCpx", "フラグメント複雑度。分子を構成するフラグメントの複雑さ", False),
        ]
        
        metadata_list = [DescriptorMetadata(n, m, c) for n, m, c in raw_meta]
        
        # SELECTED_DESCRIPTORSに定義漏れがないか検証し、あれば追加
        defined_names = {m.name for m in metadata_list}
        for name in self.SELECTED_DESCRIPTORS:
            if name not in defined_names:
                is_count = name.startswith("n")
                metadata_list.append(DescriptorMetadata(name, name, is_count))
                
        return metadata_list

    def get_descriptor_names(self) -> list[str]:
        """
        計算可能な記述子名のリストを返す。
        """
        if self.selected_only:
            return self.SELECTED_DESCRIPTORS
        else:
            if self.is_available():
                return self._get_all_mordred_names()
            return self.SELECTED_DESCRIPTORS
