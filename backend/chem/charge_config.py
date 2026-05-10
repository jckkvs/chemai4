"""
backend/chem/charge_config.py

分子電荷・スピン・プロトン化状態の設定データクラス。

量子化学計算（XTB, COSMO-RS）と記述子計算（RDKit）で共通して使用される。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# プロトン化モードの定数
ProtonationMode = Literal[
    "as_is",       # SMILESのまま使用（デフォルト）
    "auto_ph",     # UniPKaのpKaを使ってpHで自動プロトン化
    "neutral",     # 全て中性化（塩・イオンをニュートライズ）
    "max_acid",    # 最大脱プロトン化形（最も酸性的な状態）
    "max_base",    # 最大プロトン化形（最も塩基性的な状態）
]

# 部分電荷モデル
PartialChargeModel = Literal[
    "none",         # 部分電荷記述子を計算しない
    "gasteiger",    # RDKit Gasteiger-Marsili 電荷（高速・経験的）
    "xtb_mulliken", # GFN2-xTB Mulliken 電荷（精密・計算コスト高）
]


@dataclass
class MoleculeChargeConfig:
    """
    1つのSMILS列（または分子セット）に対する電荷・スピン設定。

    Attributes
    ----------
    formal_charge : int
        分子全体の形式電荷（整数）。例: 0（中性）, +1（カチオン）, -1（アニオン）
        XTBに --chrg として渡される。
        Noneのとき、SMILESから自動計算する（RDKit GetFormalCharge）。
    spin_multiplicity : int
        スピン多重度 2S+1。
        1 = 閉殻（偶電子数）
        2 = 一重ラジカル（奇電子数、例: ニトロキシドラジカル）
        3 = 三重項（例: カルベン、ビラジカル、O2）
        XTBに --uhf (M-1) として渡される。
    ph : float | None
        計算を行う溶液のpH。
        protonate_mode == "auto_ph" のときのみ使用。
        None 以外のとき、UniPKaでpKaを予測してHenderson-Hasselbalch式を適用。
    protonate_mode : ProtonationMode
        プロトン化処理の方針。
    partial_charge_model : PartialChargeModel
        部分電荷モデル。Gasteiger選択時は RDKit で計算し記述子に追加。
    consider_tautomers : bool
        True のとき、RDKit MolStandardize で互変異性体の最安定形を探索する。
    auto_charge_from_smiles : bool
        True のとき、formal_charge は SMILES の形式電荷を自動読取する。
        （SMILESに [NH4+] が含まれている場合は +1 を自動検出）
    """
    formal_charge: int = 0
    spin_multiplicity: int = 1
    ph: float | None = None
    protonate_mode: ProtonationMode = "as_is"
    partial_charge_model: PartialChargeModel = "gasteiger"
    consider_tautomers: bool = False
    auto_charge_from_smiles: bool = True

    def __post_init__(self) -> None:
        if self.spin_multiplicity < 1:
            raise ValueError(f"spin_multiplicity は1以上でなければなりません: {self.spin_multiplicity}")
        if self.formal_charge < -10 or self.formal_charge > 10:
            raise ValueError(f"formal_charge が範囲外です: {self.formal_charge}")

    @property
    def uhf(self) -> int:
        """XTBの --uhf 引数値（= スピン多重度 - 1 = 不対電子数）"""
        return self.spin_multiplicity - 1

    def to_xtb_args(self, charge_override: int | None = None) -> list[str]:
        """XTBコマンドライン引数のリストを返す。"""
        ch = charge_override if charge_override is not None else self.formal_charge
        args = ["--chrg", str(ch)]
        if self.uhf > 0:
            args += ["--uhf", str(self.uhf)]
        return args

    @classmethod
    def default(cls) -> "MoleculeChargeConfig":
        """デフォルト設定（中性・閉殻・Gasteiger電荷・SMILESから自動電荷読取）"""
        return cls()

    @classmethod
    def for_radical(cls, charge: int = 0) -> "MoleculeChargeConfig":
        """ラジカル種（スピン多重度=2）用のプリセット"""
        return cls(formal_charge=charge, spin_multiplicity=2)

    @classmethod
    def at_physiological_ph(cls) -> "MoleculeChargeConfig":
        """生理的条件（pH 7.4）でのプロトン化状態を自動設定するプリセット"""
        return cls(
            protonate_mode="auto_ph",
            ph=7.4,
            partial_charge_model="gasteiger",
        )


@dataclass
class ChargeConfigStore:
    """
    SMILES列名ごとの MoleculeChargeConfig を保持するストア。

    session_state["smiles_charge_configs"] の型として使用。
    per_molecule に分子ごとの個別オーバーライドも格納できる。
    """
    # デフォルト設定（全分子共通）
    default: MoleculeChargeConfig = field(default_factory=MoleculeChargeConfig.default)

    # 分子ごとの個別オーバーライド: SMILES文字列 → MoleculeChargeConfig
    per_molecule: dict[str, MoleculeChargeConfig] = field(default_factory=dict)

    def get_config(self, smiles: str) -> MoleculeChargeConfig:
        """指定SMILESの設定を返す（個別設定 > デフォルト設定の優先順）"""
        return self.per_molecule.get(smiles, self.default)

    def set_per_molecule(self, smiles: str, config: MoleculeChargeConfig) -> None:
        """特定の分子に個別の電荷設定を登録する"""
        self.per_molecule[smiles] = config

    def resolve_charge(self, smiles: str) -> int:
        """
        指定SMILESの実効的な形式電荷を返す。
        auto_charge_from_smiles=True の場合はRDKitから読取る。
        """
        cfg = self.get_config(smiles)
        if cfg.auto_charge_from_smiles:
            return _read_smiles_formal_charge(smiles)
        return cfg.formal_charge

    def resolve_spin(self, smiles: str) -> int:
        """指定SMILESのスピン多重度を返す"""
        return self.get_config(smiles).spin_multiplicity


def _read_smiles_formal_charge(smiles: str) -> int:
    """RDKitを使ってSMILES文字列から形式電荷を読取る。"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.GetFormalCharge(mol)
    except Exception:
        pass
    return 0
