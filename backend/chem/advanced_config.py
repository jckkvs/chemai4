"""
backend/chem/advanced_config.py

RDKit 3D構造生成・xTB計算・ML特徴量抽出の高度設定 dataclass 群。

既存の charge_config.py (MoleculeChargeConfig / ChargeConfigStore) とは独立。
こちらは構造生成パイプラインの計算パラメータとフィーチャーエンジニアリング
オプションに特化する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ConformerGenerationConfig:
    """RDKitによる3D構造生成の設定。"""
    num_conformers: int = 10           # 生成する立体配座数
    energy_window: float = 10.0        # kcal/mol以内のconformerを保持
    rmsd_cutoff: float = 0.5           # Å, 重複排除の閾値
    force_field: Literal["MMFF94", "UFF", "ETKDG"] = "ETKDG"
    optimize_geometry: bool = True


@dataclass
class XTBAdvancedConfig:
    """xTB計算の高度パラメータ。"""
    # 電子状態
    charge: Optional[int] = None       # None=SMILESから自動推定
    spin_multiplicity: int = 1         # 1=閉殻, 2=二重項, 3=三重項...

    # 計算設定
    gfn_version: Literal[0, 1, 2] = 2  # GFN0/1/2-xTB
    calc_type: Literal["opt", "sp", "freq"] = "opt"
    convergence: Literal[
        "crude", "sloppy", "loose", "normal", "tight", "vtight"
    ] = "normal"

    # 溶媒・環境
    solvent_model: Literal["none", "alpb", "cpcm"] = "none"
    solvent_name: Optional[str] = None  # "water", "methanol" 等
    temperature: float = 298.15        # K
    pressure: float = 1.0              # atm

    # 出力制御
    output_properties: list[str] = field(default_factory=lambda: [
        "energy", "homo", "lumo", "dipole", "charges", "frequencies",
    ])
    save_wavefunction: bool = False    # .wfnファイル出力（後処理用）

    # パフォーマンス
    parallel_threads: int = 4
    memory_limit_mb: int = 2048
    timeout_seconds: int = 300


@dataclass
class DescriptorExtractionConfig:
    """ML特徴量抽出の設定。"""
    # 量子化学由来
    include_orbital_features: bool = True      # HOMO/LUMO/ギャップ
    include_charge_features: bool = True       # Mulliken/CM5電荷統計
    include_vibrational_features: bool = False  # 振動数由来（freq計算時）
    include_thermo_features: bool = False       # 熱力学量（freq計算時）

    # 幾何学的
    include_3d_descriptors: bool = True        # 3D形状記述子
    include_surface_features: bool = False     # 分子表面積・体積

    # 電子構造
    include_fukui_indices: bool = False        # 福井関数（反応性指標）
    include_hardness_softness: bool = True     # 化学的硬さ・軟らかさ

    # フィンガープリント連携
    combine_with_2d_fps: bool = True           # 2Dフィンガープリントとの併用
    fps_types: list[str] = field(default_factory=lambda: [
        "morgan", "rdkit", "maccs",
    ])
