# -*- coding: utf-8 -*-
"""
backend/chem/group_contrib_adapter.py

Joback法（Joback & Reid, 1987）による原子団寄与法で
熱力学物性（沸点・融点・臨界温度等）を推定する記述子化アダプタ。

References:
  K.G. Joback and R.C. Reid, "Estimation of Pure-Component Properties
  from Group-Contributions", Chemical Engineering Communications,
  57(1-6), 233–243 (1987).

Implements: §3.9 化合物特徴量化（原子団寄与法）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# ── RDKit 可用チェック ─────────────────────────────────────
try:
    from rdkit import Chem
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# Joback原子団パラメータ データベース
# ═══════════════════════════════════════════════════════════════════
# 各タプル = (SMARTS, グループ名,
#   ΔTb, ΔTm, ΔTc, ΔPc, ΔVc,
#   ΔHf(298K), ΔGf(298K),
#   Cp_a, Cp_b, Cp_c, Cp_d)
#
# 単位: Tb/Tm/Tc [K], Pc [bar], Vc [cm³/mol],
#        Hf/Gf [kJ/mol], Cp [J/(mol·K)]
# 出典: Joback & Reid (1987), Table 1–4

@dataclass(frozen=True)
class _JobackGroup:
    smarts: str
    label: str
    dTb: float
    dTm: float
    dTc: float
    dPc: float
    dVc: float
    dHf: float
    dGf: float
    cp_a: float
    cp_b: float
    cp_c: float
    cp_d: float


_JOBACK_GROUPS: list[_JobackGroup] = [
    # ── 非環式 C ──
    _JobackGroup("[CX4;!R](-[#1])(-[#1])(-[#1])", "-CH3",
                 23.58, -5.10, 0.0141, -0.0012, 65.0,
                 -76.45, -43.96, 19.5, -8.08e-3, 1.53e-4, -9.67e-8),
    _JobackGroup("[CX4;!R](-[#1])(-[#1])", "-CH2-",
                 22.88, 11.27, 0.0189, 0.0000, 56.0,
                 -20.64, 8.42, -0.909, 9.50e-2, -5.44e-5, 1.19e-8),
    _JobackGroup("[CX4;!R](-[#1])", ">CH-",
                 21.74, 12.64, 0.0164, 0.0020, 41.0,
                 29.89, 58.36, -23.0, 2.04e-1, -2.65e-4, 1.20e-7),
    _JobackGroup("[CX4;!R;$([CX4](-[!#1])(-[!#1])(-[!#1])(-[!#1]))]", ">C<",
                 18.25, 46.43, 0.0067, 0.0043, 27.0,
                 82.23, 116.02, -66.2, 4.27e-1, -6.41e-4, 3.01e-7),
    _JobackGroup("[CX3;!R](=[CX3])", "=CH2",
                 18.18, -4.32, 0.0113, -0.0028, 56.0,
                 -9.63, 3.77, 24.5, -2.71e-2, 1.11e-4, -6.78e-8),
    _JobackGroup("[CX3;!R](=[CX3;!R])", "=CH-",
                 24.96, 8.73, 0.0129, -0.0006, 46.0,
                 37.97, 48.53, 7.87, 2.01e-2, -8.33e-6, 0.0),
    # ── 環式 C ──
    _JobackGroup("[CX4;R](-[#1])(-[#1])", "-CH2- (ring)",
                 27.15, 7.75, 0.0100, 0.0025, 48.0,
                 -26.80, -3.68, -6.03, 8.54e-2, -8.00e-6, -1.80e-8),
    _JobackGroup("[CX4;R](-[#1])", ">CH- (ring)",
                 21.78, 19.88, 0.0122, 0.0004, 38.0,
                 8.67, 40.99, -20.1, 1.62e-1, -1.60e-4, 6.24e-8),
    _JobackGroup("[cX3](:c)(:c)", "=CH- (aromatic)",
                 26.73, 8.13, 0.0082, 0.0011, 41.0,
                 -2.14, 7.37, -8.0, 1.05e-1, -9.63e-5, 3.56e-8),
    _JobackGroup("[cX3](:c)(:c)(-[!#1;!c])", "=C< (aromatic fused)",
                 31.01, 37.02, 0.0143, 0.0008, 32.0,
                 46.43, 54.05, -28.1, 2.08e-1, -3.06e-4, 1.46e-7),
    # ── O ──
    _JobackGroup("[OX2;!R](-[#1])", "-OH (alcohol)",
                 92.88, 44.45, 0.0741, 0.0112, 28.0,
                 -208.04, -189.20, 25.7, -6.91e-2, 1.77e-4, -9.88e-8),
    _JobackGroup("[OX2;!R](-[CX4])(-[CX4])", "-O- (ether)",
                 22.42, 22.23, 0.0168, 0.0015, 18.0,
                 -132.22, -105.00, 25.5, -6.32e-2, 1.11e-4, -5.48e-8),
    _JobackGroup("[CX3](=O)(-[OX2])", "-COO- (ester)",
                 76.34, 36.90, 0.0481, 0.0005, 62.0,
                 -337.92, -301.95, 24.5, 4.02e-2, 4.02e-5, -4.52e-8),
    _JobackGroup("[CX3;!R](=O)(-[#1])", "-CHO (aldehyde)",
                 72.24, 36.90, 0.0379, 0.0030, 82.0,
                 -162.03, -143.48, 30.9, -3.36e-2, 1.60e-4, -9.88e-8),
    _JobackGroup("[CX3;!R](=O)(-[CX4])", ">C=O (ketone)",
                 76.75, 61.20, 0.0481, 0.0005, 62.0,
                 -133.22, -120.50, 6.45, 6.70e-2, -3.57e-5, 2.86e-9),
    _JobackGroup("[CX3](=O)(-[OX2H])", "-COOH (acid)",
                 169.09, 155.50, 0.0791, 0.0077, 89.0,
                 -426.72, -387.87, 24.1, 4.27e-2, 8.04e-5, -6.87e-8),
    # ── N ──
    _JobackGroup("[NX3;!R](-[#1])(-[#1])", "-NH2",
                 73.23, 66.89, 0.0243, 0.0109, 38.0,
                 -22.02, 14.07, 26.9, -4.12e-2, 1.64e-4, -9.76e-8),
    _JobackGroup("[NX3;!R](-[#1])(-[CX4])", ">NH",
                 50.17, 52.66, 0.0295, 0.0077, 35.0,
                 53.47, 89.39, -1.21, 7.62e-2, -4.86e-5, 1.05e-8),
    _JobackGroup("[NX3;!R](-[CX4])(-[CX4])", ">N-",
                 11.74, 48.84, 0.0169, 0.0074, 9.0,
                 31.65, 75.61, -31.1, 2.27e-1, -3.20e-4, 1.46e-7),
    # ── S ──
    _JobackGroup("[SX2;!R](-[#1])", "-SH",
                 63.56, 20.09, 0.0031, 0.0084, 63.0,
                 -17.33, -22.99, 35.3, -7.58e-2, 1.85e-4, -1.03e-7),
    _JobackGroup("[SX2;!R](-[CX4])(-[CX4])", "-S-",
                 68.78, 34.40, 0.0119, 0.0049, 54.0,
                 41.87, 33.12, 19.6, -5.61e-3, 4.02e-5, -2.76e-8),
    # ── ハロゲン ──
    _JobackGroup("[FX1]", "-F",
                 -0.03, -15.78, 0.0111, -0.0057, 27.0,
                 -247.61, -250.83, 26.5, -9.13e-2, 1.91e-4, -1.03e-7),
    _JobackGroup("[ClX1]", "-Cl",
                 38.13, 13.55, 0.0105, -0.0049, 58.0,
                 -71.55, -64.31, 33.3, -9.63e-2, 1.87e-4, -9.96e-8),
    _JobackGroup("[BrX1]", "-Br",
                 66.86, 43.43, 0.0133, 0.0057, 71.0,
                 -29.48, -38.06, 28.6, -6.49e-2, 1.36e-4, -7.45e-8),
    _JobackGroup("[IX1]", "-I",
                 93.84, 41.69, 0.0068, -0.0034, 95.0,
                 21.06, 5.74, 32.7, -6.59e-2, 1.36e-4, -7.45e-8),
    # ── C≡ ──
    _JobackGroup("[CX2]#[CX2]", "C≡C",
                 36.90, 25.00, 0.0078, -0.0011, 36.0,
                 115.51, 109.82, 7.87, 2.01e-2, -8.33e-6, 0.0),
    # ── C≡N ──
    _JobackGroup("[CX2]#[NX1]", "-CN (nitrile)",
                 125.66, 59.89, 0.0180, 0.0031, 36.0,
                 23.61, 58.60, 36.5, -7.33e-2, 1.84e-4, -1.03e-7),
    # ── NO2 ──
    _JobackGroup("[NX3](=O)(=O)", "-NO2",
                 152.54, 127.24, 0.0437, 0.0064, 58.0,
                 -66.57, -16.83, 25.9, -3.74e-3, 1.29e-4, -8.88e-8),
]

# SMARTS → コンパイル済み パターン のキャッシュ
_COMPILED_PATTERNS: list[tuple[Any, _JobackGroup]] | None = None


def _get_compiled_patterns() -> list[tuple[Any, _JobackGroup]]:
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = []
        for grp in _JOBACK_GROUPS:
            pat = Chem.MolFromSmarts(grp.smarts)
            if pat is not None:
                _COMPILED_PATTERNS.append((pat, grp))
            else:
                logger.warning(f"SMARTS パターンのコンパイル失敗: {grp.smarts} ({grp.label})")
    return _COMPILED_PATTERNS


# ═══════════════════════════════════════════════════════════════════
# Joback法 物性推定関数
# ═══════════════════════════════════════════════════════════════════

def _count_groups(mol: Any) -> dict[str, int]:
    """分子中のJoback原子団のカウント数を返す。"""
    patterns = _get_compiled_patterns()
    counts: dict[str, int] = {}
    for pat, grp in patterns:
        matches = mol.GetSubstructMatches(pat)
        if matches:
            counts[grp.label] = len(matches)
    return counts


def _estimate_properties(mol: Any) -> dict[str, float]:
    """
    Joback法で推定した熱力学物性を辞書として返す。

    Joback法の計算式:
      Tb = 198.2 + Σ(nᵢ ΔTbᵢ)           [K]
      Tm = 122.5 + Σ(nᵢ ΔTmᵢ)           [K]
      Tc = Tb / (0.584 + 0.965 Σ(nᵢ ΔTcᵢ) - (Σ(nᵢ ΔTcᵢ))²)  [K]
      Pc = (0.113 + 0.0032 nₐ - Σ(nᵢ ΔPcᵢ))⁻²  [bar]
      Vc = 17.5 + Σ(nᵢ ΔVcᵢ)            [cm³/mol]
      ΔHf = 68.29 + Σ(nᵢ ΔHfᵢ)          [kJ/mol]
      ΔGf = 53.88 + Σ(nᵢ ΔGfᵢ)          [kJ/mol]

    References:
      Joback & Reid (1987), Eqs. 1–7
    """
    n_atoms = mol.GetNumAtoms()
    patterns = _get_compiled_patterns()

    sum_tb = sum_tm = sum_tc = sum_pc = sum_vc = 0.0
    sum_hf = sum_gf = 0.0
    sum_cp_a = sum_cp_b = sum_cp_c = sum_cp_d = 0.0
    total_group_count = 0

    for pat, grp in patterns:
        n = len(mol.GetSubstructMatches(pat))
        if n == 0:
            continue
        total_group_count += n
        sum_tb += n * grp.dTb
        sum_tm += n * grp.dTm
        sum_tc += n * grp.dTc
        sum_pc += n * grp.dPc
        sum_vc += n * grp.dVc
        sum_hf += n * grp.dHf
        sum_gf += n * grp.dGf
        sum_cp_a += n * grp.cp_a
        sum_cp_b += n * grp.cp_b
        sum_cp_c += n * grp.cp_c
        sum_cp_d += n * grp.cp_d

    if total_group_count == 0:
        return {}

    # Tb (沸点)
    Tb = 198.2 + sum_tb

    # Tm (融点)
    Tm = 122.5 + sum_tm

    # Tc (臨界温度) — Tb依存
    denom = 0.584 + 0.965 * sum_tc - sum_tc ** 2
    Tc = Tb / denom if abs(denom) > 1e-10 else np.nan

    # Pc (臨界圧力) — 原子数依存
    inv_pc = 0.113 + 0.0032 * n_atoms - sum_pc
    Pc = inv_pc ** (-2) if abs(inv_pc) > 1e-10 else np.nan

    # Vc (臨界体積)
    Vc = 17.5 + sum_vc

    # ΔHf (標準生成エンタルピー)
    Hf = 68.29 + sum_hf

    # ΔGf (標準生成ギブスエネルギー)
    Gf = 53.88 + sum_gf

    # Cp (定圧熱容量, 298K) = a + b·T + c·T² + d·T³
    T = 298.15
    Cp_298 = sum_cp_a + sum_cp_b * T + sum_cp_c * T ** 2 + sum_cp_d * T ** 3

    return {
        "joback_Tb": round(Tb, 2),
        "joback_Tm": round(Tm, 2),
        "joback_Tc": round(Tc, 2) if not np.isnan(Tc) else np.nan,
        "joback_Pc": round(Pc, 2) if not np.isnan(Pc) else np.nan,
        "joback_Vc": round(Vc, 2),
        "joback_Hf": round(Hf, 2),
        "joback_Gf": round(Gf, 2),
        "joback_Cp298": round(Cp_298, 2),
        "joback_n_groups": total_group_count,
    }


# ═══════════════════════════════════════════════════════════════════
# アダプター
# ═══════════════════════════════════════════════════════════════════

class GroupContribAdapter(BaseChemAdapter):
    """
    Joback法による原子団寄与法の記述子アダプタ。

    RDKitのみ依存。外部バイナリ不要。
    SMILES → RDKit Mol → SMARTS原子団マッチング → 物性推定。
    """

    @property
    def name(self) -> str:
        return "group_contrib"

    @property
    def description(self) -> str:
        return "Joback法による原子団寄与法での熱力学物性推定（沸点・融点・臨界温度等）"

    def is_available(self) -> bool:
        return _RDKIT_AVAILABLE

    def compute(
        self,
        smiles_list: list[str],
        **kwargs: Any,
    ) -> DescriptorResult:
        """
        SMILES → Joback法で熱力学物性を推定する。

        Returns:
            DescriptorResult: 各SMILES×9記述子のDataFrame
        """
        self._require_available()
        logger.info(f"🧪 GroupContrib (Joback法) 熱力学物性計算を開始します（{len(smiles_list)}分子）...")

        results: list[dict[str, float]] = []
        failed: list[int] = []

        for idx, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"SMILES変換失敗: {smi}")
                mol = Chem.AddHs(mol)
                props = _estimate_properties(mol)
                if not props:
                    raise ValueError(f"原子団マッチなし: {smi}")
                results.append(props)
            except Exception as e:
                logger.warning(f"GroupContrib計算失敗 (idx={idx}, smi={smi}): {e}")
                failed.append(idx)
                results.append({k: np.nan for k in [
                    "joback_Tb", "joback_Tm", "joback_Tc", "joback_Pc",
                    "joback_Vc", "joback_Hf", "joback_Gf",
                    "joback_Cp298", "joback_n_groups",
                ]})

        df = pd.DataFrame(results)
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
        )

    def get_descriptor_names(self) -> list[str]:
        return [
            "joback_Tb", "joback_Tm", "joback_Tc", "joback_Pc",
            "joback_Vc", "joback_Hf", "joback_Gf",
            "joback_Cp298", "joback_n_groups",
        ]

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata("joback_Tb", "沸点 [K] (Joback法)", is_count=False),
            DescriptorMetadata("joback_Tm", "融点 [K] (Joback法)", is_count=False),
            DescriptorMetadata("joback_Tc", "臨界温度 [K] (Joback法)", is_count=False),
            DescriptorMetadata("joback_Pc", "臨界圧力 [bar] (Joback法)", is_count=False),
            DescriptorMetadata("joback_Vc", "臨界体積 [cm³/mol] (Joback法)", is_count=False),
            DescriptorMetadata("joback_Hf", "標準生成エンタルピー [kJ/mol]", is_count=False),
            DescriptorMetadata("joback_Gf", "標準生成ギブスエネルギー [kJ/mol]", is_count=False),
            DescriptorMetadata("joback_Cp298", "定圧熱容量 298K [J/(mol·K)]", is_count=False),
            DescriptorMetadata("joback_n_groups", "マッチした原子団数", is_count=True),
        ]
