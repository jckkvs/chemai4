"""
backend/descriptors/pipeline/xtb_cosmo/config.py

XTB計算およびCOSMO-RS（またはOpenCOSMO）向けの設定データクラス。
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum
import os

class XTBMethod(Enum):
    """XTB計算手法"""
    GFN2 = "gfn2"
    GFN1 = "gfn1"
    GFN0 = "gfn0"

class CosmoSolvent(Enum):
    """COSMO-RS対応溶媒（一部）"""
    WATER = "water"
    METHANOL = "methanol"
    ETHANOL = "ethanol"
    ACETONITRILE = "acetonitrile"
    DMSO = "dmso"
    CHLOROFORM = "chloroform"
    HEXANE = "hexane"

@dataclass
class XTBConfig:
    """XTB計算設定"""
    method: XTBMethod = XTBMethod.GFN2
    charge: int = 0
    multiplicity: int = 1
    optimization: bool = True
    frequency: bool = False
    solvation_model: Optional[str] = None
    max_iterations: int = 200
    convergence_tolerance: float = 1e-4
    timeout_seconds: int = 300
    n_threads: int = 1
    memory_mb: int = 2048
    
    output_cosmo: bool = True
    output_energies: bool = True
    output_coordinates: bool = False
    
    retry_on_failure: bool = True
    max_retries: int = 2
    fallback_method: Optional[XTBMethod] = XTBMethod.GFN1

@dataclass
class CosmoConfig:
    """COSMO-RS/OpenCOSMO計算設定"""
    solvent: CosmoSolvent = CosmoSolvent.WATER
    temperature: float = 298.15        # [K]
    pressure: float = 1.01325          # [bar]
    
    calculate_delta_g: bool = True
    calculate_activity: bool = False
    calculate_solubility: bool = False
    
    energy_unit: Literal["kJ/mol", "kcal/mol", "eV"] = "kJ/mol"
    
    parameter_file: Optional[str] = field(
        default_factory=lambda: os.getenv("COSMO_PARAM_FILE")
    )
    timeout_seconds: int = 60
    
    # cosmothermかopencosmoかユーザーのバイナリパスをここで設定可能
    cosmo_engine_path: str = field(
        default_factory=lambda: os.getenv("COSMO_ENGINE_PATH", "opencosmo") # デフォルトを opencosmo に
    )

@dataclass
class PipelineConfig:
    """連携パイプライン全体設定"""
    xtb: XTBConfig = field(default_factory=XTBConfig)
    cosmo: CosmoConfig = field(default_factory=CosmoConfig)
    
    generate_3d: bool = True
    conformer_ensemble: int = 1
    rmsd_threshold: float = 0.5
    
    n_jobs: int = 1
    cache_enabled: bool = True
    cache_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.chemai/cache/xtb_cosmo")
    )
    cache_ttl_hours: int = 168
    
    continue_on_error: bool = True
    warning_threshold: int = 10
    version: str = "1.0.0"
