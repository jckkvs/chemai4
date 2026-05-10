"""
backend/descriptors/pipeline/xtb_cosmo/pipeline.py

XTBCosmoDescriptor クラス。
"""
from typing import List, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...base import DescriptorFunction
from .config import PipelineConfig, XTBConfig, CosmoConfig
from .xtb_runner import XTBRunner
from .cosmo_runner import CosmoRunner
from .file_manager import TempFileManager
from ..cache.filesystem import FileSystemCache

logger = logging.getLogger(__name__)

class XTBCosmoDescriptor(DescriptorFunction):
    """
    XTB-COSMO-RS Solvation Energy descriptor.
    
    :::chemai-descriptor
    name: XTB-COSMO-RS Solvation Energy
    engine: xtb+cosmo-rs
    category: 溶解・溶媒
    description: XTBで3D構造最適化後、COSMO-RS/OpenCOSMOで溶媒和自由エネルギーを計算
    params:
      solvent:
        type: str
        default: water
        description: 溶媒種
        options: [water, methanol, ethanol, acetonitrile, dmso, chloroform, hexane]
      temperature:
        type: float
        default: 298.15
        min: 200
        max: 500
      xtb_method:
        type: str
        default: gfn2
        options: [gfn2, gfn1, gfn0]
      n_jobs:
        type: int
        default: 1
      cache_enabled:
        type: bool
        default: true
    :::
    """
    
    def __init__(self):
        self._config: Optional[PipelineConfig] = None
        self._cache: Optional[FileSystemCache] = None
        
    @property
    def metadata(self) -> dict:
        return {} # Will be automatically evaluated from docstring by loaders if not explicitly set
        
    def __call__(
        self,
        smiles: List[str],
        solvent: str = "water",
        temperature: float = 298.15,
        xtb_method: str = "gfn2",
        n_jobs: int = 1,
        cache_enabled: bool = True,
        **kwargs
    ) -> List[Optional[float]]:

        from .config import CosmoSolvent, XTBMethod

        # Separate kwargs into XTBConfig and PipelineConfig params
        xtb_kwargs = {}
        pipeline_kwargs = {}
        xtb_fields = {f.name for f in XTBConfig.__dataclass_fields__.values()}
        pipeline_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}

        for k, v in kwargs.items():
            if k in xtb_fields:
                xtb_kwargs[k] = v
            elif k in pipeline_fields:
                pipeline_kwargs[k] = v

        # 1. Config Object Initialization
        self._config = PipelineConfig(
            xtb=XTBConfig(method=XTBMethod(xtb_method), **xtb_kwargs),
            cosmo=CosmoConfig(
                solvent=CosmoSolvent(solvent),
                temperature=temperature,
            ),
            n_jobs=n_jobs,
            cache_enabled=cache_enabled,
            **pipeline_kwargs
        )
        
        if cache_enabled:
            self._cache = FileSystemCache(self._config.cache_dir)
            
        success, msg = self._check_dependencies()
        if not success:
            logger.warning(f"Dependency check failed: {msg}")
            # Do not completely blow up, return None as indicated
            return [None] * len(smiles)
            
        results = [None] * len(smiles)
        warnings_collected = []
        
        with TempFileManager(keep_on_error=self._config.continue_on_error) as temp_mgr, ThreadPoolExecutor(max_workers=n_jobs) as executor:
            future_to_idx = {}
            for idx, smi in enumerate(smiles):
                if cache_enabled and self._cache:
                    cache_key = self._make_cache_key(smi, solvent, temperature, xtb_method)
                    cached = self._cache.get(cache_key)
                    if cached is not None:
                        results[idx] = cached
                        continue
                        
                future = executor.submit(
                    self._run_single_molecule,
                    smi, temp_mgr, idx
                )
                future_to_idx[future] = idx
                
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    success, value, warnings = future.result()
                    if success:
                        results[idx] = value
                        if cache_enabled and self._cache:
                            smi = smiles[idx]
                            cache_key = self._make_cache_key(smi, solvent, temperature, xtb_method)
                            self._cache.set(cache_key, value)
                    if warnings:
                        warnings_collected.extend(warnings)
                except Exception as e:
                    logger.error(f"Error calculating mol {idx}: {e}")
                    warnings_collected.append(f"Mol#{idx}: {type(e).__name__}: {str(e)}")
                    
        if warnings_collected and len(warnings_collected) <= self._config.warning_threshold:
            for w in warnings_collected:
                logger.warning(w)
        elif warnings_collected:
            logger.warning(f"{len(warnings_collected)} warnings generated during XTBCosmo calculation.")
            
        return results
        
    def _run_single_molecule(self, smiles: str, temp_mgr: TempFileManager, idx: int):
        warnings = []
        
        xtb_runner = XTBRunner(self._config.xtb)
        xtb_success, xtb_result, xtb_warnings = xtb_runner.run_single(
            smiles,
            work_dir=temp_mgr.create_subdir(f"xtb_{idx}"),
            molecule_id=f"mol_{idx:04d}"
        )
        warnings.extend(xtb_warnings)
        
        if not xtb_success or "cosmo_file" not in xtb_result:
            return False, None, warnings + ["XTB calculation or .cosmo generation failed."]
            
        cosmo_runner = CosmoRunner(self._config.cosmo)
        cosmo_success, cosmo_result, cosmo_warnings = cosmo_runner.run_single(
            Path(xtb_result["cosmo_file"]),
            molecule_id=f"mol_{idx:04d}",
            work_dir=temp_mgr.create_subdir(f"cosmo_{idx}")
        )
        warnings.extend(cosmo_warnings)
        
        if not cosmo_success:
            return False, None, warnings + ["COSMO calculation failed."]
            
        delta_g = cosmo_result.get("delta_g_solv_kj_mol")
        if delta_g is None:
            return False, None, warnings + ["Solvation free energy not outputted."]
            
        return True, delta_g, warnings

    def _make_cache_key(self, smiles: str, solvent: str, temperature: float, xtb_method: str) -> str:
        import hashlib
        key_str = f"{smiles}|{solvent}|{temperature}|{xtb_method}|{self._config.version}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _check_dependencies(self) -> tuple[bool, str]:
        import shutil
        import os
        
        if not shutil.which("xtb") and not os.getenv("XTB_PATH"):
            return False, "xtb binary not found"
            
        cosmo_path = self._config.cosmo.cosmo_engine_path
        if not shutil.which(cosmo_path) and not os.getenv("COSMO_ENGINE_PATH"):
            # Don't strictly fail right now, it's possible it is resolved later down the path.
            # but log warning
            pass
            
        return True, "OK"
