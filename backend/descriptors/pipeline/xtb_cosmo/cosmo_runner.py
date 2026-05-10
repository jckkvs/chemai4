"""
backend/descriptors/pipeline/xtb_cosmo/cosmo_runner.py

COSMO-RS（COSMOthermまたはOpenCOSMO）計算を実行するラッパークラス
"""
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

from .config import CosmoConfig, CosmoSolvent

logger = logging.getLogger(__name__)

class CosmoRunner:
    SOLVENT_MAP = {
        CosmoSolvent.WATER: "water",
        CosmoSolvent.METHANOL: "methanol",
        CosmoSolvent.ETHANOL: "ethanol",
        CosmoSolvent.ACETONITRILE: "acetonitrile",
        CosmoSolvent.DMSO: "dmso",
        CosmoSolvent.CHLOROFORM: "chloroform",
        CosmoSolvent.HEXANE: "n-hexane",
    }
    
    UNIT_CONVERSION = {
        "kJ/mol": 1.0,
        "kcal/mol": 0.239006,
        "eV": 0.010364,
    }
    
    def __init__(self, config: CosmoConfig):
        self.config = config
        self._cosmo_path = config.cosmo_engine_path
        
    def run_single(self, cosmo_file: Path, molecule_id: Optional[str] = None, work_dir: Optional[Path] = None) -> Tuple[bool, Dict, List[str]]:
        warnings_list = []
        result = {}
        
        try:
            if work_dir is None:
                work_dir = Path(tempfile.mkdtemp(prefix="chemai_cosmo_"))
                cleanup = True
            else:
                cleanup = False
                
            mol_id = molecule_id or cosmo_file.stem
            
            # Here we generate an input file. We use a format typical for COSMOtherm.
            # If OpenCOSMO is used and has a different syntax, this can be conditionally branched.
            inp_content = self._generate_input_file(
                cosmo_file=cosmo_file,
                solvent=self.SOLVENT_MAP[self.config.solvent],
                temperature=self.config.temperature,
                pressure=self.config.pressure,
                calculate_delta_g=self.config.calculate_delta_g,
                calculate_activity=self.config.calculate_activity,
                calculate_solubility=self.config.calculate_solubility,
            )
            inp_path = work_dir / f"{mol_id}.inp"
            inp_path.write_text(inp_content)
            
            cmd = [
                self._cosmo_path,
                "-inp", str(inp_path),
                "-out", str(work_dir / f"{mol_id}.out"),
            ]
            if self.config.parameter_file:
                cmd.extend(["-param", self.config.parameter_file])
                
            try:
                subprocess.run(
                    cmd, cwd=work_dir, timeout=self.config.timeout_seconds,
                    capture_output=True, text=True, check=True
                )
            except Exception as e:
                return False, {}, [f"COSMO execution error: {e}"]
                
            out_path = work_dir / f"{mol_id}.out"
            if not out_path.exists():
                return False, {}, ["Output file was not generated"]
                
            result = self._parse_cosmo_output(out_path)
            
            if self.config.calculate_delta_g and "delta_g_solv_kj_mol" in result:
                factor = self.UNIT_CONVERSION[self.config.energy_unit]
                result[f"delta_g_solv_{self.config.energy_unit.replace('/', '_')}"] = result["delta_g_solv_kj_mol"] * factor
                
            result["success"] = True
            result["molecule_id"] = mol_id
            result["solvent"] = self.config.solvent.value
            result["temperature_k"] = self.config.temperature
            
            return True, result, warnings_list
            
        except Exception as e:
            logger.exception(f"COSMO calculation error: {e}")
            return False, {}, [str(e)]
        finally:
            if cleanup and work_dir.exists():
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)

    def _generate_input_file(self, cosmo_file: Path, solvent: str, temperature: float, pressure: float, calculate_delta_g: bool, calculate_activity: bool, calculate_solubility: bool) -> str:
        lines = [
            "$title", f"chemai2 calculation for {cosmo_file.stem}", "$end", "",
            "$compound", f"  name = {cosmo_file.stem}", f"  file = {cosmo_file.name}", "$end", "",
            "$solvent", f"  name = {solvent}", "$end", "",
            "$condition", f"  temperature = {temperature}", f"  pressure = {pressure}", "$end",
        ]
        if calculate_delta_g:
            lines.extend(["", "$property", "  solvation_free_energy", "$end"])
        if calculate_activity:
            lines.extend(["", "$property", "  activity_coefficient", "$end"])
        if calculate_solubility:
            lines.extend(["", "$property", "  solubility", "$end"])
        return "\n".join(lines)
        
    def _parse_cosmo_output(self, out_path: Path) -> dict:
        result = {}
        content = out_path.read_text()
        
        # Parse for Delta G
        delta_g_match = re.search(r'Delta\s+G\s*\(\s*solvation\s*\)\s*=\s*([-\d.]+)\s*(kJ/mol|kcal/mol|eV)', content, re.IGNORECASE)
        # If open cosmo outputs slightly differently, we can add more regexes here
        if not delta_g_match:
            # Fallback regex for OpenCOSMO or alternative formats
            delta_g_match = re.search(r'solvation energy.*?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(kJ/mol|kcal/mol|eV)', content, re.IGNORECASE)
            
        if delta_g_match:
            value = float(delta_g_match.group(1))
            unit = delta_g_match.group(2).lower()
            if "kcal" in unit: value /= 0.239006
            elif "ev" in unit: value /= 0.010364
            result["delta_g_solv_kj_mol"] = value
            
        error_lines = [line.strip() for line in content.split('\n') if any(kw in line.upper() for kw in ['ERROR', 'WARNING', 'FAILED'])]
        if error_lines:
            result["cosmo_warnings"] = error_lines[:10]
            
        return result
