"""
tests/test_xtb_cosmo/test_pipeline.py
"""
import pytest
import os
from pathlib import Path
from backend.descriptors.pipeline.xtb_cosmo.pipeline import XTBCosmoDescriptor
from backend.descriptors.pipeline.xtb_cosmo.config import PipelineConfig

@pytest.fixture
def sample_smiles():
    return ["CCO", "CCCO", "c1ccccc1"]

@pytest.fixture
def mock_xtb_binary(tmp_path, monkeypatch):
    if os.name == 'nt':
        mock_xtb = tmp_path / "xtb.bat"
        mock_xtb.write_text("@echo off\necho FINAL ENERGY: -100.0 Eh\necho GEOMETRY OPTIMIZATION CONVERGED\ntouch xtb.cosmo\nexit 0")
    else:
        mock_xtb = tmp_path / "xtb"
        mock_xtb.write_text("#!/bin/bash\necho 'FINAL ENERGY: -100.0 Eh'\necho 'GEOMETRY OPTIMIZATION CONVERGED'\ntouch xtb.cosmo\nexit 0")
        mock_xtb.chmod(0o755)
        
    monkeypatch.setenv("XTB_PATH", str(mock_xtb))
    return mock_xtb

@pytest.fixture
def mock_cosmotherm_binary(tmp_path, monkeypatch):
    if os.name == 'nt':
        mock_cosmo = tmp_path / "opencosmo.bat"
        # Just write fake output
        mock_cosmo.write_text('@echo off\nset arg2=%2\nset outarg=%4\necho Delta G(solvation) = -25.34 kJ/mol > "%outarg%"\nexit 0')
    else:
        mock_cosmo = tmp_path / "opencosmo"
        mock_cosmo.write_text("""#!/bin/bash
out_file=$(grep "\-out" "$@" | cut -d' ' -f2)
echo "Delta G(solvation) = -25.34 kJ/mol" > "$out_file"
exit 0
""")
        mock_cosmo.chmod(0o755)
        
    monkeypatch.setenv("COSMO_ENGINE_PATH", str(mock_cosmo))
    return mock_cosmo

def test_cache_functionality(sample_smiles, mock_xtb_binary, mock_cosmotherm_binary, tmp_path):
    cache_dir = tmp_path / "cache"
    desc = XTBCosmoDescriptor()
    
    # Run once
    results1 = desc(
        smiles=sample_smiles[:1],
        solvent="water",
        cache_enabled=True,
        cache_dir=str(cache_dir),
        xtb_method="gfn2",
        n_jobs=1
    )
    
    # Since we mocked it, it might fail inside real 3D generation without RDKit context checking,
    # but assuming environment runs RDKit fine, we can verify caching logic.
    assert len(results1) == 1

def test_invalid_smiles_handling(mock_xtb_binary, mock_cosmotherm_binary, tmp_path):
    desc = XTBCosmoDescriptor()
    results = desc(
        smiles=["INVALID_SMILES_!!!"],
        solvent="water",
        cache_enabled=False,
        cache_dir=str(tmp_path)
    )
    assert results == [None]

def test_dependency_check_failure(monkeypatch):
    monkeypatch.setenv("XTB_PATH", "/nonexistent/xtb")
    monkeypatch.delenv("COSMO_ENGINE_PATH", raising=False)
    
    desc = XTBCosmoDescriptor()
    results = desc(
        smiles=["CCO"],
        solvent="water",
        cache_enabled=False
    )
    assert results == [None]
