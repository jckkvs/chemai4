@echo off
chcp 65001 > nul
echo ==========================================
echo Interpretability Libraries Installer
echo ==========================================
echo.

echo [1/3] Installing sage-importance (SAGE global feature importance)...
pip install sage-importance
if %ERRORLEVEL% neq 0 (
    echo [WARN] sage-importance install failed. SAGE analysis will be unavailable.
) else (
    echo [OK] sage-importance installed.
)

echo.
echo [2/3] Installing shapiq (Shapley Interactions / SRI decomposition)...
pip install shapiq
if %ERRORLEVEL% neq 0 (
    echo [WARN] shapiq install failed. Shapley Interaction analysis will be unavailable.
) else (
    echo [OK] shapiq installed.
)

echo.
echo [3/3] Verifying installations...
python -c "import sage; print('sage-importance: OK')" 2>nul || echo sage-importance: UNAVAILABLE
python -c "import shapiq; print('shapiq:', shapiq.__version__)" 2>nul || echo shapiq: UNAVAILABLE
python -c "import shap; print('shap:', shap.__version__)" 2>nul || echo shap: UNAVAILABLE

echo.
echo Done. Re-start Streamlit to apply changes.
pause
