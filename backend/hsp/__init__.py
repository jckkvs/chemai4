# -*- coding: utf-8 -*-
"""
backend/hsp/__init__.py

Hansen Solubility Parameters (HSP) 計算モジュール。
HSPiPy + ML予測 + COSMO-RS連携。
"""
from __future__ import annotations

try:
    from backend.hsp.hsp_calculator import HSPCalculator
except ImportError:
    HSPCalculator = None  # type: ignore[assignment,misc]

try:
    from backend.hsp.hsp_predictor import HSPPredictor
except ImportError:
    HSPPredictor = None  # type: ignore[assignment,misc]

__all__ = ["HSPCalculator", "HSPPredictor"]
