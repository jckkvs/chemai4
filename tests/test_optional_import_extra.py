"""
tests/test_optional_import_extra.py

optional_import.py の低カバレッジ部分を補うテスト。
safe_import, is_available, require, get_availability_report,
probe_all_optional_libraries を網羅。
"""
from __future__ import annotations

import pytest

from backend.utils.optional_import import (
    safe_import,
    is_available,
    require,
    get_availability_report,
    probe_all_optional_libraries,
    _availability_cache,
)


# ============================================================
# safe_import
# ============================================================

class TestSafeImport:
    def test_import_existing_module(self):
        mod = safe_import("os")
        assert mod is not None
        assert hasattr(mod, "path")

    def test_import_with_alias(self):
        mod = safe_import("os", alias="os_alias")
        assert mod is not None
        assert is_available("os_alias")

    def test_import_nonexistent(self):
        mod = safe_import("totally_nonexistent_module_xyz123")
        assert mod is None

    def test_import_nonexistent_with_alias(self):
        mod = safe_import("totally_nonexistent_xyz", alias="fake_lib")
        assert mod is None
        assert not is_available("fake_lib")

    def test_import_submodule(self):
        mod = safe_import("os.path")
        assert mod is not None


# ============================================================
# is_available
# ============================================================

class TestIsAvailable:
    def test_available(self):
        safe_import("sys", alias="sys_test")
        assert is_available("sys_test") is True

    def test_not_available(self):
        safe_import("nonexistent_xyz_987", alias="nx_test")
        assert is_available("nx_test") is False

    def test_unknown_key(self):
        assert is_available("never_imported_xyz") is False


# ============================================================
# require
# ============================================================

class TestRequire:
    def test_require_available(self):
        safe_import("json", alias="json_test")
        # Should not raise
        require("json_test")

    def test_require_unavailable(self):
        safe_import("nonexistent_xyz_456", alias="missing_test")
        with pytest.raises(RuntimeError, match="インストールされていません"):
            require("missing_test")

    def test_require_with_feature(self):
        safe_import("nonexistent_xyz_789", alias="missing2")
        with pytest.raises(RuntimeError, match="機能.*my_feature"):
            require("missing2", feature="my_feature")


# ============================================================
# get_availability_report
# ============================================================

class TestGetReport:
    def test_returns_dict(self):
        report = get_availability_report()
        assert isinstance(report, dict)

    def test_includes_probed(self):
        safe_import("math", alias="math_test")
        report = get_availability_report()
        assert "math_test" in report
        assert report["math_test"] is True


# ============================================================
# probe_all_optional_libraries
# ============================================================

class TestProbeAll:
    def test_probe_returns_dict(self):
        report = probe_all_optional_libraries()
        assert isinstance(report, dict)
        assert len(report) > 0

    def test_probe_includes_known_libs(self):
        report = probe_all_optional_libraries()
        # These should be in the report (either True or False)
        assert "lightgbm" in report
        assert "optuna" in report
        assert "rdkit" in report
