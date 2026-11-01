from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_acl_compat_surface_matches_repository_usage() -> None:
    init_source = (ROOT / "acl" / "__init__.py").read_text()
    runtime_source = (ROOT / "acl" / "rt.py").read_text()

    assert "def get_soc_name()" in init_source
    assert "def get_device_count(" in runtime_source
    assert "def get_mem_info(" in runtime_source
    assert "def memcpy(" in runtime_source
    assert "QWEN3_CANN85_HOME" in runtime_source
    assert "aclrtGetSocName" in runtime_source
    assert "aclrtGetDeviceCount" in runtime_source
    assert "aclrtGetMemInfo" in runtime_source
    assert "aclrtMemcpy" in runtime_source


def test_no_foreign_pyacl_is_added_to_runtime() -> None:
    activation = (ROOT / "tools" / "activate_cann85_a3_torch29.sh").read_text()

    assert "/ascend-toolkit/8.3" not in activation
    assert "/cann_master_9.1" not in activation
