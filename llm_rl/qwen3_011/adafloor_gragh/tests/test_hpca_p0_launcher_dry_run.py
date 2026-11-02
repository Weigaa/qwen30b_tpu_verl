import hashlib
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "run_hpca_p0_long_horizon_minimal.sh"


@pytest.fixture
def fixture_env(tmp_path):
    repo = tmp_path / "paper"
    repo.mkdir()
    source = repo / "probe.py"
    source.write_text("# fixture; not imported\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(hashlib.sha256(source.read_bytes()).hexdigest()
                        + "  /workspace/cann-recipes-train/llm_rl/qwen3_011/qwen3_shrink_aware/probe.py\n")
    public = tmp_path / "public_epoch0"
    public.mkdir()
    (public / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").touch()
    (public / "reuse.env").write_text("# not sourced by dry-run\n")
    model = public / "epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3"
    dist = model / "global_step_5/actor/dist_ckpt"
    dist.mkdir(parents=True)
    (model / "latest_checkpointed_iteration.txt").write_text("5\n")
    (dist / "metadata.json").write_text("{}\n")
    for i in range(32):
        (dist / f"{i}.distcp").write_bytes(b"fixture")
    env = {**os.environ, "PAPER_EAGER_REPO": str(repo), "PAPER_EAGER_MANIFEST": str(manifest),
           "COMMON_EPOCH0_ROOT": str(public), "HPCA_P0_QUALITY_OUTPUT_ROOT": str(tmp_path / "output"),
           "HPCA_P0_QUALITY_SEED": "505", "HPCA_P0_QUALITY_TOTAL_EPOCHS": "5",
           "HPCA_P0_QUALITY_VARIANTS": "vanilla adafloor_n_f2 adafloor_p_f4", "PAPER_EAGER_REQUIRE_CLEAN": "0"}
    return env


def test_dry_run_validates_exact_protocol_without_creating_output(fixture_env):
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=fixture_env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    for token in ("dry_run=PASS", "seed=505", "epochs=1-4", "steps_per_epoch=5",
                  "responses_per_step=512", "graph_mode=disabled", "actor_frozen=false"):
        assert token in result.stdout
    assert not Path(fixture_env["HPCA_P0_QUALITY_OUTPUT_ROOT"]).exists()


@pytest.mark.parametrize("fault", ["public_output", "symlink_public_output", "bad_seed", "bad_epochs", "bad_variant", "missing_shard"])
def test_dry_run_rejects_unsafe_or_wrong_protocol(fixture_env, tmp_path, fault):
    if fault == "public_output":
        fixture_env["HPCA_P0_QUALITY_OUTPUT_ROOT"] = fixture_env["COMMON_EPOCH0_ROOT"] + "/bad"
    elif fault == "symlink_public_output":
        alias = tmp_path / "alias"
        alias.symlink_to(fixture_env["COMMON_EPOCH0_ROOT"], target_is_directory=True)
        fixture_env["HPCA_P0_QUALITY_OUTPUT_ROOT"] = str(alias / "bad")
    elif fault == "bad_seed":
        fixture_env["HPCA_P0_QUALITY_SEED"] = "505x"
    elif fault == "bad_epochs":
        fixture_env["HPCA_P0_QUALITY_TOTAL_EPOCHS"] = "2"
    elif fault == "bad_variant":
        fixture_env["HPCA_P0_QUALITY_VARIANTS"] = "graph_variant"
    else:
        next(Path(fixture_env["COMMON_EPOCH0_ROOT"]).rglob("*.distcp")).unlink()
    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"], env=fixture_env,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert not Path(fixture_env["HPCA_P0_QUALITY_OUTPUT_ROOT"]).exists()
