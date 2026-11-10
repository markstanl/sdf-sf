"""unit tests for the pure config logic in pipelines/run_pipeline.py (no subprocesses)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from run_pipeline import axolotl_cmd, preflight


def sdf_cfg(arm="sdf", seed=42, model="Qwen/Base", out_dir="sdf_vs_no_sdf"):
    return {"base_model": model, "seed": seed,
            "datasets": [{"path": f"data/{out_dir}/{arm}_mix.jsonl", "type": "completion"}],
            "output_dir": f"data/{out_dir}/models/{arm}-seed{seed}-mid"}


def make_cfgs(tokenizer="tok", model="Qwen/Base"):
    pipe = {"name": "p", "model": model, "arms": ["sdf", "no_sdf"], "seeds": [42],
            "stages": {"corpora": "c.yaml", "sdf": "configs/sdf", "sft": "f.yaml", "evals": []}}
    cfgs = {
        "corpora": {"tokenizer": tokenizer, "arms": {"sdf": {}}, "control_arm": "no_sdf",
                    "out_dir": "sdf_vs_no_sdf"},
        "sdf": {"sdf-seed42": sdf_cfg("sdf", 42, model),
                "no_sdf-seed42": sdf_cfg("no_sdf", 42, model)},
        "sft": {"tokenizer": "tok"},
    }
    return pipe, cfgs


def test_preflight_passes_on_consistent_configs():
    pipe, cfgs = make_cfgs()
    preflight(pipe, cfgs, ["sdf", "no_sdf"], [42])


def test_preflight_rejects_tokenizer_mismatch():
    pipe, cfgs = make_cfgs(tokenizer="other")
    with pytest.raises(AssertionError, match="tokenizer mismatch"):
        preflight(pipe, cfgs, ["sdf"], [42])


def test_preflight_rejects_base_model_mismatch():
    pipe, cfgs = make_cfgs()
    pipe["model"] = "other"
    with pytest.raises(AssertionError, match="base_model"):
        preflight(pipe, cfgs, ["sdf"], [42])


def test_preflight_rejects_unknown_arm():
    pipe, cfgs = make_cfgs()
    with pytest.raises(AssertionError, match="not in corpora arms"):
        preflight(pipe, cfgs, ["other"], [42])


def test_preflight_rejects_seed_mismatch():
    pipe, cfgs = make_cfgs()
    cfgs["sdf"]["sdf-seed42"]["seed"] = 43
    with pytest.raises(AssertionError, match="config seed"):
        preflight(pipe, cfgs, ["sdf"], [42])


def test_preflight_rejects_dataset_not_matching_corpora_out_dir():
    pipe, cfgs = make_cfgs()
    cfgs["sdf"]["sdf-seed42"]["datasets"][0]["path"] = "data/elsewhere/sdf_mix.jsonl"
    with pytest.raises(AssertionError, match="from corpora out_dir"):
        preflight(pipe, cfgs, ["sdf"], [42])


def test_axolotl_cmd_single_gpu(monkeypatch):
    monkeypatch.delenv("WHY_GEN_AXOLOTL_VENV", raising=False)
    assert axolotl_cmd("train", Path("c.yaml"), 1) == ["axolotl", "train", Path("c.yaml")]
    assert axolotl_cmd("preprocess", Path("c.yaml"), 1) == ["axolotl", "preprocess", Path("c.yaml")]


def test_axolotl_cmd_multi_gpu_uses_accelerate_and_deepspeed(monkeypatch):
    monkeypatch.delenv("WHY_GEN_AXOLOTL_VENV", raising=False)
    cmd = axolotl_cmd("train", Path("c.yaml"), 4)
    assert cmd[:3] == ["accelerate", "launch", "--num_processes"]
    assert "--use_deepspeed" in cmd and "-m" in cmd


def test_axolotl_cmd_prefers_venv_binaries(monkeypatch):
    monkeypatch.setenv("WHY_GEN_AXOLOTL_VENV", "/venvs/axo")
    assert axolotl_cmd("train", Path("c.yaml"), 1)[0] == "/venvs/axo/bin/axolotl"
    assert axolotl_cmd("train", Path("c.yaml"), 2)[0] == "/venvs/axo/bin/accelerate"
