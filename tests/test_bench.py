"""Throughput benchmark (M3): synthetic batches, warmup excluded, MFU and time estimate."""
from conftest import tiny_config_dict
from llm.config import load_config_dict
from llm.model.gpt import expected_num_params
from llm.train.bench import bench, peak_flops


def test_bench_cpu_tiny(stories_path):
    cfg = load_config_dict(tiny_config_dict(stories_path))
    r = bench(cfg, steps=3, warmup=1)
    assert r["tokens_per_s"] > 0 and r["step_s"] > 0
    assert r["params"] == expected_num_params(cfg.model, cfg.tokenizer.vocab_size)
    assert r["mfu"] is None  # CPU has no peak in the table
    assert abs(r["hours_for"](r["tokens_per_s"] * 3600) - 1.0) < 1e-9


def test_peak_table():
    assert peak_flops("NVIDIA H100 80GB HBM3", "bf16") == 989e12
    assert peak_flops("Tesla T4", "fp16") == 65e12 and peak_flops("cpu", "fp32") is None
