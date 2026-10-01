#!/usr/bin/env python3
"""E3 CPU-only check: what does a preset's loader load, against the model card?

Instantiates the model on the META device from its config alone (no weights are
downloaded, no GPU is touched) and compares parameter counts with and without a
vision tower against the Hub's own numbers. It answers, for
``Qwen/Qwen3.5-4B`` (the ``qwen3.5-4b`` preset, Hub tag ``image-text-to-text``):
does the preset's loading path pull in the vision tower?

Network use is limited to: ``config.json``, the safetensors index
(``model.safetensors.index.json``: tensor NAMES, a few hundred KB), the model
card (``README.md``) and the Hub API's metadata (``model_info``). No weight file
is requested.

    python scripts/e3_meta_device_check.py --out docs/receipts/2026-10-e3-vram/cpu/qwen3.5-4b_meta.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import e3_lib as L  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default="qwen3.5-4b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import torch
    import transformers
    from huggingface_hub import hf_hub_download, model_info
    from transformers import AutoConfig, AutoModelForCausalLM

    from backpropagate.config import MODEL_PRESETS
    from backpropagate.trainer import Trainer, _estimate_param_count_billions

    preset = MODEL_PRESETS[a.preset]
    model_id = preset.model_id
    arch = L.fetch_arch(model_id)  # text-only class on the meta device + composite count

    cfg = AutoConfig.from_pretrained(model_id)
    names_text: list[tuple[str, int]] = []
    from transformers.models.auto.auto_factory import _get_model_class

    mc = _get_model_class(cfg, AutoModelForCausalLM._model_mapping)
    use = cfg.get_text_config() if mc.config_class == cfg.sub_configs.get("text_config") else cfg
    with torch.device("meta"):
        m_text = mc(use)
    names_text = [(n, p.numel()) for n, p in m_text.named_parameters()]

    composite_by_prefix: dict[str, int] = {}
    composite_total = None
    if getattr(cfg, "vision_config", None) is not None:
        from transformers import AutoModelForImageTextToText

        mc2 = _get_model_class(cfg, AutoModelForImageTextToText._model_mapping)
        with torch.device("meta"):
            m_full = mc2(cfg)
        c: Counter[str] = Counter()
        for n, p in m_full.named_parameters():
            c[".".join(n.split(".")[:2])] += p.numel()
        composite_by_prefix = dict(c)
        composite_total = sum(c.values())

    info = model_info(model_id)
    st = getattr(info, "safetensors", None)
    hub_total = getattr(st, "total", None)
    hub_by_dtype = dict(getattr(st, "parameters", {}) or {})

    ckpt: dict = {}
    try:
        with open(hf_hub_download(model_id, "model.safetensors.index.json"), encoding="utf-8") as fh:
            keys = list(json.load(fh)["weight_map"])
        pref: Counter[str] = Counter(".".join(k.split(".")[:2]) for k in keys)
        ckpt = {"tensors": len(keys), "by_prefix": dict(pref),
                "language_model_tensors": sum(v for k, v in pref.items() if "language_model" in k),
                "visual_tensors": sum(v for k, v in pref.items() if "visual" in k or "vision" in k),
                "mtp_tensors": sum(v for k, v in pref.items() if k.startswith("mtp"))}
    except Exception as exc:  # noqa: BLE001 - single-file checkpoints have no index
        ckpt = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    card_lines: list[str] = []
    try:
        with open(hf_hub_download(model_id, "README.md"), encoding="utf-8") as fh:
            card = fh.read()
        card_lines = [m.group(0).strip() for m in re.finditer(r"(?im)^.*number of parameters.*$", card)]
    except Exception as exc:  # noqa: BLE001
        card_lines = [f"model card unavailable: {type(exc).__name__}"]

    loader_src = inspect.getsource(Trainer._load_with_transformers)
    text_params = sum(c for _, c in names_text)
    vision_in_text = sum(c for n, c in names_text if "visual" in n or "vision" in n)
    vision_params = None if composite_total is None else composite_total - text_params
    answer = {
        "does_the_transformers_path_load_the_vision_tower": vision_in_text > 0,
        "why": (f"Trainer._load_with_transformers calls AutoModelForCausalLM.from_pretrained, whose mapping resolves "
                f"the composite {type(cfg).__name__} to {type(m_text).__name__} on the text config: "
                f"{len(names_text)} tensors, {vision_in_text} vision parameters; the checkpoint's "
                f"{ckpt.get('visual_tensors')} visual tensors and {ckpt.get('mtp_tensors')} MTP tensors are not "
                "instantiated (they are still downloaded: they live in the same shards)."),
        "uses_AutoModelForCausalLM_in_source": "AutoModelForCausalLM.from_pretrained" in loader_src,
        "unsloth_path": ("NOT measured here; source reading only. Unsloth 2026.5.8 has no 'qwen3_5' entry anywhere "
                         "in its package (grep), so Qwen3.5 would take the generic path, where unsloth/models/loader.py "
                         "sets `is_vlm = ... hasattr(model_config, 'vision_config')` and picks AutoModelForVision2Seq, "
                         "which would instantiate the vision tower. The sweep records `vision_tensors_loaded` and "
                         "`model_class` for every Unsloth-on point; that is the measurement."),
    }
    rec = {
        "model": model_id, "preset": a.preset, "transformers": transformers.__version__, "torch": torch.__version__,
        "device": "meta (no weights, no GPU)",
        "text_only": {"class": type(m_text).__name__, "config_class": type(use).__name__, "tensors": len(names_text),
                      "params": text_params, "params_b": round(text_params / 1e9, 4),
                      "vision_params": vision_in_text,
                      "tied_embeddings": bool(getattr(use, "tie_word_embeddings", False))},
        "composite": {"params": composite_total, "by_prefix": composite_by_prefix, "vision_params": vision_params},
        "hub": {"pipeline_tag": info.pipeline_tag, "safetensors_total": hub_total, "by_dtype": hub_by_dtype},
        "checkpoint_index": ckpt,
        "model_card_stated": card_lines,
        "reconciliation": {
            "hub_total_minus_composite": None if (hub_total is None or composite_total is None) else hub_total - composite_total,
            "hub_total_minus_text_only": None if hub_total is None else hub_total - text_params,
            "note": "the Hub total counts every tensor in the checkpoint: language model + vision tower + MTP head"},
        "library_param_count_billions_name_derived": _estimate_param_count_billions(a.preset),
        "architecture": {k: arch.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                                   "num_key_value_heads", "vocab_size", "full_attention_layers",
                                                   "linear_attention_layers")},
        "answer": answer,
    }
    text = json.dumps(rec, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
