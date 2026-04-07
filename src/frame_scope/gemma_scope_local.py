from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open
from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import (
    handle_config_defaulting,
    str_to_dtype,
)


def _tensor_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: tuple(handle.get_slice(key).get_shape())
            for key in handle.keys()
        }


def _normalize_model_name(model_name: str) -> str:
    if "google" not in model_name:
        model_name = "google/" + model_name
    model_name = model_name.replace("-v3", "-3")
    if "270m" in model_name:
        model_name = model_name.replace("-pt", "")
    return model_name


def _infer_folder_name(folder: Path) -> str:
    if folder.parent == folder:
        raise ValueError(f"Cannot infer Gemma Scope folder name from {folder}")
    return f"{folder.parent.name}/{folder.name}"


def build_sae_lens_cfg(
    folder: str | Path,
    device: str = "cpu",
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    folder = Path(folder)
    raw_cfg = json.loads((folder / "config.json").read_text())
    if raw_cfg.get("architecture") != "jump_relu":
        raise ValueError(
            f"Unexpected Gemma Scope architecture in {folder}: {raw_cfg.get('architecture')}"
        )

    folder_name = _infer_folder_name(folder)
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match is None:
        raise ValueError(f"Could not infer layer number from {folder_name}")
    layer = int(layer_match.group(1))

    hook_name_out = None
    d_out = None
    if "resid_post" in folder_name:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "attn_out" in folder_name:
        hook_name = f"blocks.{layer}.attn.hook_z"
    elif "mlp_out" in folder_name:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "transcoder" in folder_name:
        hook_name = f"blocks.{layer}.hook_mlp_in"
        hook_name_out = f"blocks.{layer}.hook_mlp_out"
    else:
        raise ValueError(f"Unsupported Gemma Scope site in {folder_name}")

    shapes = _tensor_shapes(folder / "params.safetensors")
    d_in, d_sae = shapes["w_enc"]

    architecture = "jumprelu"
    if "transcoder" in folder_name:
        architecture = (
            "jumprelu_skip_transcoder"
            if raw_cfg.get("affine_connection", False)
            else "jumprelu_transcoder"
        )
        d_out = shapes["w_dec"][-1]

    cfg: dict[str, Any] = {
        "architecture": architecture,
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": _normalize_model_name(raw_cfg["model_name"]),
        "hook_name": hook_name,
        "hook_head_index": None,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "hf_hook_name": raw_cfg.get("hf_hook_point_in"),
        "device": device,
    }
    if hook_name_out is not None:
        cfg["hook_name_out"] = hook_name_out
        cfg["hf_hook_name_out"] = raw_cfg.get("hf_hook_point_out")
    if d_out is not None:
        cfg["d_out"] = d_out
    if cfg_overrides is not None:
        cfg.update(cfg_overrides)
    return cfg


def materialize_cfg_json(
    folder: str | Path,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    folder = Path(folder)
    output_path = folder / "cfg.json"
    if output_path.exists() and not force:
        return output_path
    cfg = build_sae_lens_cfg(folder, device=device)
    output_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return output_path


def materialize_weights_link(folder: str | Path, force: bool = False) -> Path:
    folder = Path(folder)
    source = folder / "params.safetensors"
    target = folder / "sae_weights.safetensors"
    if target.exists() or target.is_symlink():
        if not force:
            return target
        target.unlink()
    target.symlink_to(source.name)
    return target


def materialize_tree(root: str | Path, device: str = "cpu", force: bool = False) -> list[Path]:
    root = Path(root)
    written: list[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        folder = config_path.parent
        if not (folder / "params.safetensors").exists():
            continue
        materialize_weights_link(folder, force=force)
        written.append(materialize_cfg_json(folder, device=device, force=force))
    return written


def load_local_gemma_scope_sae(
    folder: str | Path,
    device: str = "cpu",
    dtype: str = "float32",
) -> SAE:
    folder = Path(folder)
    cfg_dict = build_sae_lens_cfg(folder, device=device, cfg_overrides={"dtype": dtype})
    cfg_dict = handle_config_defaulting(cfg_dict)

    state_dict: dict[str, Any] = {}
    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            match key:
                case "w_enc":
                    state_dict["W_enc"] = tensor
                case "w_dec":
                    state_dict["W_dec"] = tensor
                case "b_enc":
                    state_dict["b_enc"] = tensor
                case "b_dec":
                    state_dict["b_dec"] = tensor
                case "threshold":
                    state_dict["threshold"] = tensor
                case "affine_skip_connection":
                    state_dict["W_skip"] = tensor
                case _:
                    state_dict[key] = tensor

    sae_config_cls = SAE.get_sae_config_class_for_architecture(cfg_dict["architecture"])
    sae_cfg = sae_config_cls.from_dict(cfg_dict)
    sae_cls = SAE.get_sae_class_for_architecture(sae_cfg.architecture())

    target_device = sae_cfg.device
    sae_cfg.device = "meta"
    sae = sae_cls(sae_cfg)
    sae.cfg.device = target_device
    sae.process_state_dict_for_loading(state_dict)
    sae.load_state_dict(state_dict, assign=True)
    return sae.to(dtype=str_to_dtype(dtype), device=target_device)
