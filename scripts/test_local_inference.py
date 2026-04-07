#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("artifacts/models/gemma-3-1b-it")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local inference smoke test against Gemma 3 1B IT.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local Gemma model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Frame-Scope studies the frame operators of transformer layers by",
        help="Prompt to feed into the model.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Execution device. 'auto' prefers MPS and falls back to CPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate before stopping.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Print the full decoded sequence including the prompt instead of only the new continuation.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"Model path not found: {args.model_path}")

    device = resolve_device(args.device)
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is not currently available in this Python environment.")

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = args.temperature
    else:
        generation_kwargs["do_sample"] = False

    torch.set_grad_enabled(False)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    model = model.to(device)
    model.eval()
    if args.temperature <= 0:
        model.generation_config.top_k = None
        model.generation_config.top_p = None
    load_elapsed = time.perf_counter() - load_start

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    generation_start = time.perf_counter()
    output = model.generate(**inputs, **generation_kwargs)
    generation_elapsed = time.perf_counter() - generation_start

    prompt_tokens = inputs["input_ids"].shape[-1]
    total_tokens = output.shape[-1]
    new_tokens = total_tokens - prompt_tokens

    if args.include_prompt:
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)

    print(f"model_path: {args.model_path}")
    print(f"device: {device}")
    print(f"load_seconds: {load_elapsed:.2f}")
    print(f"generation_seconds: {generation_elapsed:.2f}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"generated_new_tokens: {new_tokens}")
    print("---")
    print(text)


if __name__ == "__main__":
    main()
