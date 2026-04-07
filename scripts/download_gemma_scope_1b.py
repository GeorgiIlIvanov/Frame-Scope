#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sae_patterns(release: dict) -> list[str]:
    patterns: list[str] = []
    width = release["width"]
    l0 = release["l0"]
    for site in release["sites"]:
        for layer in release["layers"]:
            prefix = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
            for filename in release["files"]:
                patterns.append(f"{prefix}/{filename}")
    return patterns


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_repo(repo_id: str, local_dir: Path, allow_patterns: Iterable[str] | None, gated: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in this environment. Run scripts/bootstrap_macos.sh first."
        ) from exc

    ensure_dir(local_dir)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        )
    except GatedRepoError as exc:
        if gated:
            raise SystemExit(
                f"{repo_id} is gated. Run `hf auth login` with an account that has accepted the Gemma license, then rerun."
            ) from exc
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the selected Gemma 3 1B + Gemma Scope artifacts.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/gemma_scope_1b.json"),
        help="Path to the local download manifest.",
    )
    parser.add_argument(
        "--base",
        choices=["none", "it", "both"],
        default="it",
        help="Which gated Gemma 3 base models to download.",
    )
    parser.add_argument(
        "--saes",
        choices=["none", "it", "both"],
        default="both",
        help="Which Gemma Scope SAE repos to download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without contacting Hugging Face.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    root = args.manifest.resolve().parent.parent

    selected_base_ids = {"google/gemma-3-1b-it"}
    if args.base == "both":
        selected_base_ids.add("google/gemma-3-1b-pt")
    elif args.base == "none":
        selected_base_ids = set()

    selected_sae_ids = {"google/gemma-scope-2-1b-it", "google/gemma-scope-2-1b-pt"}
    if args.saes == "it":
        selected_sae_ids = {"google/gemma-scope-2-1b-it"}
    elif args.saes == "none":
        selected_sae_ids = set()

    print("Frame-Scope 1B download plan")
    print(f"Manifest: {args.manifest}")

    for base in manifest["base_models"]:
        if base["repo_id"] not in selected_base_ids:
            continue
        print(f"- base model: {base['repo_id']} -> {base['local_dir']}")

    for release in manifest["sae_releases"]:
        if release["repo_id"] not in selected_sae_ids:
            continue
        patterns = sae_patterns(release)
        print(f"- SAE repo: {release['repo_id']} -> {release['local_dir']}")
        print(f"  files: {len(patterns)} selected artifacts")

    if args.dry_run:
        return

    for base in manifest["base_models"]:
        if base["repo_id"] not in selected_base_ids:
            continue
        local_dir = root / base["local_dir"]
        download_repo(base["repo_id"], local_dir, allow_patterns=None, gated=True)

    for release in manifest["sae_releases"]:
        if release["repo_id"] not in selected_sae_ids:
            continue
        local_dir = root / release["local_dir"]
        download_repo(release["repo_id"], local_dir, allow_patterns=sae_patterns(release))


if __name__ == "__main__":
    main()
