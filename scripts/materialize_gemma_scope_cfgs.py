#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from frame_scope.gemma_scope_local import materialize_tree


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sae-lens cfg.json files for locally downloaded Gemma Scope folders."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="artifacts/saes",
        help="Root directory containing downloaded Gemma Scope artifacts.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string to encode into generated cfg.json files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite cfg.json even if it already exists.",
    )
    args = parser.parse_args()

    written = materialize_tree(Path(args.root), device=args.device, force=args.force)
    print(f"Generated {len(written)} cfg.json files under {args.root}")


if __name__ == "__main__":
    main()
