"""
Canary check for reproducibility across two identical training runs.

This script compares `checkpoint_metrics.json` and `best_checkpoint.json` from
two run roots (baseline vs candidate) to detect silent non-determinism, where running
the same job produces different outputs with the same seeds and config, which should not happen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _compare_values(a: Any, b: Any, *, path: str, tol: float) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            raise AssertionError(f"Boolean mismatch at {path}: {a} vs {b}")
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        af = float(a)
        bf = float(b)
        if not math.isclose(af, bf, rel_tol=0.0, abs_tol=tol):
            raise AssertionError(
                f"Numeric mismatch at {path}: {af} vs {bf} (tol={tol})"
            )
        return

    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"None mismatch at {path}: {a} vs {b}")
        return

    if isinstance(a, str) and isinstance(b, str):
        if a != b:
            raise AssertionError(f"String mismatch at {path}: {a} vs {b}")
        return

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise AssertionError(
                f"List length mismatch at {path}: {len(a)} vs {len(b)}"
            )
        for index, (av, bv) in enumerate(zip(a, b, strict=True)):
            _compare_values(av, bv, path=f"{path}[{index}]", tol=tol)
        return

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(
                f"Dict keys mismatch at {path}: {sorted(a.keys())} vs {sorted(b.keys())}"
            )
        for key in sorted(a.keys()):
            _compare_values(a[key], b[key], path=f"{path}.{key}", tol=tol)
        return

    if a != b:
        raise AssertionError(f"Value mismatch at {path}: {a!r} vs {b!r}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact not found: {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reproducibility canary artifacts"
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="smoke")
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    rel = Path(args.run_label) / args.algo / f"seed_{args.seed}"
    checkpoint_a = args.baseline_root / rel / "checkpoint_metrics.json"
    checkpoint_b = args.candidate_root / rel / "checkpoint_metrics.json"
    best_a = args.baseline_root / rel / "best_checkpoint.json"
    best_b = args.candidate_root / rel / "best_checkpoint.json"

    payload_a = _load_json(checkpoint_a)
    payload_b = _load_json(checkpoint_b)
    best_payload_a = _load_json(best_a)
    best_payload_b = _load_json(best_b)

    _compare_values(payload_a, payload_b, path="checkpoint_metrics", tol=args.tol)
    _compare_values(
        best_payload_a, best_payload_b, path="best_checkpoint", tol=args.tol
    )

    print(
        f"[REPRO][{args.algo}] PASS | run_label={args.run_label} seed={args.seed} tol={args.tol}"
    )


if __name__ == "__main__":
    main()
