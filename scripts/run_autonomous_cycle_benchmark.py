from __future__ import annotations

import argparse
import json
from pathlib import Path

from bci_autodiscovery.evaluation import run_synthetic_cycle_benchmark_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the synthetic autonomous research-cycle benchmark."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_synthetic_cycle_benchmark_file(
        specification_path=args.spec,
        output_path=args.output,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
