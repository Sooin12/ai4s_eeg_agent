"""Read-only inspection CLI for historical human-approved protocol artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol_artifacts import ProtocolArtifactError, ProtocolArtifactRegistry, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect legacy human-approval protocol artifacts. This historical interface "
            "cannot register, revise, approve, or activate protocols."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    artifact = subparsers.add_parser("inspect-artifact")
    artifact.add_argument("--artifact", type=Path, required=True)
    registry = subparsers.add_parser("inspect-registry")
    registry.add_argument("--dataset-id", required=True)
    registry.add_argument("--registry-root", type=Path, default=Path("artifacts/protocols"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-artifact":
            value = load_json(args.artifact.expanduser().resolve())
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        registry = ProtocolArtifactRegistry(
            root=args.registry_root,
            dataset_id=args.dataset_id,
        )
        print(json.dumps(registry.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ProtocolArtifactError, KeyError, OSError, json.JSONDecodeError) as exc:
        print("status: blocked")
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
