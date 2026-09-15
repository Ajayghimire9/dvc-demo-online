"""CLI for the deterministic CustomerFlow curation stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_source, prepare_frame, write_ingestion_manifest


def run(source: str | Path, output: str | Path) -> int:
    source_path, output_path = Path(source), Path(output)
    frame = prepare_frame(load_source(source_path), require_target=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    write_ingestion_manifest(source_path, output_path, frame)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "--source", dest="source", default="examples/customers.csv")
    parser.add_argument("--output", default="data/curated/customers.csv")
    args = parser.parse_args()
    print(f"curated_rows={run(args.source, args.output)}")


if __name__ == "__main__":
    main()
