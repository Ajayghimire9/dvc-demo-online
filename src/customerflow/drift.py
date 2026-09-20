"""CLI for CustomerFlow production drift checks."""
from __future__ import annotations

import argparse
from .monitoring import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="artifacts/drift-report.json")
    args = parser.parse_args()
    report = write_report(args.reference, args.current, args.output)
    print(f"drift_status={report['overall_status']} output={args.output}")


if __name__ == "__main__":
    main()
