"""Command-line entry point for CustomerFlow training."""

from __future__ import annotations

import argparse

from .model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/curated/customers.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    metrics = train_model(args.data, args.output)
    print(f"selected_model={metrics['selected_model']} test_rmse={metrics['test']['rmse']}")


if __name__ == "__main__":
    main()
