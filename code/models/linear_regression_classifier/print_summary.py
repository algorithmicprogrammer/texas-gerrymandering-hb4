from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def load_metrics() -> Dict[str, float]:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with metrics_path.open() as fh:
        return json.load(fh)


def load_classification_report() -> Dict[str, Dict[str, float]]:
    report_path = ARTIFACTS_DIR / "classification_report.csv"
    with report_path.open() as fh:
        reader = csv.reader(fh)
        header: List[str] = []
        metrics: Dict[str, Dict[str, float]] = {}
        for idx, row in enumerate(reader):
            if idx == 0:
                header = [entry.strip() for entry in row[1:]]
                continue
            metric_name = row[0].strip()
            if not metric_name:
                continue
            metrics[metric_name] = {
                label: float(value) for label, value in zip(header, row[1:]) if value
            }
    return metrics


def parse_p_values() -> Dict[str, float]:
    summary_path = ARTIFACTS_DIR / "ols_summary.txt"
    with summary_path.open() as fh:
        lines = fh.readlines()

    p_values: Dict[str, float] = {}
    in_table = False
    for line in lines:
        if line.strip().startswith("const"):
            in_table = True
        if not in_table:
            continue
        if not line.strip():
            continue
        if set(line.strip()) == {"="} or line.strip().startswith("Omnibus"):
            break
        parts = line.split()
        if len(parts) != 7:
            continue
        feature = parts[0]
        try:
            p_value = float(parts[4])
        except ValueError:
            continue
        p_values[feature] = p_value
    return p_values


def format_float(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    metrics = load_metrics()
    classification_report = load_classification_report()
    p_values = parse_p_values()

    print("Linear Regression Classifier Summary")
    print("=" * 40)
    print(f"Selected variant: {metrics.get('variant', 'unknown')}")
    print(f"Decision threshold: {metrics.get('threshold')}")
    print("\nPerformance Metrics:")
    for key in ("accuracy", "balanced_accuracy", "f1", "mse", "r2"):
        if key in metrics:
            print(f"  - {key.replace('_', ' ').title()}: {format_float(metrics[key])}")

    if classification_report:
        print("\nClassification Report:")
        for label, values in classification_report.items():
            pretty_label = label.replace("_", " ").title()
            metrics_str = ", ".join(
                f"{col}: {format_float(val)}" for col, val in values.items()
            )
            print(f"  - {pretty_label}: {metrics_str}")

    if p_values:
        print("\nCoefficient P-Values (OLS surrogate):")
        for feature, p_value in p_values.items():
            print(f"  - {feature}: {format_float(p_value)}")


if __name__ == "__main__":
    main()