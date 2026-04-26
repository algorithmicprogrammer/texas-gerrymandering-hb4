#!/usr/bin/env python3
"""Generate TX_logit_params.csv for the ensemble elections model.

This script fits one logistic calibration model per (model_type, subgroup)
combination expected by `compute_final_dist(..., logit=True)`:

- model_type: statewide, equal, district
- subgroup:   Black, Latino, Neither

Input calibration CSV format (required columns):
- model_type
- subgroup
- raw_prob        (uncalibrated probability, usually in [0, 1])
- is_opportunity  (0/1 observed label)

Output columns match downstream expectations:
- model_type
- subgroup
- coef
- intercept
- n_obs

If you pass --identity, the script ignores fitting and writes an identity mapping
(coef=1, intercept=0) for all nine combinations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MODEL_TYPES = ("statewide", "equal", "district")
EXPECTED_SUBGROUPS = ("Black", "Latino", "Neither")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def _fit_logit_one_feature(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> tuple[float, float]:
    """Fit logistic regression y ~ intercept + coef*x via Newton-Raphson."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0:
        raise ValueError("Cannot fit logistic model with zero rows.")
    if np.unique(y).size < 2:
        # Complete separation / single-class case.
        # Use a stable fallback so downstream code still runs.
        return 1.0, 0.0

    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=float)  # [intercept, coef]

    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        w = np.clip(p * (1.0 - p), 1e-8, None)

        grad = X.T @ (y - p)
        # Hessian for log-likelihood with ridge penalty
        H = X.T @ (X * w[:, None]) + ridge * np.eye(2)

        step = np.linalg.solve(H, grad)
        beta_new = beta + step

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break

        beta = beta_new

    intercept, coef = float(beta[0]), float(beta[1])
    return coef, intercept


def _validate_input(df: pd.DataFrame) -> None:
    required = {"model_type", "subgroup", "raw_prob", "is_opportunity"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    bad_models = sorted(set(df["model_type"]) - set(EXPECTED_MODEL_TYPES))
    bad_groups = sorted(set(df["subgroup"]) - set(EXPECTED_SUBGROUPS))
    if bad_models:
        raise ValueError(f"Unexpected model_type values: {bad_models}")
    if bad_groups:
        raise ValueError(f"Unexpected subgroup values: {bad_groups}")


def generate_identity_params() -> pd.DataFrame:
    rows = []
    for model_type in EXPECTED_MODEL_TYPES:
        for subgroup in EXPECTED_SUBGROUPS:
            rows.append(
                {
                    "model_type": model_type,
                    "subgroup": subgroup,
                    "coef": 1.0,
                    "intercept": 0.0,
                    "n_obs": 0,
                }
            )
    return pd.DataFrame(rows)


def generate_fitted_params(calibration_df: pd.DataFrame) -> pd.DataFrame:
    _validate_input(calibration_df)

    rows = []
    for model_type in EXPECTED_MODEL_TYPES:
        for subgroup in EXPECTED_SUBGROUPS:
            sub = calibration_df[
                (calibration_df["model_type"] == model_type)
                & (calibration_df["subgroup"] == subgroup)
            ].copy()

            if sub.empty:
                raise ValueError(
                    f"No rows for combination model_type={model_type}, subgroup={subgroup}."
                )

            x = pd.to_numeric(sub["raw_prob"], errors="coerce").to_numpy()
            y = pd.to_numeric(sub["is_opportunity"], errors="coerce").to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if x.size == 0:
                raise ValueError(
                    f"All rows became invalid after numeric coercion for {model_type}/{subgroup}."
                )

            y = np.where(y >= 0.5, 1.0, 0.0)
            coef, intercept = _fit_logit_one_feature(x, y)

            rows.append(
                {
                    "model_type": model_type,
                    "subgroup": subgroup,
                    "coef": coef,
                    "intercept": intercept,
                    "n_obs": int(x.size),
                }
            )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TX_logit_params.csv for besag_clifford_vra_opportunity.py"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Calibration CSV with model_type/subgroup/raw_prob/is_opportunity columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TX_logit_params.csv"),
        help="Output CSV path (default: TX_logit_params.csv in current directory).",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Write identity calibration (coef=1, intercept=0) instead of fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.identity:
        out = generate_identity_params()
    else:
        if args.input is None:
            raise ValueError("--input is required unless --identity is provided.")
        calibration_df = pd.read_csv(args.input)
        out = generate_fitted_params(calibration_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote {len(out)} rows to {args.output}")
    print(out)


if __name__ == "__main__":
    main()
