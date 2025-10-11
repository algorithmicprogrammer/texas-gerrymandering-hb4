import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path("code/models/linear_regression_classifier/01_preprocess.ipynb")


def load_preprocess_namespace():
    with NOTEBOOK_PATH.open() as f:
        nb = json.load(f)
    namespace = {}
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "pd.read_csv" in source:
            break
        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_numeric_feature_list_uses_compactness_composite():
    ns = load_preprocess_namespace()

    numeric = ns["NUMERIC"]
    compactness_metrics = ns["COMPACTNESS_METRICS"]

    assert "compactness_pca_score" in numeric
    for column in compactness_metrics:
        assert column not in numeric


def test_build_dataset_adds_compactness_score_and_metadata():
    ns = load_preprocess_namespace()

    build_dataset = ns["build_dataset"]
    compactness_metrics = ns["COMPACTNESS_METRICS"]
    final_csv = ns["FINAL_CSV"]

    df = pd.read_csv(final_csv)
    transformed, metadata = build_dataset(df)

    assert "compactness_pca_score" in transformed
    assert metadata["metrics"] == compactness_metrics
    assert set(metadata).issuperset({
        "metrics",
        "scaler_mean",
        "scaler_scale",
        "pca_components",
        "explained_variance_ratio",
        "sign_correction",
    })


def test_compactness_scores_match_metadata_components():
    ns = load_preprocess_namespace()

    build_dataset = ns["build_dataset"]
    compactness_metrics = ns["COMPACTNESS_METRICS"]
    final_csv = ns["FINAL_CSV"]

    df = pd.read_csv(final_csv)
    transformed, metadata = build_dataset(df)

    compactness_values = transformed[compactness_metrics].to_numpy(dtype=float)
    mean = np.asarray(metadata["scaler_mean"], dtype=float)
    scale = np.asarray(metadata["scaler_scale"], dtype=float)
    components = np.asarray(metadata["pca_components"], dtype=float)

    scaled = (compactness_values - mean) / scale
    expected = scaled.dot(components)

    np.testing.assert_allclose(
        expected,
        transformed["compactness_pca_score"].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
    )
