from __future__ import annotations

import duckdb


def _col_exists(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema='main' AND table_name=? AND column_name=?
    LIMIT 1
    """
    return con.execute(q, [table, col]).fetchone() is not None


def build_ensemble_distribution(
    con: duckdb.DuckDBPyConnection,
    ensemble_id: str,
    metric_column: str = "n_opportunity_districts",
) -> None:
    """
    Build summary distribution table for an ensemble over a single metric column.

    Requires:
      - plan table with plan_type='ENSEMBLE' and ensemble_id
      - plan_metrics table with a column metric_column, keyed by plan_id
    Creates/overwrites:
      - ensemble_distribution
    """
    if not ensemble_id:
        raise ValueError("build_ensemble_distribution: ensemble_id is required")

    if not _col_exists(con, "plan_metrics", metric_column):
        cols = con.execute("DESCRIBE plan_metrics").fetchdf()
        raise ValueError(
            f"plan_metrics is missing metric column {metric_column!r}. "
            f"Available columns: {cols['column_name'].tolist()}"
        )

    # Ensure there ARE ensemble plans for this id
    n_plans = con.execute(
        """
        SELECT COUNT(*)
        FROM plan
        WHERE plan_type='ENSEMBLE' AND ensemble_id=?
        """,
        [ensemble_id],
    ).fetchone()[0]

    if n_plans == 0:
        sample = con.execute("SELECT * FROM plan LIMIT 10").fetchdf()
        raise ValueError(
            "No ensemble plans found under current filter (plan table).\n"
            f"ensemble_id={ensemble_id!r}\n\nSample plan rows:\n{sample.to_string(index=False)}"
        )

    # Create distribution summary. (DuckDB supports quantile_cont.)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ensemble_distribution AS
        WITH ens AS (
            SELECT
                p.ensemble_id,
                pm.{metric_column}::DOUBLE AS metric_value
            FROM plan_metrics pm
            JOIN plan p USING (plan_id)
            WHERE p.plan_type='ENSEMBLE'
              AND p.ensemble_id = ?
              AND pm.{metric_column} IS NOT NULL
        )
        SELECT
            ensemble_id,
            '{metric_column}' AS metric_column,
            COUNT(*)::BIGINT AS n_plans,
            AVG(metric_value) AS mean,
            STDDEV_SAMP(metric_value) AS sd,
            MIN(metric_value) AS min,
            MAX(metric_value) AS max,
            quantile_cont(metric_value, 0.01) AS q01,
            quantile_cont(metric_value, 0.05) AS q05,
            quantile_cont(metric_value, 0.10) AS q10,
            quantile_cont(metric_value, 0.25) AS q25,
            quantile_cont(metric_value, 0.50) AS q50,
            quantile_cont(metric_value, 0.75) AS q75,
            quantile_cont(metric_value, 0.90) AS q90,
            quantile_cont(metric_value, 0.95) AS q95,
            quantile_cont(metric_value, 0.99) AS q99
        FROM ens
        GROUP BY ensemble_id
        """,
        [ensemble_id],
    )

    # Also store the raw draws (useful for plotting)
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ensemble_distribution_draws AS
        SELECT
            p.ensemble_id,
            p.plan_id,
            pm.{metric_column}::DOUBLE AS metric_value
        FROM plan_metrics pm
        JOIN plan p USING (plan_id)
        WHERE p.plan_type='ENSEMBLE'
          AND p.ensemble_id = ?
          AND pm.{metric_column} IS NOT NULL
        """,
        [ensemble_id],
    )

    print("[ensemble] built ensemble_distribution and ensemble_distribution_draws")


def build_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    ensemble_id: str,
    metric_column: str = "n_opportunity_districts",
) -> None:
    """
    Compare enacted plan(s) against ensemble distribution for a single metric.

    Creates/overwrites:
      - plan_vs_ensemble
    """
    if not ensemble_id:
        raise ValueError("build_plan_vs_ensemble: ensemble_id is required")

    if not _col_exists(con, "plan_metrics", metric_column):
        cols = con.execute("DESCRIBE plan_metrics").fetchdf()
        raise ValueError(
            f"plan_metrics is missing metric column {metric_column!r}. "
            f"Available columns: {cols['column_name'].tolist()}"
        )

    # Build percentile rank of each non-ensemble plan in this ensemble_id
    con.execute(
        f"""
        CREATE OR REPLACE TABLE plan_vs_ensemble AS
        WITH draws AS (
            SELECT pm.{metric_column}::DOUBLE AS v
            FROM plan_metrics pm
            JOIN plan p USING (plan_id)
            WHERE p.plan_type='ENSEMBLE'
              AND p.ensemble_id=?
              AND pm.{metric_column} IS NOT NULL
        ),
        targets AS (
            SELECT
                p.plan_id,
                p.plan_type,
                p.ensemble_id,
                pm.{metric_column}::DOUBLE AS plan_value
            FROM plan_metrics pm
            JOIN plan p USING (plan_id)
            WHERE p.plan_type != 'ENSEMBLE'
              AND p.ensemble_id=?
              AND pm.{metric_column} IS NOT NULL
        )
        SELECT
            t.*,
            (SELECT COUNT(*) FROM draws) AS n_draws,
            (SELECT AVG(CASE WHEN d.v <= t.plan_value THEN 1 ELSE 0 END) FROM draws d) AS percentile_leq
        FROM targets t
        """,
        [ensemble_id, ensemble_id],
    )

    print("[ensemble] built plan_vs_ensemble")

