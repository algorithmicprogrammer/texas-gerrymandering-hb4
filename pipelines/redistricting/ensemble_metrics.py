# redistricting/ensemble_metrics.py
from __future__ import annotations

from typing import Iterable
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
    metric_columns: Iterable[str] = ("n_opportunity_districts",),
) -> None:
    """
    Build ensemble distribution tables for one or more metric columns in plan_metrics.

    Writes:
      - ensemble_distribution: one row per (ensemble_id, opp_def_id, metric_name) with summary stats
      - ensemble_distribution_draws: raw draws per (plan_id, opp_def_id, metric_name)
    """
    if not ensemble_id:
        raise ValueError("build_ensemble_distribution: ensemble_id is required")

    metric_columns = list(metric_columns)
    for mc in metric_columns:
        if not _col_exists(con, "plan_metrics", mc):
            cols = con.execute("DESCRIBE plan_metrics").fetchdf()
            raise ValueError(
                f"plan_metrics is missing metric column {mc!r}. "
                f"Available columns: {cols['column_name'].tolist()}"
            )

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

    # Clear prior results for this ensemble_id
    con.execute("DELETE FROM ensemble_distribution WHERE ensemble_id = ?", [ensemble_id])

    # Build raw draws table (replace fully; it is a helper table used for plotting)
    union_draws = []
    for mc in metric_columns:
        union_draws.append(
            f"""
            SELECT
                p.ensemble_id,
                pm.opp_def_id,
                '{mc}' AS metric_name,
                p.plan_id,
                pm.{mc}::DOUBLE AS metric_value
            FROM plan_metrics pm
            JOIN plan p USING (plan_id)
            WHERE p.plan_type='ENSEMBLE'
              AND p.ensemble_id = '{ensemble_id}'
              AND pm.{mc} IS NOT NULL
            """
        )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ensemble_distribution_draws AS
        {" UNION ALL ".join(union_draws)}
        """
    )

    # Summary stats by opp_def_id × metric_name
    con.execute(
        """
        INSERT INTO ensemble_distribution
        SELECT
            ensemble_id,
            opp_def_id,
            metric_name,
            COUNT(*)::BIGINT AS n_plans,
            AVG(metric_value) AS mean,
            STDDEV_SAMP(metric_value) AS sd,
            quantile_cont(metric_value, 0.01) AS p01,
            quantile_cont(metric_value, 0.05) AS p05,
            quantile_cont(metric_value, 0.10) AS p10,
            quantile_cont(metric_value, 0.25) AS p25,
            quantile_cont(metric_value, 0.50) AS p50,
            quantile_cont(metric_value, 0.75) AS p75,
            quantile_cont(metric_value, 0.90) AS p90,
            quantile_cont(metric_value, 0.95) AS p95,
            quantile_cont(metric_value, 0.99) AS p99,
            MIN(metric_value) AS min,
            MAX(metric_value) AS max
        FROM ensemble_distribution_draws
        WHERE ensemble_id = ?
        GROUP BY ensemble_id, opp_def_id, metric_name
        """,
        [ensemble_id],
    )

    print("[ensemble] built ensemble_distribution and ensemble_distribution_draws")


def build_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    ensemble_id: str,
    metric_columns: Iterable[str] = ("n_opportunity_districts",),
) -> None:
    """
    Compare enacted plan(s) against ensemble distribution for one or more metrics.

    Writes:
      - plan_vs_ensemble: one row per (plan_id, ensemble_id, opp_def_id, metric_name)
        with percentile, z-score, tail probs, delta-from-mean.
    """
    if not ensemble_id:
        raise ValueError("build_plan_vs_ensemble: ensemble_id is required")

    metric_columns = list(metric_columns)
    for mc in metric_columns:
        if not _col_exists(con, "plan_metrics", mc):
            cols = con.execute("DESCRIBE plan_metrics").fetchdf()
            raise ValueError(
                f"plan_metrics is missing metric column {mc!r}. "
                f"Available columns: {cols['column_name'].tolist()}"
            )

    # Clear prior comparisons for this ensemble
    con.execute("DELETE FROM plan_vs_ensemble WHERE ensemble_id = ?", [ensemble_id])

    # Ensure draws exist (build_ensemble_distribution should have made it, but be robust)
    existing = con.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='main'
        """
    ).fetchdf()["table_name"].tolist()

    if "ensemble_distribution_draws" not in set(existing):
        build_ensemble_distribution(con, ensemble_id=ensemble_id, metric_columns=metric_columns)

    # Build comparisons: every non-ensemble plan with this ensemble_id (typically ENACTED)
    con.execute(
        """
        INSERT INTO plan_vs_ensemble
        WITH draws AS (
            SELECT
                ensemble_id,
                opp_def_id,
                metric_name,
                metric_value AS v
            FROM ensemble_distribution_draws
            WHERE ensemble_id = ?
        ),
        draw_stats AS (
            SELECT
                ensemble_id,
                opp_def_id,
                metric_name,
                AVG(v) AS mean_v,
                STDDEV_SAMP(v) AS sd_v
            FROM draws
            GROUP BY ensemble_id, opp_def_id, metric_name
        ),
        targets AS (
            SELECT
                p.plan_id,
                p.ensemble_id,
                pm.opp_def_id,
                m.metric_name,
                pm_val::DOUBLE AS plan_value
            FROM plan p
            JOIN plan_metrics pm USING (plan_id)
            -- generate one row per metric via VALUES
            JOIN (
                SELECT * FROM (VALUES
                    """ +
        ", ".join([f"('{mc}')" for mc in metric_columns]) +
        """) AS t(metric_name)
            ) m ON TRUE
            -- pick the appropriate metric value using CASE
            CROSS JOIN LATERAL (
                SELECT
                    CASE m.metric_name
        """
        +
        "\n".join([f"                        WHEN '{mc}' THEN pm.{mc}" for mc in metric_columns])
        +
        """
                        ELSE NULL
                    END AS pm_val
            ) x
            WHERE p.plan_type != 'ENSEMBLE'
              AND p.ensemble_id = ?
              AND pm_val IS NOT NULL
        ),
        scored AS (
            SELECT
                t.plan_id,
                t.ensemble_id,
                t.opp_def_id,
                t.metric_name,
                t.plan_value,
                -- percentile P(draw <= plan_value)
                (SELECT AVG(CASE WHEN d.v <= t.plan_value THEN 1.0 ELSE 0.0 END)
                 FROM draws d
                 WHERE d.ensemble_id=t.ensemble_id
                   AND d.opp_def_id=t.opp_def_id
                   AND d.metric_name=t.metric_name) AS percentile,
                ds.mean_v,
                ds.sd_v
            FROM targets t
            JOIN draw_stats ds
              ON ds.ensemble_id=t.ensemble_id
             AND ds.opp_def_id=t.opp_def_id
             AND ds.metric_name=t.metric_name
        )
        SELECT
            plan_id,
            ensemble_id,
            opp_def_id,
            metric_name,
            plan_value,
            percentile,
            CASE WHEN sd_v IS NULL OR sd_v=0 THEN NULL ELSE (plan_value - mean_v)/sd_v END AS z_score,
            percentile AS tail_prob_low,
            (1.0 - percentile) AS tail_prob_high,
            (plan_value - mean_v) AS delta_from_mean
        FROM scored
        """,
        [ensemble_id, ensemble_id],
    )

    print("[ensemble] built plan_vs_ensemble")


