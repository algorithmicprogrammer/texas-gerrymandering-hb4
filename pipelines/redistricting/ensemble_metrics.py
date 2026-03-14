from __future__ import annotations
import duckdb


def build_ensemble_distribution(
    con: duckdb.DuckDBPyConnection,
    ensemble_id: str,
    metric_columns: list[str],
) -> None:
    """
    Prong-1 diagnostic distribution builder using plan_metrics.
    """
    con.execute("DELETE FROM ensemble_distribution WHERE ensemble_id = ?", [ensemble_id])

    value_expr = ", ".join([f"('{mc}', {mc})" for mc in metric_columns])

    con.execute(
        f"""
        INSERT INTO ensemble_distribution
        WITH long_metrics AS (
            SELECT
                p.ensemble_id,
                pm.opp_def_id,
                metric_name,
                metric_value::DOUBLE AS v
            FROM plan_metrics pm
            JOIN plan p USING (plan_id)
            CROSS JOIN LATERAL (
                SELECT * FROM (VALUES {value_expr}) AS t(metric_name, metric_value)
            ) x
            WHERE p.plan_type = 'ENSEMBLE'
              AND p.ensemble_id = ?
              AND metric_value IS NOT NULL
        )
        SELECT
            ensemble_id,
            opp_def_id,
            metric_name,
            COUNT(*) AS n_plans,
            AVG(v) AS mean,
            STDDEV_SAMP(v) AS sd,
            quantile_cont(v, 0.01) AS p01,
            quantile_cont(v, 0.05) AS p05,
            quantile_cont(v, 0.10) AS p10,
            quantile_cont(v, 0.25) AS p25,
            quantile_cont(v, 0.50) AS p50,
            quantile_cont(v, 0.75) AS p75,
            quantile_cont(v, 0.90) AS p90,
            quantile_cont(v, 0.95) AS p95,
            quantile_cont(v, 0.99) AS p99,
            MIN(v) AS min,
            MAX(v) AS max
        FROM long_metrics
        GROUP BY ensemble_id, opp_def_id, metric_name
        """,
        [ensemble_id],
    )
    print("[ensemble] built ensemble_distribution")


def build_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    ensemble_id: str,
    metric_columns: list[str],
) -> None:
    """
    Compare enacted plans to ensemble on Prong-1 diagnostic metrics.
    """
    con.execute("DELETE FROM plan_vs_ensemble WHERE ensemble_id = ?", [ensemble_id])

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
            JOIN (
                SELECT * FROM (VALUES
                    """ + ", ".join([f"('{mc}')" for mc in metric_columns]) + """
                ) AS t(metric_name)
            ) m ON TRUE
            CROSS JOIN LATERAL (
                SELECT
                    CASE m.metric_name
        """ + "\n".join([f"                        WHEN '{mc}' THEN pm.{mc}" for mc in metric_columns]) + """
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


def build_ei_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    ei_run_id: str,
    ensemble_id: str,
) -> None:
    """
    Main paper comparison:
      p = P_{pi ~ rho}( O(pi) <= O(pi_enacted) )
    where O(pi) = sum_k minority_preferred_win_prob_k.
    """
    con.execute(
        "DELETE FROM ei_plan_vs_ensemble WHERE ei_run_id = ? AND ensemble_id = ?",
        [ei_run_id, ensemble_id],
    )

    con.execute(
        """
        INSERT INTO ei_plan_vs_ensemble
        WITH ensemble_draws AS (
            SELECT
                s.ei_run_id,
                s.plan_id,
                p.ensemble_id,
                s.opportunity_score
            FROM plan_opportunity_score s
            JOIN plan p ON p.plan_id = s.plan_id
            WHERE s.ei_run_id = ?
              AND p.plan_type = 'ENSEMBLE'
              AND p.ensemble_id = ?
        ),
        ensemble_stats AS (
            SELECT
                ei_run_id,
                ensemble_id,
                AVG(opportunity_score) AS ensemble_mean,
                STDDEV_SAMP(opportunity_score) AS ensemble_sd
            FROM ensemble_draws
            GROUP BY ei_run_id, ensemble_id
        ),
        enacted_targets AS (
            SELECT
                s.ei_run_id,
                s.plan_id,
                p.ensemble_id,
                s.opportunity_score
            FROM plan_opportunity_score s
            JOIN plan p ON p.plan_id = s.plan_id
            WHERE s.ei_run_id = ?
              AND p.plan_type != 'ENSEMBLE'
              AND p.ensemble_id = ?
        )
        SELECT
            t.ei_run_id,
            t.plan_id,
            t.ensemble_id,
            t.opportunity_score,
            es.ensemble_mean,
            es.ensemble_sd,
            (
              SELECT AVG(CASE WHEN d.opportunity_score <= t.opportunity_score THEN 1.0 ELSE 0.0 END)
              FROM ensemble_draws d
              WHERE d.ei_run_id = t.ei_run_id AND d.ensemble_id = t.ensemble_id
            ) AS percentile,
            (
              SELECT AVG(CASE WHEN d.opportunity_score <= t.opportunity_score THEN 1.0 ELSE 0.0 END)
              FROM ensemble_draws d
              WHERE d.ei_run_id = t.ei_run_id AND d.ensemble_id = t.ensemble_id
            ) AS p_value_left,
            (
              SELECT AVG(CASE WHEN d.opportunity_score >= t.opportunity_score THEN 1.0 ELSE 0.0 END)
              FROM ensemble_draws d
              WHERE d.ei_run_id = t.ei_run_id AND d.ensemble_id = t.ensemble_id
            ) AS p_value_right,
            CASE
              WHEN es.ensemble_sd IS NULL OR es.ensemble_sd = 0 THEN NULL
              ELSE (t.opportunity_score - es.ensemble_mean) / es.ensemble_sd
            END AS z_score,
            (t.opportunity_score - es.ensemble_mean) AS delta_from_mean
        FROM enacted_targets t
        JOIN ensemble_stats es
          ON es.ei_run_id = t.ei_run_id
         AND es.ensemble_id = t.ensemble_id
        """,
        [ei_run_id, ensemble_id, ei_run_id, ensemble_id],
    )
    print("[ensemble] built ei_plan_vs_ensemble")
