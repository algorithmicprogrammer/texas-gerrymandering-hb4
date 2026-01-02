from __future__ import annotations

import duckdb


def _table_has_column(con: duckdb.DuckDBPyConnection, table: str, col: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return any(r[1] == col for r in rows)


def _get_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [r[1] for r in rows]


def _ensemble_plan_filter(con: duckdb.DuckDBPyConnection, ensemble_id: str | None) -> tuple[str, list]:
    """
    Returns (where_sql, params) selecting ensemble plans from plan p.

    Priority:
      1) If plan has ensemble_id column and ensemble_id provided -> p.ensemble_id = ?
      2) Else if ensemble_id provided -> CAST(p.plan_id AS VARCHAR) LIKE ?
      3) Else if plan has plan_type -> p.plan_type = 'ensemble'
      4) Else -> 1=1
    """
    has_ensemble_id = _table_has_column(con, "plan", "ensemble_id")
    has_plan_type = _table_has_column(con, "plan", "plan_type")

    if ensemble_id:
        if has_ensemble_id:
            return "p.ensemble_id = ?", [ensemble_id]
        return "CAST(p.plan_id AS VARCHAR) LIKE ?", [f"%{ensemble_id}%"]

    if has_plan_type:
        return "p.plan_type = 'ensemble'", []

    return "1=1", []


def _plan_metrics_is_long(con: duckdb.DuckDBPyConnection) -> bool:
    """
    Determine whether plan_metrics is long-format:
      columns include metric_name and metric_value
    """
    cols = set(_get_columns(con, "plan_metrics"))
    return ("metric_name" in cols) and ("metric_value" in cols)


def build_ensemble_distribution(
    con: duckdb.DuckDBPyConnection,
    metric_name: str,
    ensemble_id: str | None = None,
) -> None:
    """
    Build ensemble_distribution for a metric.

    Supports TWO plan_metrics schemas:
      A) long-format:  plan_metrics(plan_id, metric_name, metric_value, ...)
      B) wide-format:  plan_metrics(plan_id, <metric columns...>) and metric_name is a column name
    """
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "plan_metrics" not in tables:
        raise ValueError("plan_metrics table not found. Run the metrics stage first.")
    if "plan" not in tables:
        raise ValueError("plan table not found. Run the load stage first.")

    where_sql, params = _ensemble_plan_filter(con, ensemble_id)

    # Clear previous rows for this key
    con.execute(
        """
        DELETE FROM ensemble_distribution
        WHERE metric_name = ?
          AND ensemble_id IS NOT DISTINCT FROM ?
        """,
        [metric_name, ensemble_id],
    )

    is_long = _plan_metrics_is_long(con)

    if is_long:
        # Long format: filter rows by pm.metric_name
        n = con.execute(
            f"""
            SELECT COUNT(*)
            FROM plan p
            JOIN plan_metrics pm ON pm.plan_id = p.plan_id
            WHERE {where_sql}
              AND pm.metric_name = ?
            """,
            params + [metric_name],
        ).fetchone()[0]

        if n == 0:
            sample = con.execute("SELECT * FROM plan LIMIT 10").fetchdf()
            pm_cols = con.execute("PRAGMA table_info('plan_metrics')").fetchdf()
            raise ValueError(
                "No ensemble plans/metrics found under current filter (long plan_metrics).\n"
                f"ensemble_id={ensemble_id!r}, metric_name={metric_name!r}\n\n"
                "plan_metrics columns:\n"
                f"{pm_cols[['name','type']].to_string(index=False)}\n\n"
                "Sample plan rows:\n"
                f"{sample.to_string(index=False)}"
            )

        con.execute(
            f"""
            INSERT INTO ensemble_distribution (ensemble_id, metric_name, plan_id, metric_value)
            SELECT
                ? AS ensemble_id,
                pm.metric_name,
                pm.plan_id,
                pm.metric_value
            FROM plan p
            JOIN plan_metrics pm ON pm.plan_id = p.plan_id
            WHERE {where_sql}
              AND pm.metric_name = ?
            """,
            [ensemble_id] + params + [metric_name],
        )
        return

    # Wide format: metric_name is a COLUMN of plan_metrics
    cols = set(_get_columns(con, "plan_metrics"))
    if metric_name not in cols:
        # Give a helpful error listing the available metric columns
        # (exclude obvious ID-ish columns)
        likely_metrics = [c for c in sorted(cols) if c not in {"plan_id", "district_id"}]
        raise ValueError(
            "plan_metrics is wide-format, but the requested metric_name is not a column.\n"
            f"Requested metric_name={metric_name!r}\n"
            f"Available columns (likely metrics): {likely_metrics[:50]}"
            + (" ..." if len(likely_metrics) > 50 else "")
        )

    n = con.execute(
        f"""
        SELECT COUNT(*)
        FROM plan p
        JOIN plan_metrics pm ON pm.plan_id = p.plan_id
        WHERE {where_sql}
          AND pm.{metric_name} IS NOT NULL
        """,
        params,
    ).fetchone()[0]

    if n == 0:
        sample = con.execute("SELECT * FROM plan LIMIT 10").fetchdf()
        raise ValueError(
            "No ensemble plans found under current filter (wide plan_metrics).\n"
            f"ensemble_id={ensemble_id!r}, metric_column={metric_name!r}\n\n"
            "Sample plan rows:\n"
            f"{sample.to_string(index=False)}"
        )

    # Insert using the column as the metric_value; store metric_name as label
    con.execute(
        f"""
        INSERT INTO ensemble_distribution (ensemble_id, metric_name, plan_id, metric_value)
        SELECT
            ? AS ensemble_id,
            ? AS metric_name,
            pm.plan_id,
            CAST(pm.{metric_name} AS DOUBLE) AS metric_value
        FROM plan p
        JOIN plan_metrics pm ON pm.plan_id = p.plan_id
        WHERE {where_sql}
          AND pm.{metric_name} IS NOT NULL
        """,
        [ensemble_id, metric_name] + params,
    )


def build_plan_vs_ensemble(
    con: duckdb.DuckDBPyConnection,
    metric_name: str,
    ensemble_id: str | None = None,
) -> None:
    """
    Compare each plan's metric_value to the ensemble distribution:
      - ensemble_mean
      - ensemble_sd
      - ensemble_pctl (empirical percentile)
    """
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if "ensemble_distribution" not in tables:
        raise ValueError("ensemble_distribution not found. Run build_ensemble_distribution first.")
    if "plan_metrics" not in tables:
        raise ValueError("plan_metrics not found. Run metrics stage first.")

    # Clear previous
    con.execute(
        """
        DELETE FROM plan_vs_ensemble
        WHERE metric_name = ?
          AND ensemble_id IS NOT DISTINCT FROM ?
        """,
        [metric_name, ensemble_id],
    )

    is_long = _plan_metrics_is_long(con)

    if is_long:
        # plan_metrics has metric_name/metric_value rows
        con.execute(
            """
            INSERT INTO plan_vs_ensemble (
                ensemble_id, metric_name, plan_id, metric_value,
                ensemble_mean, ensemble_sd, ensemble_pctl
            )
            WITH dist AS (
                SELECT metric_value
                FROM ensemble_distribution
                WHERE metric_name = ?
                  AND ensemble_id IS NOT DISTINCT FROM ?
            ),
            stats AS (
                SELECT
                    AVG(metric_value)::DOUBLE AS mu,
                    STDDEV_SAMP(metric_value)::DOUBLE AS sd
                FROM dist
            )
            SELECT
                ? AS ensemble_id,
                pm.metric_name,
                pm.plan_id,
                pm.metric_value,
                s.mu AS ensemble_mean,
                s.sd AS ensemble_sd,
                (
                    SELECT AVG(CASE WHEN d.metric_value <= pm.metric_value THEN 1 ELSE 0 END)::DOUBLE
                    FROM dist d
                ) AS ensemble_pctl
            FROM plan_metrics pm
            CROSS JOIN stats s
            WHERE pm.metric_name = ?
            """,
            [metric_name, ensemble_id, ensemble_id, metric_name],
        )
        return

    # Wide format: metric_name is a column name
    cols = set(_get_columns(con, "plan_metrics"))
    if metric_name not in cols:
        raise ValueError(
            f"plan_metrics is wide-format but has no column {metric_name!r}."
        )

    con.execute(
        f"""
        INSERT INTO plan_vs_ensemble (
            ensemble_id, metric_name, plan_id, metric_value,
            ensemble_mean, ensemble_sd, ensemble_pctl
        )
        WITH dist AS (
            SELECT metric_value
            FROM ensemble_distribution
            WHERE metric_name = ?
              AND ensemble_id IS NOT DISTINCT FROM ?
        ),
        stats AS (
            SELECT
                AVG(metric_value)::DOUBLE AS mu,
                STDDEV_SAMP(metric_value)::DOUBLE AS sd
            FROM dist
        )
        SELECT
            ? AS ensemble_id,
            ? AS metric_name,
            pm.plan_id,
            CAST(pm.{metric_name} AS DOUBLE) AS metric_value,
            s.mu AS ensemble_mean,
            s.sd AS ensemble_sd,
            (
                SELECT AVG(CASE WHEN d.metric_value <= CAST(pm.{metric_name} AS DOUBLE) THEN 1 ELSE 0 END)::DOUBLE
                FROM dist d
            ) AS ensemble_pctl
        FROM plan_metrics pm
        CROSS JOIN stats s
        WHERE pm.{metric_name} IS NOT NULL
        """,
        [metric_name, ensemble_id, ensemble_id, metric_name],
    )

