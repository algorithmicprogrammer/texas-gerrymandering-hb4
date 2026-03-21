from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


R_SCRIPT = r'''
suppressPackageStartupMessages({
  library(jsonlite)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(eiPack)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]

cfg <- fromJSON(input_path, simplifyVector = TRUE)
cvap <- read_csv(cfg$cvap_csv, show_col_types = FALSE)
returns <- read_csv(cfg$returns_csv, show_col_types = FALSE)
weights <- read_csv(cfg$weights_csv, show_col_types = FALSE)

groups <- cfg$groups
all_results <- list()

group_pref_rows <- list()
precinct_draw_rows <- list()

i <- 1
for (eid in unique(weights$election_id)) {
  w <- weights |> filter(election_id == eid) |> slice(1)
  election_weight <- ifelse("weight" %in% names(w), w$weight[[1]], 1.0)
  subret <- returns |> filter(election_id == eid)
  if (nrow(subret) == 0) next

  wide_ret <- subret |>
    select(vtd_geoid, candidate_id, votes) |>
    tidyr::pivot_wider(names_from = candidate_id, values_from = votes, values_fill = 0)

  merged <- cvap |> inner_join(wide_ret, by = "vtd_geoid")
  cand_cols <- setdiff(names(wide_ret), "vtd_geoid")
  merged$None <- pmax(merged$cvap_total - rowSums(as.matrix(merged[, cand_cols, drop = FALSE])), 0)

  row_cols <- groups
  col_cols <- c(cand_cols, "None")

  # ei.MD.bayes takes matrices of row and column marginals for RxC tables.
  row_margins <- as.matrix(merged[, row_cols, drop = FALSE])
  col_margins <- as.matrix(merged[, col_cols, drop = FALSE])

  fit <- ei.MD.bayes(t = row_margins, total = rowSums(row_margins), y = col_margins, sample = 1000, burnin = 200)

  # fit$Beta.sims is draws x precinct x (R*C) in eiPack implementations used in practice.
  beta <- fit$Beta.sims
  dims <- dim(beta)
  ndraws <- dims[[1]]
  nprec <- dims[[2]]
  rc <- dims[[3]]

  arr <- array(beta, dim = c(ndraws, nprec, length(row_cols), length(col_cols)))

  for (g_idx in seq_along(row_cols)) {
    g <- row_cols[[g_idx]]
    pref_counts <- setNames(rep(0, length(cand_cols)), cand_cols)
    for (d in seq_len(ndraws)) {
      totals <- rep(0, length(cand_cols))
      for (c_idx in seq_along(cand_cols)) {
        totals[c_idx] <- sum(arr[d, , g_idx, c_idx] * merged[[g]])
      }
      pref_counts[which.max(totals)] <- pref_counts[which.max(totals)] + 1
    }
    pref_prob <- pref_counts / ndraws
    pref_cand <- names(pref_prob)[which.max(pref_prob)]
    p <- max(pref_prob)
    confidence <- 1 / (1 + exp(18 - 26 * p))
    group_pref_rows[[length(group_pref_rows) + 1]] <- data.frame(
      election_id = eid,
      group_id = g,
      preferred_candidate_id = pref_cand,
      pref_prob = p,
      confidence = confidence,
      election_weight = election_weight
    )
  }

  for (d in seq_len(ndraws)) {
    for (p_idx in seq_len(nprec)) {
      precinct_draw_rows[[length(precinct_draw_rows) + 1]] <- data.frame(
        election_id = eid,
        draw = d,
        vtd_geoid = merged$vtd_geoid[[p_idx]],
        json = toJSON(as.list(as.data.frame(arr[d, p_idx, , , drop = FALSE])), auto_unbox = TRUE)
      )
    }
  }
}

pref_df <- bind_rows(group_pref_rows)
draw_df <- bind_rows(precinct_draw_rows)
write_json(list(group_preferences = pref_df, precinct_draws = draw_df), output_path, auto_unbox = TRUE)
'''


def build_king_brennan_inputs(
    con: duckdb.DuckDBPyConnection,
    candidate_returns_path: str,
    elections_path: str,
    out_dir: str,
    group_cols: dict[str, str] | None = None,
) -> dict[str, Any]:
    group_cols = group_cols or {
        "HCVAP": "cvap_hisp",
        "BCVAP": "cvap_nh_black",
        "WCVAP": "cvap_nh_white",
        "OCVAP": "cvap_other",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cvap = con.execute(
        f"""
        SELECT vtd_geoid, cvap_total,
               {group_cols['HCVAP']} AS HCVAP,
               {group_cols['BCVAP']} AS BCVAP,
               {group_cols['WCVAP']} AS WCVAP,
               {group_cols['OCVAP']} AS OCVAP
        FROM geo_vtd
        """
    ).df()
    cvap_csv = out / "cvap_inputs.csv"
    cvap.to_csv(cvap_csv, index=False)

    candidate_returns = pd.read_parquet(candidate_returns_path) if candidate_returns_path.endswith(".parquet") else pd.read_csv(candidate_returns_path)
    returns_csv = out / "candidate_returns.csv"
    candidate_returns.to_csv(returns_csv, index=False)

    elections = pd.read_parquet(elections_path) if elections_path.endswith(".parquet") else pd.read_csv(elections_path)
    weights_csv = out / "election_weights.csv"
    elections[[c for c in elections.columns if c in {"election_id", "weight"}]].assign(weight=lambda d: d.get("weight", 1.0)).to_csv(weights_csv, index=False)

    return {
        "cvap_csv": str(cvap_csv),
        "returns_csv": str(returns_csv),
        "weights_csv": str(weights_csv),
        "groups": ["HCVAP", "BCVAP", "WCVAP", "OCVAP"],
    }


def run_king_ei_multielection(
    con: duckdb.DuckDBPyConnection,
    candidate_returns_path: str,
    elections_path: str,
    output_json: str,
    rscript_bin: str = "Rscript",
) -> str:
    if shutil.which(rscript_bin) is None:
        raise RuntimeError("Rscript was not found. Install R and eiPack to run King's EI via ei.MD.bayes.")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg = build_king_brennan_inputs(con, candidate_returns_path, elections_path, td)
        input_json = td_path / "king_input.json"
        script_path = td_path / "run_ei.R"
        input_json.write_text(json.dumps(cfg), encoding="utf-8")
        script_path.write_text(R_SCRIPT, encoding="utf-8")

        proc = subprocess.run(
            [rscript_bin, str(script_path), str(input_json), str(output_json)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"King EI R job failed. stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
    return output_json
