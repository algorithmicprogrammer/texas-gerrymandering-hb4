# =============================================================================
# run_ei.R
# -----------------------------------------------------------------------------
# Reproduces the RxC Multinomial-Dirichlet ecological inference from
# Becker, Duchin, Gold & Hirsch (2021) "Computational Redistricting and
# the Voting Rights Act", Election Law Journal 20(4).
#
# Uses eiPack::ei.MD.bayes() — the same hierarchical Bayesian model used
# in the paper (via eiPack, which implements Rosen et al. 2001).
#
# Inputs (all in same directory as this script):
#   tx_precincts_for_ei.csv      — precinct data from data engineering pipeline
#   Candidate_Race_Party.csv     — candidate race/party lookup
#   TX_elections.csv             — election set definitions
#   recency_weights.csv          — year → weight mapping
#   dropped_elecs.csv            — election sets to exclude (may be empty)
#
# Outputs written to ei_outputs/:
#   statewide_rxc_EI_preferences.csv   — statewide candidate-of-choice by group
#   mean_prec_vote_counts.csv          — precinct-level mean vote count by race
#   prec_count_quants.csv              — precinct-level octile quantiles by race
# =============================================================================

library(eiPack)

set.seed(42)

# ── MCMC settings (match paper: ~10k effective draws) ─────────────────────────
SAMPLE  <- 10000   # posterior draws to keep
BURNIN  <- 10000   # burn-in draws to discard
THIN    <- 10      # keep every 10th → 1000 effective samples per chain
LAMBDA1 <- 4       # eiPack prior hyperparameter (paper default)
LAMBDA2 <- 2       # eiPack prior hyperparameter (paper default)
NTUNES  <- 10      # tuning rounds (paper default)

# ── Demographic group columns (R=4, must sum to 1 per precinct) ───────────────
GROUP_COLS   <- c("hisp_prop", "black_prop", "white_prop", "other_prop")
GROUP_LABELS <- c("Hispanic", "Black", "White", "Other")

# ── Election definitions: election key → (candidate vote cols, total col) ─────
# Keys match the Election column in TX_elections.csv.
# UncommittedR_24P is excluded — no race identity, excluded from CSV too.
ELECTIONS <- list(

  `24G_President` = list(
    cands = c("TrumpR_24G", "HarrisD_24G"),
    total = "TOTVOTE_PRES_24G"
  ),

  `24P_President` = list(
    cands = c("BidenD_24P", "CornejoD_24P", "LockeD_24P", "LozadaD_24P",
              "PerezD_24P", "PhillipsD_24P", "UygurD_24P", "WilliamsonD_24P",
              "BinkleyR_24P", "HaleyR_24P", "StuckenbergR_24P", "TrumpR_24P",
              "ChristieR_24P", "RamaswamyR_24P", "HutchinsonR_24P", "DeSantisR_24P"),
    total = NULL   # computed below as sum of all primary cands
  ),

  `24G_US_Sen` = list(
    cands = c("CruzR_24G", "AllredD_24G"),
    total = "TOTVOTE_SEN_24G"
  ),

  `24P_US_Sen` = list(
    cands = c("AllredD_24P", "GomezD_24P", "GonzalezD_24P", "GutierrezD_24P",
              "HassanD_24P", "KeoughD_24P", "PrillimanD_24P", "ShermanD_24P",
              "TchenkoD_24P",
              "CruzR_24P", "GibsonR_24P", "LopezR_24P"),
    total = NULL
  ),

  `24G_RR_Comm_1` = list(
    cands = c("CraddickR_24G", "CulbertD_24G"),
    total = "TOTVOTE_RRC_24G"
  ),

  `24P_RR_Comm_1` = list(
    cands = c("BurchD_24P", "CulbertD_24P",
              "ClarkR_24P", "CraddickR_24P", "HowellR_24P",
              "MatlockR_24P", "ReyesR_24P"),
    total = NULL
  )
)

# ── Load supporting files ─────────────────────────────────────────────────────
cat("Loading input files...\n")

vtd <- read.csv("tx_precincts_for_ei.csv", stringsAsFactors = FALSE)
crp <- read.csv("Candidate_Race_Party.csv", stringsAsFactors = FALSE)
elec_sets <- read.csv("TX_elections.csv", stringsAsFactors = FALSE,
                      check.names = FALSE)
recency <- read.csv("recency_weights.csv", stringsAsFactors = FALSE,
                    check.names = FALSE)
dropped_raw <- read.csv("dropped_elecs.csv", stringsAsFactors = FALSE,
                        check.names = FALSE)

# Parse dropped election sets (may be empty)
dropped_col <- colnames(dropped_raw)[1]
dropped_sets <- dropped_raw[[dropped_col]]
dropped_sets <- dropped_sets[!is.na(dropped_sets) & dropped_sets != ""]

# Parse recency weights: column name = year, value = weight
recency_year   <- as.integer(colnames(recency)[1])
recency_weight <- as.numeric(recency[1, 1])
cat(sprintf("  Recency weight: year=%d  weight=%.2f\n",
            recency_year, recency_weight))

# Drop excluded election sets from the election table
if (length(dropped_sets) > 0) {
  elec_sets <- elec_sets[!elec_sets[["Election Set"]] %in% dropped_sets, ]
  cat(sprintf("  Dropped %d election sets: %s\n",
              length(dropped_sets), paste(dropped_sets, collapse = ", ")))
}

# Keep only elections whose key is in ELECTIONS list
active_elections <- intersect(names(ELECTIONS), elec_sets[["Election"]])
cat(sprintf("  Active elections: %s\n", paste(active_elections, collapse = ", ")))

# Filter vtd to CVAP > 0
vtd <- vtd[!is.na(vtd$CVAP) & vtd$CVAP > 0, ]
cat(sprintf("  Precincts after CVAP>0 filter: %d\n", nrow(vtd)))

# ── Output directory ──────────────────────────────────────────────────────────
dir.create("ei_outputs", showWarnings = FALSE)

# ── Output accumulators ───────────────────────────────────────────────────────
statewide_rows   <- list()
mean_counts_list <- list()
quant_counts_list <- list()

# ── Main EI loop ──────────────────────────────────────────────────────────────
for (elec_key in active_elections) {

  spec      <- ELECTIONS[[elec_key]]
  cand_cols <- spec$cands
  total_col <- spec$total
  n_cands   <- length(cand_cols)

  cat("\n============================================================\n")
  cat(sprintf("Election: %s  (%d candidates)\n", elec_key, n_cands))

  # Verify all candidate columns exist
  missing_cols <- setdiff(cand_cols, colnames(vtd))
  if (length(missing_cols) > 0) {
    warning(sprintf("  Skipping %s — missing columns: %s",
                    elec_key, paste(missing_cols, collapse = ", ")))
    next
  }

  # Build total column if not pre-computed
  if (is.null(total_col)) {
    vtd[[paste0("TOTAL_", elec_key)]] <- rowSums(vtd[, cand_cols, drop = FALSE])
    total_col <- paste0("TOTAL_", elec_key)
  }

  # Keep only precincts with votes in this election
  sub <- vtd[vtd[[total_col]] > 0, ]
  n   <- nrow(sub)
  cat(sprintf("  Precincts with votes: %d\n", n))

  if (n < 50) {
    warning(sprintf("  Skipping %s — too few precincts (%d)", elec_key, n))
    next
  }

  # ── Build eiPack formula ────────────────────────────────────────────────────
  # LHS = cbind(candidate vote counts)
  # RHS = cbind(group fractions)
  lhs <- paste0("cbind(", paste(cand_cols, collapse = ", "), ")")
  rhs <- paste0("cbind(", paste(GROUP_COLS, collapse = ", "), ")")
  frm <- as.formula(paste(lhs, "~", rhs))

  # ── Tune MCMC ───────────────────────────────────────────────────────────────
  cat("  Tuning...\n")
  tune <- tuneMD(
    formula  = frm,
    data     = sub,
    total    = sub[[total_col]],
    ntunes   = NTUNES,
    ndraws   = 200,
    lambda1  = LAMBDA1,
    lambda2  = LAMBDA2,
    verbose  = FALSE
  )

  # ── Run MCMC ────────────────────────────────────────────────────────────────
  cat(sprintf("  Running MCMC (burnin=%d sample=%d thin=%d)...\n",
              BURNIN, SAMPLE, THIN))
  out <- ei.MD.bayes(
    formula   = frm,
    data      = sub,
    total     = sub[[total_col]],
    tune.list = tune,
    lambda1   = LAMBDA1,
    lambda2   = LAMBDA2,
    sample    = SAMPLE,
    burnin    = BURNIN,
    thin      = THIN,
    ret.beta  = "r",      # return precinct-level Beta draws as R object
    ret.mcmc  = FALSE,    # return as arrays (not coda objects)
    verbose   = 1000
  )

  # ── Extract Beta array ──────────────────────────────────────────────────────
  # dim: [n_groups, n_cands, n_precincts, n_eff_draws]
  # Beta[g, c, i, s] = fraction of group g voting for candidate c
  #                    in precinct i at draw s
  Beta    <- out$draws$Beta
  n_draws <- dim(Beta)[4]
  cvap    <- sub$CVAP

  cat(sprintf("  Beta array: %s  (groups x cands x precincts x draws)\n",
              paste(dim(Beta), collapse=" x ")))

  # ── 1. statewide_rxc_EI_preferences ────────────────────────────────────────
  cat("  Computing statewide preferences...\n")

  for (g_idx in seq_along(GROUP_LABELS)) {
    group     <- GROUP_LABELS[g_idx]
    group_pop <- sub[[GROUP_COLS[g_idx]]] * cvap   # group CVAP count per precinct

    # For each draw: state-level support = weighted mean of Beta across precincts
    total_group_pop <- sum(group_pop)
    draw_state_support <- matrix(0, nrow = n_draws, ncol = n_cands)

    for (s in seq_len(n_draws)) {
      for (c_idx in seq_len(n_cands)) {
        draw_state_support[s, c_idx] <-
          sum(Beta[g_idx, c_idx, , s] * group_pop) / total_group_pop
      }
    }

    # Mean support per candidate over draws
    mean_support <- colMeans(draw_state_support)

    # Fraction of draws each candidate was preferred (argmax)
    draw_coc <- apply(draw_state_support, 1, which.max)
    frac_preferred <- tabulate(draw_coc, nbins = n_cands) / n_draws

    # Candidate of choice = highest mean support
    coc_idx  <- which.max(mean_support)
    coc_name <- cand_cols[coc_idx]

    statewide_rows[[length(statewide_rows) + 1]] <- data.frame(
      election             = elec_key,
      group                = group,
      candidate_of_choice  = coc_name,
      mean_support         = round(mean_support[coc_idx], 6),
      frac_draws_preferred = round(frac_preferred[coc_idx], 6),
      stringsAsFactors     = FALSE
    )
  }

  # ── 2. mean_prec_vote_counts ────────────────────────────────────────────────
  # Mean over draws of (Beta * group_pop) = estimated vote count by race
  cat("  Computing mean precinct vote counts...\n")

  for (g_idx in seq_along(GROUP_LABELS)) {
    group     <- GROUP_LABELS[g_idx]
    group_pop <- sub[[GROUP_COLS[g_idx]]] * cvap

    for (c_idx in seq_len(n_cands)) {
      # matrix: n_precincts x n_draws of vote count estimates
      count_matrix <- sapply(seq_len(n_draws),
                             function(s) Beta[g_idx, c_idx, , s] * group_pop)
      mean_counts <- rowMeans(count_matrix)

      mean_counts_list[[length(mean_counts_list) + 1]] <- data.frame(
        CNTYVTD    = sub$CNTYVTD,
        election   = elec_key,
        group      = group,
        candidate  = cand_cols[c_idx],
        mean_count = round(mean_counts, 4),
        stringsAsFactors = FALSE
      )
    }
  }

  # ── 3. prec_count_quants ────────────────────────────────────────────────────
  # Octile (1/8 through 8/8) quantiles of the draw distribution
  cat("  Computing precinct count quantiles...\n")
  octile_probs <- (1:8) / 8

  for (g_idx in seq_along(GROUP_LABELS)) {
    group     <- GROUP_LABELS[g_idx]
    group_pop <- sub[[GROUP_COLS[g_idx]]] * cvap

    for (c_idx in seq_len(n_cands)) {
      count_matrix <- sapply(seq_len(n_draws),
                             function(s) Beta[g_idx, c_idx, , s] * group_pop)

      for (q_idx in seq_along(octile_probs)) {
        quant_vals <- apply(count_matrix, 1, quantile,
                            probs = octile_probs[q_idx])

        quant_counts_list[[length(quant_counts_list) + 1]] <- data.frame(
          CNTYVTD   = sub$CNTYVTD,
          election  = elec_key,
          group     = group,
          candidate = cand_cols[c_idx],
          octile    = q_idx,
          quantile  = octile_probs[q_idx],
          value     = round(quant_vals, 4),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  cat(sprintf("  Done: %s\n", elec_key))

  # Free memory before next election
  rm(out, Beta, draw_state_support, count_matrix)
  gc()
}

# ── Write outputs ──────────────────────────────────────────────────────────────
cat("\nWriting outputs to ei_outputs/...\n")

statewide_df <- do.call(rbind, statewide_rows)
write.csv(statewide_df,
          "ei_outputs/statewide_rxc_EI_preferences.csv",
          row.names = FALSE)
cat(sprintf("  statewide_rxc_EI_preferences.csv: %d rows\n", nrow(statewide_df)))

mean_df <- do.call(rbind, mean_counts_list)
write.csv(mean_df,
          "ei_outputs/mean_prec_vote_counts.csv",
          row.names = FALSE)
cat(sprintf("  mean_prec_vote_counts.csv: %d rows\n", nrow(mean_df)))

quant_df <- do.call(rbind, quant_counts_list)
write.csv(quant_df,
          "ei_outputs/prec_count_quants.csv",
          row.names = FALSE)
cat(sprintf("  prec_count_quants.csv: %d rows\n", nrow(quant_df)))

cat("\n=== EI complete ===\n")
cat("Next step: fill in ingroup_weight.csv using statewide_rxc_EI_preferences.csv,\n")
cat("then proceed to TX_elections_model.py\n")
