#!/usr/bin/env python
#
# Loads ATAC signals, log-transforms for testing, applies independent filtering,
# runs Welch's t-tests with BH-FDR, and computes:
#   - standard log2FC from RAW means
#   - raw mean difference (group2 - group1)
#   - log-mean difference (group2 - group1)

import h5py
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests
import os

# 1)set-up
h5_path = "signals_with_replicates.h5"
output_dir = "t_test_results_log_only"
os.makedirs(output_dir, exist_ok=True)

ALPHA = 0.05            # FDR threshold
FILTER_BY = "mean"      # "mean" or "var" (computed on log scale across all samples in both groups)
FILTER_QUANTILE = 0.25  # drop bottom 25% by the chosen filter statistic (independent filtering)

sample_types = ["orer", "cic6", "zldrnai", "gd7"]
mutants = ["cic6", "zldrnai", "gd7"]
time_points = ["1to15", "15to2", "2to25", "25to3"]

# 2) load raw data 
print(f"Loading raw replicate data from {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    signal_matrix = f['signal_matrix'][:]  
    experiment_types = [s.decode('utf-8') for s in f['experiment_types'][:]]
    regions_df = pd.DataFrame({
        "chr": [s.decode('utf-8') for s in f['region_chr'][:]],
        "start": f["region_start"][:],
        "end": f["region_end"][:],
    })
print("Data loaded successfully.")

# 3) pre-process the data (log for testing; keep raw for FC)
print("\nPre-processing Data")
print("Applying log1p transformation for testing...")
signal_log = np.log1p(signal_matrix)  # for t-tests
signal_raw = signal_matrix            # keep true raw for fold-change and raw means
print("Preprocessing complete.")


def run_contrast(name1, name2, label1, label2, cols1, cols2):
    """Run filtering, t-tests, FDR; compute log2FC and mean differences for one contrast."""
    # Independent filtering stat on all columns in both groups (log scale)
    all_cols = cols1 + cols2
    row_mean_all = np.nanmean(signal_log[:, all_cols], axis=1)
    row_var_all  = np.nanvar(signal_log[:, all_cols], axis=1, ddof=1)
    filt_stat = row_mean_all if FILTER_BY == "mean" else row_var_all
    thresh = np.nanquantile(filt_stat, FILTER_QUANTILE)
    keep_mask = filt_stat > thresh    # rows kept for testing

    # Precompute group means (for reporting) on log and raw scales
    mean1_log = np.nanmean(signal_log[:, cols1], axis=1)
    mean2_log = np.nanmean(signal_log[:, cols2], axis=1)
    mean1_raw = np.nanmean(signal_raw[:, cols1], axis=1)
    mean2_raw = np.nanmean(signal_raw[:, cols2], axis=1)

    # Standard log2FC from RAW means (+1 to avoid div-by-zero)
    log2fc = np.log2((mean2_raw + 1.0) / (mean1_raw + 1.0))
    # Raw mean difference (group2 - group1)
    mean_diff_raw = mean2_raw - mean1_raw
    # Log-mean difference (group2 - group1) on log1p scale
    mean_diff_log = mean2_log - mean1_log

    # Row-wise tests only on kept rows
    n_rows = signal_log.shape[0]
    pvals = np.full(n_rows, np.nan, dtype=float)

    kept_indices = np.where(keep_mask)[0]
    for i in kept_indices:
        x = signal_log[i, cols1]
        y = signal_log[i, cols2]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        stat, p = ttest_ind(x, y, equal_var=True, nan_policy="omit") #CHANGED TO TRUE BUT DID NOT RUN AGAIN
        pvals[i] = p

    # Assemble output DataFrame
    df = regions_df.copy()
    df["p_value"] = pvals
    df[f"mean_{label1}_log"] = mean1_log
    df[f"mean_{label2}_log"] = mean2_log
    df[f"mean_{label1}_raw"] = mean1_raw
    df[f"mean_{label2}_raw"] = mean2_raw
    df["log2_fold_change"] = log2fc
    df["mean_diff_raw"] = mean_diff_raw
    df["mean_diff_log"] = mean_diff_log
    df["kept_for_testing"] = keep_mask

    # FDR only on rows we actually tested
    tested = df["kept_for_testing"] & df["p_value"].notna()
    df["q_value"] = np.nan
    if tested.any():
        _, q, _, _ = multipletests(df.loc[tested, "p_value"], alpha=ALPHA, method="fdr_bh")
        df.loc[tested, "q_value"] = q

    # significant = q < ALPHA (no effect-size threshold here)
    sig = df[df["q_value"] < ALPHA].copy()
    return df, sig

# 4) Time point comparisons (adjacent)
print("\nRunning Time Point Comparisons (adjacent)")
for sample in sample_types:
    for tp1, tp2 in zip(time_points, time_points[1:]):
        g1 = f"{sample}_{tp1}_atac"
        g2 = f"{sample}_{tp2}_atac"
        cols1 = [i for i, n in enumerate(experiment_types) if n.startswith(g1)]
        cols2 = [i for i, n in enumerate(experiment_types) if n.startswith(g2)]
        if not cols1 or not cols2:
            continue
        print(f"Comparing {g1} vs {g2} ...")
        all_df, sig_df = run_contrast(g1, g2, tp1, tp2, cols1, cols2)
        if all_df is None:
            continue
        base = os.path.join(output_dir, f"t-test_timepoint_{sample}_{tp1}_vs_{tp2}")
        all_df.sort_values("p_value", na_position="last").to_csv(base+"_ALL.tsv", sep="\t", index=False, float_format="%.4g")
        sig_df.sort_values("q_value").to_csv(base+"_SIGNIFICANT.tsv", sep="\t", index=False, float_format="%.4g")
        print(f"-> kept {int(all_df['kept_for_testing'].sum())} rows; found {len(sig_df)} significant")

# 5) Mutant vs Orer comparisons
print("\nRunning Mutant vs. Wild-type Comparisons")
for tp in time_points:
    for mutant in mutants:
        wt = f"orer_{tp}_atac"
        mt = f"{mutant}_{tp}_atac"
        cols1 = [i for i, n in enumerate(experiment_types) if n.startswith(wt)]
        cols2 = [i for i, n in enumerate(experiment_types) if n.startswith(mt)]
        if not cols1 or not cols2:
            continue
        print(f"Comparing {wt} vs {mt} ...")
        all_df, sig_df = run_contrast(wt, mt, "WT", mutant, cols1, cols2)
        if all_df is None:
            continue
        base = os.path.join(output_dir, f"t-test_mutant_{mutant}_vs_orer_{tp}")
        all_df.sort_values("p_value", na_position="last").to_csv(base+"_ALL.tsv", sep="\t", index=False, float_format="%.4g")
        sig_df.sort_values("q_value").to_csv(base+"_SIGNIFICANT.tsv", sep="\t", index=False, float_format="%.4g")
        print(f"-> kept {int(all_df['kept_for_testing'].sum())} rows; found {len(sig_df)} significant")

print("\nAll comparisons completed successfully.")
