#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ATAC Performance Plots: Model vs Replicates

h5_mean_path = "combined_all_signals_after_filtering.h5"
h5_reps_path = "signals_with_replicates.h5"
pred_dir     = "per_task_predictions_atac"   # files: pred_<task>_atac.csv

out_prefix   = "performance_plots_atac"
pred_column  = None   
tasks_file   = None  


# In[2]:


import re, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    a = a[m]; b = b[m]
    if a.std()==0 or b.std()==0: return np.nan
    return float(np.corrcoef(a, b)[0,1])

def _decode(arr):
    return [x.decode() if isinstance(x,(bytes,bytearray)) else str(x) for x in arr]


# In[3]:


preds = read_predictions(Path(pred_dir), pred_column)
signals_mean, signals_reps, task_names = load_signals(Path(h5_path))

print(f"Loaded predictions for {len(preds)} tasks.")
print("Task names from H5:", "found" if task_names is not None else "NOT found")

if task_names is not None:
    print(f"#tasks in H5: {len(task_names)}")


# In[ ]:


def load_mean_signals(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        X = f["signal_matrix"][...]                           # (N, Tmean)
        chrom = _decode(f["region_chr"][...])                 # (N,)
        start = f["region_start"][...].astype(np.int64)
        end   = f["region_end"][...].astype(np.int64)
        names = _decode(f["experiment_types"][...]) if "experiment_types" in f else None
    reg_df = pd.DataFrame({"chr":chrom, "start":start, "end":end})
    reg_df["id_colon"]  = reg_df["chr"] + ":" + reg_df["start"].astype(str) + "-" + reg_df["end"].astype(str)   # chr:start-end
    reg_df["id_underscore"] = reg_df["chr"] + "_" + reg_df["start"].astype(str) + "_" + reg_df["end"].astype(str) # chr_start_end
    return X, names, reg_df

def load_rep_signals(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        Xreps = f["signal_matrix"][...]                      # (N, Treps)
        names = _decode(f["experiment_types"][...]) if "experiment_types" in f else None
    return Xreps, names


# In[ ]:


def read_prediction_csv(p: Path, pred_col: Optional[str]):
    """
    Returns: task_name, y_pred(np.ndarray), ids(pd.Series or None), df
    - Region identity is taken from (chrom,start,end) if present
      or from 'region_id' if present (supports 'chr:start-end' or 'chr_start_end')
    - Prediction column: if pred_col is None, auto-pick the column whose name starts with 'pred_';
      if none, fall back to first numeric column.
    """
    task = p.name[len("pred_"):-len(".csv")]
    df = pd.read_csv(p)

    # pick prediction column
    if pred_col and pred_col in df.columns:
        y = df[pred_col].to_numpy()
    else:
        pred_cols = [c for c in df.columns if c.lower().startswith("pred_")]
        if len(pred_cols) == 1:
            y = df[pred_cols[0]].to_numpy()
        elif len(pred_cols) > 1:
            # try to pick the one matching the file's task name
            candidates = [c for c in pred_cols if c.endswith(task)]
            y = df[candidates[0]].to_numpy() if candidates else df[pred_cols[0]].to_numpy()
        else:
            # fallback: first numeric col
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise ValueError(f"No numeric columns in {p}")
            y = df[num_cols[0]].to_numpy()

    # region identifiers
    lc = {c.lower(): c for c in df.columns}
    ids = None
    if "chrom" in lc and "start" in lc and "end" in lc:
        ids = (df[lc["chrom"]].astype(str) + ":" +
               df[lc["start"]].astype(str) + "-" +
               df[lc["end"]].astype(str))
    elif "region_id" in lc:
        rid = df[lc["region_id"]].astype(str)
        if rid.str.contains(":").any() and rid.str.contains("-").any():
            ids = rid  # already chr:start-end
        else:
            # assume chr_start_end, convert to chr:start-end
            parts = rid.str.split("_", n=2, expand=True)
            if parts.shape[1] == 3:
                ids = parts[0] + ":" + parts[1] + "-" + parts[2]
            else:
                ids = rid  # last-ditch: carry as-is, we'll try both keys
    else:
        print(f"[WARN] {p.name}: no (chrom,start,end) or region_id; cannot align.")
        ids = None

    return task, y, ids, df

def list_prediction_files(pred_dir: Path):
    files = sorted(pred_dir.glob("pred_*_atac.csv"))
    if not files:
        raise RuntimeError(f"No pred_*_atac.csv in {pred_dir}")
    return files


# In[ ]:


import re
BASE_RE = re.compile(r"^(orer|zldrnai|gd7|cic6)_(\d+to\d+)_atac", re.IGNORECASE)

def base_task_name(s: str):
    m = BASE_RE.search(s)
    return f"{m.group(1)}_{m.group(2)}_atac" if m else None

def condition_and_time(task: str):
    m = BASE_RE.search(task)
    return (m.group(1), m.group(2)) if m else ("other","NA")

def group_rep_columns(rep_names):
    from collections import defaultdict
    buckets = defaultdict(list)
    for j, nm in enumerate(rep_names):
        b = base_task_name(nm)
        if b:
            buckets[b].append(j)
    return dict(buckets)


# In[ ]:


# Load H5
X_mean, names_mean, reg_df = load_mean_signals(h5_mean_path)   # (N, 24)
X_reps, names_reps = load_rep_signals(h5_reps_path)            # (N, 68)
assert X_reps.shape[0] == X_mean.shape[0], "Region counts differ between H5s."
N = len(reg_df)

# Fast maps from region id strings to H5 row index
id_to_row_colon = pd.Series(np.arange(N, dtype=int), index=reg_df["id_colon"]).to_dict()
id_to_row_us    = pd.Series(np.arange(N, dtype=int), index=reg_df["id_underscore"]).to_dict()

# mean: map base-task → column
mean_base_to_col = {}
if names_mean:
    for j, nm in enumerate(names_mean):
        b = base_task_name(nm)
        if b and b not in mean_base_to_col:
            mean_base_to_col[b] = j

# reps: map base-task → list of replicate columns
rep_base_to_cols = group_rep_columns(names_reps or [])

# read preds
pred_files = list_prediction_files(Path(pred_dir))
if tasks_file:
    keep = set([t.strip() for t in Path(tasks_file).read_text().splitlines() if t.strip()])
    pred_files = [p for p in pred_files if p.name[len("pred_"):-len(".csv")] in keep]

rows = []
for p in pred_files:
    task, yhat, ids, dfp = read_prediction_csv(p, pred_column)
    cond, timebin = condition_and_time(task)
    base = base_task_name(task)

    if ids is None:
        print(f"[SKIP] {task}: cannot align (no region identifiers).")
        continue

    # map each id to H5 row index (try colon and underscore keys)
    h5_rows = []
    miss = 0
    for s in ids:
        i = id_to_row_colon.get(s)
        if i is None:
            # try underscore variant
            if "_" in s and ":" not in s:
                i = id_to_row_us.get(s)
            else:
                # also try converting colon↔underscore quickly
                if ":" in s and "-" in s:
                    parts = s.replace(":", "_").replace("-", "_")
                    i = id_to_row_us.get(parts)
        if i is None:
            miss += 1
        else:
            h5_rows.append(i)
    h5_rows = np.array(h5_rows, dtype=int)
    if len(h5_rows) < 50:
        print(f"[WARN] {task}: only {len(h5_rows)}/{len(ids)} rows matched. Check coordinates.")
        if len(h5_rows) == 0:
            continue

    # align yhat to matched rows (assumes same order as ids)
    if len(yhat) != len(ids):
        print(f"[WARN] {task}: pred length {len(yhat)} != ids {len(ids)}; truncating to min.")
    m = min(len(yhat), len(ids), len(h5_rows))
    yhat_sel = np.asarray(yhat[:m], dtype=float)
    rows_sel = h5_rows[:m]

    # Model vs Mean R
    r_model = np.nan
    col_mean = mean_base_to_col.get(base)
    if col_mean is not None:
        r_model = _pearsonr(yhat_sel, X_mean[rows_sel, col_mean])
    else:
        
        rs = [ _pearsonr(yhat_sel, X_mean[rows_sel, j]) for j in range(X_mean.shape[1]) ]
        j_best = int(np.nanargmax(np.abs(rs)))
        r_model = float(rs[j_best])
        col_mean = j_best
        print(f"[AUTO-MAP] {task} → mean col {j_best} (R={r_model:.3f})")


    # Replicates R (mean pairwise across replicate columns for this base, on matched rows)
    r_reps = np.nan
    rep_cols = rep_base_to_cols.get(base, [])
    if len(rep_cols) >= 2:
        pairs = []
        for a in range(len(rep_cols)):
            for b in range(a+1, len(rep_cols)):
                v1 = X_reps[rows_sel, rep_cols[a]]
                v2 = X_reps[rows_sel, rep_cols[b]]
                pairs.append(_pearsonr(v1, v2))
        if pairs:
            r_reps = float(np.nanmean(pairs))
    else:
        pass  # optional debug print


    rows.append(dict(task=task, condition=cond, time=timebin, group="Model",       R=r_model))
    rows.append(dict(task=task, condition=cond, time=timebin, group="Replicates", R=r_reps))

df = pd.DataFrame(rows)
display(df.head(20))
df.to_csv(f"{out_prefix}_per_task_R.csv", index=False)
print("Wrote:", f"{out_prefix}_per_task_R.csv")


# In[ ]:


# Summary (all tasks)
fig, ax = plt.subplots(figsize=(6,6), dpi=140)
groups = ["Model", "Replicates"]
data = [df.loc[df.group==g, "R"].dropna().to_numpy() for g in groups]
if all(len(d)==0 for d in data):
    print("⚠️ No R values to plot.")
else:
    box = ax.boxplot(data, tick_labels=groups, patch_artist=True)
    for patch, c in zip(box['boxes'], ["#8db6ff", "#555555"]):
        patch.set_facecolor(c); patch.set_alpha(0.9)
    ax.set_ylabel("Pearson R"); ax.set_title("ATAC performance (all tasks)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_summary_boxes.png", bbox_inches="tight")
    print("Wrote:", f"{out_prefix}_summary_boxes.png")
    plt.show()

# Per-condition
conds = ["orer","zldrnai","gd7","cic6"]
pos = np.arange(len(conds)); width = 0.35
y_model = [df[(df.condition==c)&(df.group=="Model")]["R"].dropna().to_numpy() for c in conds]
y_reps  = [df[(df.condition==c)&(df.group=="Replicates")]["R"].dropna().to_numpy() for c in conds]

if all(len(y)==0 for y in y_model+y_reps):
    print("⚠️ No per-condition data to plot.")
else:
    fig2, ax2 = plt.subplots(figsize=(8,6), dpi=140)
    bp1 = ax2.boxplot(y_model, positions=pos-width/2, widths=width*0.9, patch_artist=True)
    bp2 = ax2.boxplot(y_reps,  positions=pos+width/2, widths=width*0.9, patch_artist=True)
    for p in bp1['boxes']: p.set_facecolor("#8db6ff"); p.set_alpha(0.9)
    for p in bp2['boxes']: p.set_facecolor("#555555"); p.set_alpha(0.9)
    ax2.set_xticks(pos); ax2.set_xticklabels(conds)
    ax2.set_ylabel("Pearson R"); ax2.set_title("ATAC performance per condition")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(handles=[plt.Rectangle((0,0),1,1, color="#8db6ff"),
                        plt.Rectangle((0,0),1,1, color="#555555")],
               labels=["Model","Replicates"], loc="lower right", frameon=False)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_per_condition_boxes.png", bbox_inches="tight")
    print("Wrote:", f"{out_prefix}_per_condition_boxes.png")
    plt.show()


# In[ ]:


import h5py, numpy as np, pandas as pd
from pathlib import Path

h5_path = Path("signals_with_replicates.h5")

def walk_h5(h5, path="/", max_items=200):
    out = []
    def _walk(g, prefix=""):
        for k in g.keys():
            obj = g[k]
            full = f"{prefix}{k}"
            if isinstance(obj, h5py.Group):
                out.append(("GROUP", full, "", ""))
                _walk(obj, prefix=full + "/")
            else:
                shp = getattr(obj, "shape", None)
                dt  = getattr(obj, "dtype", None)
                out.append(("DATASET", full, shp, str(dt)))
    _walk(h5[path])
    return out

with h5py.File(h5_path, "r") as f:
    rows = walk_h5(f)
    print(f"Top-level keys: {list(f.keys())}")
    # Show first ~200 items
    for kind, name, shp, dt in rows[:200]:
        print(f"{kind:8s}  {name:50s}  shape={shp}  dtype={dt}")
    # Check for obvious fields
    for key in ("task_names","tasks","all_task_names","signal_matrix","signal_matrix_reps"):
        print(f"[check] key '{key}':", key in f)
    # If attrs exist:
    try:
        if "signal_matrix" in f and hasattr(f["signal_matrix"], "attrs"):
            print("signal_matrix attrs:", dict(f["signal_matrix"].attrs))
    except Exception as e:
        print("attr read error:", e)


# In[ ]:


import h5py, numpy as np, pandas as pd, re
from collections import Counter

with h5py.File("signals_with_replicates.h5", "r") as f:
    Xreps = f["signal_matrix"][...]
    exp_types = f["experiment_types"][...]
    exp_types = [et.decode() if isinstance(et, (bytes, bytearray)) else str(et) for et in exp_types]

print("Xreps shape (regions, columns):", Xreps.shape)
print("#experiment_types:", len(exp_types))
print("Does length match #columns? -->", len(exp_types) == Xreps.shape[1])

print("\nFirst 20 experiment_types:")
for s in exp_types[:20]:
    print("  ", s)


# In[ ]:


# Accept anything after "..._atac"
BASE_RE = re.compile(r"^(orer|zldrnai|gd7|cic6)_(\d+to\d+)_atac", re.IGNORECASE)

def base_task_name(s: str):
    m = BASE_RE.search(s)
    return f"{m.group(1)}_{m.group(2)}_atac" if m else None

# Group replicate columns by base task
from collections import defaultdict
rep_base_to_cols = defaultdict(list)
for j, nm in enumerate(exp_types):
    b = base_task_name(nm)
    if b:
        rep_base_to_cols[b].append(j)

# Show a summary: how many replicate columns per base task?
counts = {b: len(js) for b, js in rep_base_to_cols.items()}
print("Replicate counts per base task (should be ~3 each):")
for b, c in sorted(counts.items()):
    print(f"  {b:20s} -> {c}")


example_base = next(iter(rep_base_to_cols.keys()), None)
if example_base:
    print(f"\nExample base: {example_base}")
    for j in rep_base_to_cols[example_base]:
        print("   ", exp_types[j])

