#!/usr/bin/env python
# This script trains a gReLU model on ChIP-nexus data, applying quantile normalization.

import os
import numpy as np
import pandas as pd
import torch
import h5py
import anndata as ad
import sys
import glob

# Add gReLU to path
sys.path.insert(0, "../gReLU/src")
import grelu.resources
import grelu.lightning
from grelu.data.dataset import AnnDataSeqDataset

experiment = "final_chip_model_2"
if not os.path.exists(experiment):
    os.makedirs(experiment)

training = True
seq_len = 2000
genome = "dm6"

# Load Data
print("Loading data...")
h5_path = "combined_all_signals_after_filtering.h5"
with h5py.File(h5_path, "r") as f:
    signal_matrix = f["signal_matrix"][:]

bed_path = "filtered_peaks.bed"
bed = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom", "start", "end"])

# Pre-processing
# 1. Filter by trusted chromosomes (CRITICAL FIX)
trusted_chroms = ['chr2L', 'chr2R', 'chr3L', 'chr3R']
keep_chroms_mask = bed['chrom'].isin(trusted_chroms)
bed = bed[keep_chroms_mask].reset_index(drop=True)
signal_matrix = signal_matrix[keep_chroms_mask.values, :]
assert signal_matrix.shape[0] == len(bed)

# 2. Remove out-of-boundary peaks
print("Filtering out-of-boundary peaks...")
genome_file = "data/dm6.fa.fai"
chromsizes = {line.split("\t")[0]: int(line.split("\t")[1]) for line in open(genome_file)}
margin = seq_len // 2
bed['chrom_size'] = bed['chrom'].map(chromsizes)
bed['center'] = bed['start'] + (bed['end'] - bed['start']) // 2
keep_mask = (bed['center'] >= margin) & (bed['center'] <= (bed['chrom_size'] - margin))
bed = bed[keep_mask].reset_index(drop=True)
signal_matrix = signal_matrix[keep_mask.values, :]
assert signal_matrix.shape[0] == len(bed)

# 3. Filter to ChIP-nexus columns
all_task_names = [
    "orer_gaf_mbt_nexus", "zldrnai_15to2_atac", "orer_dl_mbt_nexus",
    "zldrnai_1to15_atac", "orer_zld_mbt_nexus", "gd7_1to15_atac",
    "orer_h3k27ac_mbt_seq", "gd7_15to2_atac", "orer_2to25_atac",
    "orer_cad_mbt_nexus", "zldrnai_25to3_atac", "gd7_2to25_atac",
    "orer_wce_mbt_seq", "orer_1to15_atac", "orer_twi_mbt_nexus",
    "cic6_2to25_atac", "orer_25to3_atac", "zldrnai_2to25_atac",
    "orer_bcd_mbt_nexus", "cic6_25to3_atac", "orer_15to2_atac",
    "gd7_25to3_atac", "orer_h3k4me1_mbt_seq", "orer_mbt_100u_mnase"
]
tasks_series = pd.Series(all_task_names)
chip_mask = tasks_series.str.contains("nexus")
chip_task_names = tasks_series[chip_mask].tolist()
chip_indices = tasks_series.index[chip_mask].tolist()
signal_matrix_chip = signal_matrix[:, chip_indices]
print(f"Selected {len(chip_task_names)} ChIP-nexus tasks.")

# 4. Quantile normalize (vectorized for speed)
print("Applying quantile normalization...")
def quantile_normalize_vectorized(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    ranked_mean = sorted_matrix.mean(axis=1)
    ranks = np.argsort(np.argsort(matrix, axis=0), axis=0)
    return ranked_mean[ranks]

signal_matrix_chip = quantile_normalize_vectorized(signal_matrix_chip)

# 5. Create AnnData object
adata = ad.AnnData(
    X=signal_matrix_chip.T,
    obs=pd.DataFrame({"task_name": chip_task_names}),
    var=bed.drop(columns=['chrom_size', 'center'])
)
adata.var.index = adata.var["chrom"] + "_" + adata.var["start"].astype(str) + "_" + adata.var["end"].astype(str)
adata.obs.index = adata.obs["task_name"]

# 6. Split data (cleaner method)
train_mask = adata.var["chrom"].isin(['chr2L', 'chr3L', 'chr3R'])
ad_train = adata[:, train_mask].copy()

chr2R_mask = adata.var["chrom"] == "chr2R"
ad_chr2R = adata[:, chr2R_mask].copy()

n_total = ad_chr2R.n_vars
indices = np.arange(n_total)
np.random.seed(123) # for reproducibility
np.random.shuffle(indices)
val_indices = indices[:n_total//2]
test_indices = indices[n_total//2:]

ad_val = ad_chr2R[:, val_indices].copy()
ad_test = ad_chr2R[:, test_indices].copy()
print(f"Split data into: Train ({ad_train.n_vars}), Val ({ad_val.n_vars}), Test ({ad_test.n_vars})")

# Create Datasets
print("Creating PyTorch datasets...")
train_dataset = AnnDataSeqDataset(ad_train, genome=genome, seq_len=seq_len, rc=True)
val_dataset = AnnDataSeqDataset(ad_val, genome=genome, seq_len=seq_len)
test_dataset = AnnDataSeqDataset(ad_test, genome=genome, seq_len=seq_len)

# Define Model
print("Defining model...")
model_params = {
    'model_type': 'ConvMLPModel',
    'n_tasks': adata.n_obs,
    'seq_len': seq_len,
    'n_conv': 5, 'stem_channels': 256, 'channel_init': 192,
    'channel_mult': 1.1, 'kernel_size': 9, 'pool_func': "avg",
    'pool_size': 4, 'conv_norm': True, 'act_func': "gelu",
    'mlp_hidden_size': [1024, 512], 'mlp_norm': True, 'dropout': 0.1,
}
train_params = {
    'task': 'regression', 'loss': 'mse', 'lr': 1e-3, 'batch_size': 128,
    'max_epochs': 100, 'logger': 'csv', 'save_dir': experiment,
    'early_stopping_monitor': 'val_loss', 'early_stopping_patience': 5,
    'early_stopping_mode': 'min', 'checkpoint': True,
    'num_workers': 4, 'devices': [0],
}
model = grelu.lightning.LightningModel(model_params, train_params)

# Train or Load Model
if training:
    print("Starting training...")
    trainer = model.train_on_dataset(train_dataset=train_dataset, val_dataset=val_dataset)
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
else:
    print("Skipping training, loading last checkpoint...")
    checkpoint_files = glob.glob(f"{experiment}/**/*.ckpt", recursive=True)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in '{experiment}'.")
    best_checkpoint_path = checkpoint_files[0]

print(f"Loading model from: {best_checkpoint_path}")
model = grelu.lightning.LightningModel.load_from_checkpoint(best_checkpoint_path)

# Evaluate
print("\nEvaluating model performance on test set...")
test_metrics = model.test_on_dataset(dataset=test_dataset, devices=[0], num_workers=4, batch_size=128)
print("\n--- Evaluation Complete ---")
print("Test set performance metrics:")
print(test_metrics)