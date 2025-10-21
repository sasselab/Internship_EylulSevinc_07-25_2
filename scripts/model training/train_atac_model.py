#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
import torch



# In[ ]:


experiment = "final_atac_model_2"
if not os.path.exists(experiment):
    os.makedirs(experiment)


# In[ ]:


import sys
sys.path.insert(0, "../gReLU/src")
import grelu.resources


# In[ ]:


genome = 'dm6'
seq_len =  2000 # Length of the input sequence
training = True # Set to True to train model or False to use saved model


# In[ ]:


import h5py
import anndata as ad
import pandas as pd
import numpy as np

# Load the signal matrix from the .h5 file.
h5_path = "combined_all_signals_after_filtering.h5"
with h5py.File(h5_path, "r") as f:
    signal_matrix = f["signal_matrix"][:]

# Load corresponding BED file with region coordinates
bed_path = "filtered_peaks.bed"
bed = pd.read_csv(bed_path, sep="\t", header=None, names=["chrom","start","end"])

trusted_chroms = ['chr2L', 'chr2R', 'chr3L', 'chr3R']
keep_mask = bed['chrom'].isin(trusted_chroms)
bed = bed[keep_mask].reset_index(drop=True)
signal_matrix = signal_matrix[keep_mask.values, :]

assert signal_matrix.shape[0] == len(bed)

# This block will remove any peaks that are too close to the chromosome ends
# to support the requested seq_len of 2000.
print("Pre-filtering peaks based on seq_len to prevent out-of-boundary errors...")

# 1. Get the official chromosome sizes for the dm6 genome 
genome_file = "data/dm6.fa.fai"
chromsizes = {}
with open(genome_file, "r") as f:
    for line in f:
        chrom, size, *_ = line.strip().split("\t")
        chromsizes[chrom] = int(size)
        
expected_len = seq_len


# 1. Define the required margin on each side of a peak's center
margin = seq_len // 2 # For seq_len=2000, this is 1000

# 2. Map the size of each chromosome to every peak in the BED file.
# This creates a new 'chrom_size' column for easy comparison.
bed['chrom_size'] = bed['chrom'].map(chromsizes)

# 3. Calculate the center for all peaks at once (this is fast and vectorized).
bed['center'] = bed['start'] + (bed['end'] - bed['start']) // 2

# 4. Create the final boolean mask with a single, clean operation.
# A peak is kept if its center is within the valid margins of its chromosome.
keep_mask_filtered = (bed['center'] >= margin) & (bed['center'] <= (bed['chrom_size'] - margin))

# 5. Apply the final mask to both your BED dataframe and your signal matrix
original_count = len(bed)
bed = bed[keep_mask_filtered].reset_index(drop=True)
signal_matrix = signal_matrix[keep_mask_filtered.values, :]

print(f"Removed {original_count - len(bed)} peaks that were too close to chromosome ends for seq_len={seq_len}.")
print(f"Kept {len(bed)} valid peaks.")

# Your assertion should still pass
assert signal_matrix.shape[0] == len(bed)

# In[ ]:


# Construct the AnnData object from the data

# gReLU's data loaders expect an AnnData object where:
# - ad.X contains the signal matrix (transposed to tasks x regions)
# - ad.var contains the region coordinates
# - ad.obs contains the names of the tasks/experiments

# Define the full list of task names corresponding to the original signal_matrix
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

# Create a pandas Series to easily find indices
all_tasks_series = pd.Series(all_task_names)

# Find the names and original indices of ATAC-seq tasks
is_atac_mask = all_tasks_series.str.contains("atac")
atac_task_names = all_tasks_series[is_atac_mask].tolist()
atac_indices = all_tasks_series.index[is_atac_mask].tolist()

print(f"Found and selected {len(atac_task_names)} ATAC-seq tasks.")

# Filter the original signal_matrix to keep only the ATAC-seq columns
signal_matrix_atac = signal_matrix[:, atac_indices]


# Construct the AnnData object from the filtered ATAC-seq data
print("Creating AnnData object...")

# The signal matrix needs to be in the shape (n_tasks, n_regions)
X = signal_matrix_atac.T # Use the filtered matrix

# The .var attribute holds the information about the genomic regions
var = bed

# The .obs attribute holds information about the tasks (experiments).
obs = pd.DataFrame({"task_name" : atac_task_names}) # Use the filtered names

#Create the AnnData object
ad = ad.AnnData(X, obs=obs, var=var)

# In[ ]:


#Set the index for .var and .obs to be compliant with gReLU
ad.var.index = ad.var["chrom"] + "_" + ad.var["start"].astype(str) + "_" + ad.var["end"].astype(str)
ad.obs.index = ad.obs["task_name"]

print("AnnData object created with shape (tasks, regions):", ad.shape)
print("Variable (regoin) info example:\n", ad.var.head())
print("\nObservations (task) info example: \n", ad.obs.head())


# In[ ]:


# Download the genome and split data
genome = "dm6"

# Split chr2R into validation and test sets
# Step 1: Randomly split chr2R into val/test
chr2R_mask = ad.var["chrom"] == "chr2R"
chr2R_indices = np.where(chr2R_mask)[0]
np.random.shuffle(chr2R_indices)
val_indices = chr2R_indices[:len(chr2R_indices)//2]
test_indices = chr2R_indices[len(chr2R_indices)//2:]

# Step 2: Temporarily rename chromosome for splitting
ad.var.iloc[val_indices, ad.var.columns.get_loc("chrom")] = "chr2R_val"
ad.var.iloc[test_indices, ad.var.columns.get_loc("chrom")] = "chr2R_test"

# Split the AnnData object using gReLU's utility function
ad_train, ad_val, ad_test = grelu.data.preprocess.split(
    ad,
    train_chroms=['chr2L', 'chr3L', 'chr3R'],
    val_chroms=['chr2R_val'],
    test_chroms=['chr2R_test']
)

# Restore real chromosome names
for d in [ad_train, ad_val, ad_test]:
    d.var["chrom"] = d.var["chrom"].replace({
        "chr2R_val": "chr2R",
        "chr2R_test": "chr2R"
    })

# In[ ]:


# Create labeled sequence datasets
# This gReLU class will automatically handle fetching DNA sequences for the
# coordinates in ad.var and pairing them with the target signals in ad.X.

print("\nCreating PyTorch datasets...")

import grelu.data.dataset

# Data augmentation is applied to improve model robustness

train_dataset = grelu.data.dataset.AnnDataSeqDataset(
    ad_train.copy(),
    genome=genome,
    seq_len=seq_len,
    rc=True, # Augmentation: randomly reverse-complement the sequence
    max_seq_shift=0, # Augmentation: randomly shift the sequence by up to max_seq_shift bp, however, intervals were already padded in my data.
)

# No augmentation for validation and test sets
val_dataset = grelu.data.dataset.AnnDataSeqDataset(ad_val.copy(), genome=genome, seq_len=seq_len)
test_dataset = grelu.data.dataset.AnnDataSeqDataset(ad_test.copy(), genome=genome, seq_len=seq_len)

print("Datasets created successfully.")


# In[ ]:


import glob

print("\nDefining and training the model...")
import grelu.lightning

# These parameters define the model architecture. We'll use ConvMLP Model.
model_params = {
    'model_type': 'ConvMLPModel',
    'n_tasks': ad.shape[0],
    'seq_len': 2000,

    # --- Body Parameters ---
    'n_conv': 5,
    'stem_channels': 256,
    'channel_init': 192,
    'channel_mult': 1.1,
    'kernel_size': 9,
    'pool_func': "avg",
    'pool_size': 4,
    'conv_norm': True,
    'act_func': "gelu",

    # --- Head Parameters ---
    'mlp_hidden_size': [1024, 512],
    'mlp_norm': True,
    'dropout': 0.1,
}

# --- CHANGE START ---
# These parameters configure the training process
train_params = {
    'task': 'regression',
    'loss': 'mse', # Mean Squared Error is good for regression
    'lr': 1e-3, # Learning rate
    'batch_size': 128, # Adjust based on GPU memory
    
    # We set a high max_epochs and let early stopping decide when to stop training.
    'max_epochs': 100,
    
    'logger': 'csv', # Log results to a CSV file
    'save_dir': experiment, # Directory to save model and logs
    
    # Add parameters for Early Stopping.
    # This will monitor the validation loss. If it doesn't improve for
    # 'patience' number of epochs, training will stop. The model from the
    # best epoch (lowest val_loss) will be saved automatically because
    # 'checkpoint' is True.
    'early_stopping_monitor': 'val_loss', # The metric to watch.
    'early_stopping_patience': 5,        # Number of epochs to wait for improvement.
    'early_stopping_mode': 'min',        # We want to MINIMIZE the validation loss.

    'checkpoint': True, # Save the best model checkpoint (based on val_loss by default).
    'num_workers': 4, # Number of CPU cores for data loading
    'devices': [0], # Use the first GPU. Set to 'cpu' if no GPU.
}
# --- CHANGE END ---


# Initialize the LightningModel wrapper
model = grelu.lightning.LightningModel(model_params, train_params)
print("\nModel architecture:\n", model)


# Start training if the 'training' flag is True
if training:
    print("\nStarting training...")
    trainer = model.train_on_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    print("Training complete!")

    # Get the path to the best model from the trainer object
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
else:
    print("Training skipped. Finding last saved model checkpoint...")
    # Search for the .ckpt file within the experiment directory
    checkpoint_files = glob.glob(f"{experiment}/**/*.ckpt", recursive=True)
    if not checkpoint_files:
        raise FileNotFoundError(f"Error: No checkpoint file (.ckpt) found in '{experiment}'. Please train a model first by setting training = True.")
    # Use the first checkpoint found
    best_checkpoint_path = checkpoint_files[0]

# --- This part runs in both cases ---
print(f"\nLoading model from: {best_checkpoint_path}")
model = grelu.lightning.LightningModel.load_from_checkpoint(best_checkpoint_path)


# In[ ]:


# Evaluate the model's performance on the test set

# The test set contains data the model has never seen before, giving us an
# unbiased measure of its performance.

print("Evaluating model performance on the test set...")

test_metrics = model.test_on_dataset(
    dataset=test_dataset,
    devices=[0], # Use the same GPU as training
    num_workers=4, # Use the same number of workers
    batch_size=128, # Use the same batch size
)

print("\n--- Evaluation Complete ---")
print("Test set performance metrics:")
print(test_metrics)


# In[ ]: