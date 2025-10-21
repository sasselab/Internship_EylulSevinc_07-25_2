#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Combine signals in bigWig files within regions defined in a .bed file

import pyBigWig
import pandas as pd
import h5py
import os
import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt


# In[ ]:


#Set-up path
data_dir = "data"

bed_path= "filtered_peaks.bed"
output_h5 = "combined_all_signals_after_filtering.h5"

#Load BED regions
regions = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"])  
regions["chr"] = regions["chr"].apply(lambda x: "chr" + x if not x.startswith("chr") else x)

# Classify bw files by experiment type
bw_files = [f for f in os.listdir(data_dir) if f.endswith(".bw")]
experiment_categories = defaultdict(list)

expected_categories = [
    "orer_1to15_atac", "gd7_1to15_atac", "zldrnai_1to15_atac", "cic6_1to15_atac",
    "orer_15to2_atac", "gd7_15to2_atac", "zldrnai_15to2_atac", "cic6_15to2_atac",
    "orer_2to25_atac", "gd7_2to25_atac", "zldrnai_2to25_atac", "cic6_2to25_atac",
    "orer_25to3_atac", "gd7_25to3_atac", "zldrnai_25to3_atac", "cic6_25to3_atac",
    "orer_bcd_mbt_nexus", "orer_cad_mbt_nexus", "orer_dl_mbt_nexus", "orer_gaf_mbt_nexus",
    "orer_twi_mbt_nexus", "orer_zld_mbt_nexus", "orer_h3k4me1_mbt_seq", "orer_h3k27ac_mbt_seq",
    "orer_wce_mbt_seq", "orer_mbt_100u_mnase"
   
]

for cat in expected_categories:
    experiment_categories[cat] = []


# In[ ]:


#Assign bw files to experiment categories

for f in bw_files:
    if "_orer_1to15_atac_" in f:
        experiment_categories["orer_1to15_atac"].append(f)
    elif "_orer_15to2_atac_" in f:
        experiment_categories["orer_15to2_atac"].append(f)
    elif "_orer_2to25_atac_" in f:
        experiment_categories["orer_2to25_atac"].append(f)
    elif "_orer_25to3_atac_" in f:
        experiment_categories["orer_25to3_atac"].append(f)
    elif "gd7_1to15_atac" in f:
        experiment_categories["gd7_1to15_atac"].append(f)
    elif "gd7_15to2_atac" in f:
        experiment_categories["gd7_15to2_atac"].append(f)
    elif "gd7_2to25_atac" in f:
        experiment_categories["gd7_2to25_atac"].append(f)
    elif "gd7_25to3_atac" in f:
        experiment_categories["gd7_25to3_atac"].append(f)
    elif "zldrnai_1to15_atac" in f:
        experiment_categories["zldrnai_1to15_atac"].append(f)
    elif "zldrnai_15to2_atac" in f:
        experiment_categories["zldrnai_15to2_atac"].append(f)
    elif "zldrnai_2to25_atac" in f:
        experiment_categories["zldrnai_2to25_atac"].append(f)
    elif "zldrnai_25to3_atac" in f:
        experiment_categories["zldrnai_25to3_atac"].append(f)
    elif "cic6_1to15_atac" in f:
        experiment_categories["cic6_1to15_atac"].append(f)
    elif "cic6_15to2_atac" in f:
        experiment_categories["cic6_15to2_atac"].append(f)
    elif "cic6_2to25_atac" in f:
        experiment_categories["cic6_2to25_atac"].append(f)
    elif "cic6_25to3_atac" in f:
        experiment_categories["cic6_25to3_atac"].append(f)
    elif "orer_bcd_mbt_nexus" in f:
        experiment_categories["orer_bcd_mbt_nexus"].append(f)
    elif "orer_cad_mbt_nexus" in f:
        experiment_categories["orer_cad_mbt_nexus"].append(f)
    elif "orer_dl_mbt_nexus" in f:  
        experiment_categories["orer_dl_mbt_nexus"].append(f)
    elif "orer_gaf_mbt_nexus" in f:
        experiment_categories["orer_gaf_mbt_nexus"].append(f)
    elif "orer_twi_mbt_nexus" in f:
        experiment_categories["orer_twi_mbt_nexus"].append(f)
    elif "orer_zld_mbt_nexus" in f:
        experiment_categories["orer_zld_mbt_nexus"].append(f)
    elif "orer_h3k4me1_mbt_seq" in f: 
        experiment_categories["orer_h3k4me1_mbt_seq"].append(f)
    elif "orer_h3k27ac_mbt_seq" in f:
        experiment_categories["orer_h3k27ac_mbt_seq"].append(f)
    elif "orer_wce_mbt_seq" in f:
        experiment_categories["orer_wce_mbt_seq"].append(f)
    elif "orer_mbt_100u_mnase" in f:
        experiment_categories["orer_mbt_100u_mnase"].append(f)
    else:
        experiment_categories["Other"].append(f)



# In[10]:


#Helper function for unstranded signals

def get_signal_unstranded(path, regions_df):
    with pyBigWig.open(path) as bw:
        chrom_sizes = bw.chroms() #gives you a dictionary of all chromosomes in the file and their lengths.
                                #{'chr2L': 23513712, 'chr3R': 27905053, ...}

        signal = []
        for _, row in regions_df.iterrows():
            ch, start, end = row["chr"], int(row["start"]), int(row["end"])
            if ch not in chrom_sizes or start >= end or start < 0 or end > chrom_sizes[ch]:
                signal.append(0.0)
                continue
            values = np.nan_to_num(bw.values(ch, start, end, numpy=True)) #If NaN is returned, it is replaced with 0.0
            signal.append(values.mean()) #avg signal values in the region
        return np.array(signal) #total signal summed over that region returned as a NumPy array.
            

#Helper function for stranded signals (ChIP-nexus)
def get_signal_stranded(path_pos, path_neg, regions_df):
    with pyBigWig.open(path_pos) as bw_pos, pyBigWig.open(path_neg) as bw_neg:
        chrom_sizes = bw_pos.chroms()
        signal = []
        for _, row in regions_df.iterrows():
            ch, start, end = row["chr"], int(row["start"]), int(row["end"])
            if ch not in chrom_sizes or start >= end or start < 0 or end > chrom_sizes[ch]:
                signal.append(0.0)
                continue
            pos_vals = np.nan_to_num(bw_pos.values(ch, start, end, numpy=True)) #Numpy array
            neg_vals = np.abs(np.nan_to_num(bw_neg.values(ch, start, end, numpy=True)))  # Use absolute values for negative strand
            signal.append((pos_vals + neg_vals).mean())
        return np.array(signal)

experiment_signals = {}


# In[12]:


all_signals = {}


# In[ ]:


# Print warnings for any expected category that ended up empty
for cat in expected_categories:
    if len(experiment_categories[cat]) == 0:
        print(f"WARNING: No files matched for experiment category: {cat}")

for exp_type, files in experiment_categories.items():
    print(f"Processing {exp_type} ({len(files)} files)")
    n_windows = len(regions)
    if "nexus" in exp_type:
        # Group files into positive/negative pairs
        bw_grouped = defaultdict(dict)
        for fname in files:
            full_path = os.path.join(data_dir, fname)
            if "_positive" in fname:
                key = fname.replace("_positive.bw", "")
                bw_grouped[key]["pos"] = full_path
            elif "_negative" in fname:
                key = fname.replace("_negative.bw", "")
                bw_grouped[key]["neg"] = full_path
        signals_list = []
        for key, paths in bw_grouped.items():
            if "pos" in paths and "neg" in paths:
                signal = get_signal_stranded(paths["pos"], paths["neg"], regions)
                signals_list.append(signal)
            else:
                print(f"Skipping incomplete ChIP-nexus pair: {key}")
        if signals_list:
            combined_signal = np.mean(signals_list, axis=0)
        else:
            combined_signal = np.zeros(n_windows)
    else:
        signals_list = []
        for fname in files:
            full_path = os.path.join(data_dir, fname)
            signal = get_signal_unstranded(full_path, regions)
            signals_list.append(signal)
        if signals_list:
            combined_signal = np.mean(signals_list, axis=0)
        else:
            combined_signal = np.zeros(n_windows)
    experiment_signals[exp_type] = combined_signal
print("done")

import pickle

with open("experiment_signals.pkl", "wb") as f:
    pickle.dump(experiment_signals, f)



# In[ ]:


# Convert to 2D matrix: [n_regions x n_experiment_types]
exp_types = list(experiment_signals.keys())
matrix_2d = np.stack([experiment_signals[exp] for exp in exp_types], axis=1)

print(matrix_2d[:5])  # Print first 5 rows for verification

# Save all as .h5
print("Saving 2D signal matrix...")
with h5py.File(output_h5, "w") as hf:
    hf.create_dataset("signal_matrix", data=matrix_2d)
    hf.create_dataset("experiment_types", data=np.array(exp_types, dtype="S"))
    hf.create_dataset("region_chr", data=np.array(regions["chr"], dtype="S"))
    hf.create_dataset("region_start", data=regions["start"].values)
    hf.create_dataset("region_end", data=regions["end"].values)


# In[ ]:


"""
import pickle
import h5py

with open("experiment_signals.pkl", "rb") as f:
    experiment_signals = pickle.load(f)

# Signal across all experiments


print("Combining signals across all experiments")
with h5py.File("total_combined_signal_all_experiments.h5", "w") as hf:
    for exp_type, signal_array in experiment_signals.items():
        hf.create_dataset(exp_type, data=signal_array)
    total = sum(experiment_signals.values())
    hf.create_dataset("total_combined_signal_all_experiments", data=total)
    """


# In[ ]:


with h5py.File(output_h5, "r") as f:
    print("Datasets in signal_matrix.h5:")
    for key in f.keys():
        print(f"- {key}, shape: {f[key].shape}, dtype: {f[key].dtype}")

       


# In[ ]:


import h5py
import numpy as np

with h5py.File(output_h5, "r") as f:
    signal_matrix = f["signal_matrix"]
    
    # Check shape and type
    print("Shape:", signal_matrix.shape)  # Expect (num_regions, num_experiments)
    print("Dtype:", signal_matrix.dtype)  # Typically float32

    # Peek at first 5 rows (each row = a genomic region)
    print("\nFirst 5 rows (all columns):")
    print(signal_matrix[:5])

    # Peek at first 5 columns of first 5 regions
    print("\nSubset [0:5, 0:5] (rows and columns):")
    print(signal_matrix[:5, :5])

    # Peek at a specific region (e.g., region 10)
    print("\nRegion 10 signal across all experiments:")
    print(signal_matrix[10])  # shape = (25,)

    # Peek at a specific experiment (e.g., experiment 3)
    print("\nExperiment 3 signal across first 10 regions:")
    print(signal_matrix[:10, 2])  # column 2 = 3rd experiment


# In[ ]:


#For combined_all_signals.h5

#This creates a bedGraph file of signals on regions of a given bed file.
# bedGraph file can be visualized in IGV

# Load the HDF5 file
import h5py
import numpy as np
import os

# Load data
with h5py.File(output_h5, "r") as f:
    chr_list = f["region_chr"][:].astype(str)
    start_list = f["region_start"][:]
    end_list = f["region_end"][:]
    signal_matrix = f["signal_matrix"][:]
    experiment_types = f["experiment_types"][:].astype(str)

# Output folder
output_folder = "bedgraphs_after_filtering"
os.makedirs(output_folder, exist_ok=True)

# Write one bedGraph file per experiment
for i, exp_name in enumerate(experiment_types):
    out_path = os.path.join(output_folder, f"{exp_name}.bedGraph")
    with open(out_path, "w") as f_out:
        for chrom, start, end, signal in zip(chr_list, start_list, end_list, signal_matrix[:, i]):
            f_out.write(f"{chrom}\t{start}\t{end}\t{signal:.4f}\n")

print("bedGraph files saved in folder:", output_folder)

