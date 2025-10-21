# Generates scatter plots for ATAC-seq data
# Compares signals of orer to each mutant (zldrnai, cic6, gd7) for each time point (1to15, 15to2, 2to25, 25to3)

import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import seaborn as sns 

# Define timepoints and mutant names
timepoints = ["1to15", "15to2", "2to25", "25to3"]
mutants = ["zldrnai", "gd7", "cic6"]

# Function to get experiment names
def get_exp_name(mutant, timepoint):
    return f"{mutant}_{timepoint}_atac"

# Quantile Normalization Function
def quantile_normalize(matrix):
    x = matrix.copy()
    ranked_mean = np.mean(np.sort(x, axis=0), axis=1)
    ranks = np.argsort(np.argsort(x, axis=0), axis=0)
    return ranked_mean[ranks]

# File path
h5_path = "combined_all_signals_after_filtering.h5"
with h5py.File(h5_path, "r") as f:
    signal_matrix = f["signal_matrix"][:]
    experiment_types = [s.decode("utf-8") for s in f["experiment_types"][:]]

# Apply Quantile Normalization
normalized_signal_matrix = quantile_normalize(signal_matrix)

# Create a mapping: experiment name -> column index
name_to_index = {name: i for i, name in enumerate(experiment_types)}

# Create Output Directories
output_dir = "region_distribution_figures"
scatter_dir = os.path.join(output_dir, "scatterplots_atac")
os.makedirs(scatter_dir, exist_ok=True)
scatter_dir_qnorm = os.path.join(output_dir, "scatterplots_atac_qnorm")
os.makedirs(scatter_dir_qnorm, exist_ok=True)
control_dir = os.path.join(output_dir, "normalization_effect_plots")
os.makedirs(control_dir, exist_ok=True)

# Downsampling
n_points = signal_matrix.shape[0]
sample_size = 5000
use_sampling = n_points > sample_size
print(f"Total data points: {n_points}. Downsampling to {sample_size} for plotting.")



# Generate comparison scatter plots
print("Generating comparison plots...")
for tp in timepoints:
    orer_name = get_exp_name("orer", tp)

    if orer_name not in name_to_index:
        continue

    x_idx = name_to_index[orer_name]
    x_signal = np.log1p(signal_matrix[:, x_idx])
    x_signal_qnorm = np.log1p(normalized_signal_matrix[:, x_idx])

    for mutant in mutants:
        mutant_name = get_exp_name(mutant, tp)

        if mutant_name not in name_to_index:
            continue

        y_idx = name_to_index[mutant_name]
        y_signal = np.log1p(signal_matrix[:, y_idx])
        y_signal_qnorm = np.log1p(normalized_signal_matrix[:, y_idx])

        # Select a random subset of data for plotting
        if use_sampling:
            random_indices = np.random.choice(n_points, size=sample_size, replace=False)
            x_plot, y_plot = x_signal[random_indices], y_signal[random_indices]
            x_plot_qnorm, y_plot_qnorm = x_signal_qnorm[random_indices], y_signal_qnorm[random_indices]
        else:
            x_plot, y_plot = x_signal, y_signal
            x_plot_qnorm, y_plot_qnorm = x_signal_qnorm, y_signal_qnorm

        # plot 1: Original Data 
        plt.figure(figsize=(6, 6))
        plt.scatter(x_plot, y_plot, alpha=0.3, s=10) # Increased alpha/size for sampled data
        sns.kdeplot(x=x_plot, y=y_plot, levels=5, colors='k', linewidths=0.7)
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='y=x') 
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel(f"log(1 + {orer_name} signal)"); plt.ylabel(f"log(1 + {mutant_name} signal)")
        plt.title(f"ATAC: {orer_name} vs {mutant_name}"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(scatter_dir, f"scatter_{orer_name}_vs_{mutant_name}.png"), dpi=300)
        plt.close()

        # plot 2: Quantile Normalized Data 
        plt.figure(figsize=(6, 6))
        plt.scatter(x_plot_qnorm, y_plot_qnorm, alpha=0.3, s=10)
        sns.kdeplot(x=x_plot_qnorm, y=y_plot_qnorm, levels=5, colors='k', linewidths=0.7)
        lims_qnorm = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims_qnorm, lims_qnorm, 'r--', alpha=0.75, zorder=5, label='y=x')
        plt.xlim(lims_qnorm); plt.ylim(lims_qnorm)
        plt.xlabel(f"log(1 + {orer_name} signal)"); plt.ylabel(f"log(1 + {mutant_name} signal)")
        plt.title(f"ATAC: {orer_name} vs {mutant_name} (Quantile Normalized)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(scatter_dir_qnorm, f"scatter_qnorm_{orer_name}_vs_{mutant_name}.png"), dpi=300)
        plt.close()
print("Comparison plots saved.")


# Generate "Before vs. After" control plots
print("\nGenerating before-vs-after normalization control plots...")
for i, exp_name in enumerate(experiment_types):
    before_signal = np.log1p(signal_matrix[:, i])
    after_signal = np.log1p(normalized_signal_matrix[:, i])
    
    # Select a random subset for the control plot
    if use_sampling:
        random_indices = np.random.choice(n_points, size=sample_size, replace=False)
        before_plot, after_plot = before_signal[random_indices], after_signal[random_indices]
    else:
        before_plot, after_plot = before_signal, after_signal

    plt.figure(figsize=(6, 6))
    plt.scatter(before_plot, after_plot, alpha=0.3, s=10)
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=5, label='y=x (No Change)')
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Original Signal (log1p)"); plt.ylabel("Quantile Normalized Signal (log1p)")
    plt.title(f"Effect of Normalization on {exp_name}"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(control_dir, f"norm_effect_{exp_name}.png"), dpi=300)
    plt.close()
print("Control plots saved.")