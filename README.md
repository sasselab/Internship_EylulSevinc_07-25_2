# Internship_EylulSevinc_07-25
# Project Title

## Overview
This repository contains documentation and code for my internship research titled "Understanding Cis-Regulatory code of chromatin accessibilty in the early *Drosophila melanogaster* development using Deep Learning"

## Contents
This repository has the following key folders:
    - raw_data
    - data
    - directory_models
    - outputs
        -
        -


## Introduction
During early *Drosophila melanogaster* embryogenesis, large-scale chromatin remodelling establishes the first gene-regulatory programs that pattern the embry along its anterior-posterior and dorsoventral axes. Chromatin accessibility, the extent which DNA is open for transcription factor (TF) binding, is a critical aspect of this regulatory code. Understanding how specific DNA motifs enable certain TFs to bind and regulate these regions is key to decoding the cis-regulatory logic of development.

Chromatin accessibility determines which enhancers are active: open regions permit TF binding, while closed regions remain repressed. During zygotic genome activation, the pioneer TF Zelda establishes a baseline of open chromatin, and patterning TFs such as Dorsal, Twist, Bicoid, and Caudal refine enhancer activity and regulate Hox genes that define embryonic segments (Brennan et al., 2023).

Recent studies have shown that DNA sequence features alone can predict chromatin accessibility and TF binding when modeled with deep learning. Methods such as BPNet and ChromBPNet learn how combinations and spatial arrangements of sequence motifs encode regulatory activity (Kopp et al., 2020; Avsec et al., 2021). Attribution approaches including DeepLIFT and Integrated Gradients then identify which motifs most strongly influence model predictions, revealing the underlying cis-regulatory code (Shrikumar et al., 2017; Gupta et al., 2022).

Applications of theese models to early *Drosophila* embryos have uncovered canonical motifs for TFs such as Zelda, Dorsal, Twist, Bicoid, and Caudal, as well as motif co-occurence patterns that predict enhancer activity (Natarajan et al. 2023; Combs et al., 2023).

Building on this framework, this project aimed to identify sequence motfis that drive chromatin accessibilty and TF binding in early *Drosophila* emryos. Using publicly available ATAC-seq and ChIP-nexus data from multiple developmental stages and TF-mutant backgrounds (Brennan et al., 2023; Natarajan et al., 2023), a convolutional multilayer perceptron (Conv-MLP) model implemented in gReLU was trained on 200-bp genomic windows to predict accessibility and binding signals. Comparing predictions between wild-type and mutant conditions at different time-points enabled the detection of regions and motifs driving changes in chromatin accessibility during early *Drosophila* embryonic development. 

## Methodology
### Data Preprocessing
The datasets used in this project were obtained from Brennan et al. (2023), which investigated transcription factor binding and chromatin accessibility during early Drosophila melanogaster embryogenesis. All raw and processed data were downloaded from the Gene Expression Omnibus (GEO) under the accession number GSE218852, aligned to the Drosophila reference genome dm6. The dataset includes ATAC-seq, ChIP-nexus, MNase-seq, and histone modification (H3K4me1, WCE) signal tracks in bigWig (bw) format, along with corresponding narrowPeak files identifying regions of significant signal enrichment.
For the ChIP-nexus experiments, signal tracks were provided separately for the positive and negative DNA strands, covering transcription factors Bicoid (Bcd), Caudal (Cad), Dorsal (Dl), GAGA factor (GAF), Twist (Twi), and Zelda (Zld) in the orer (wild-type) background at the mid-blastula transition (MBT) time point. The ATAC-seq datasets covered four developmental stages (1–1.5 h, 1.5–2 h, 2–2.5 h, and 2.5–3 h after egg laying) and multiple genetic backgrounds (orer, gd7, zldRNAi, cic6).
To construct the training data, a union set of peaks was generated from all available narrowPeak files across both ATAC-seq and ChIP datasets. Each peak region was then extended by adding overhangs on both ends and subdivided into non-overlapping 200 bp windows, ensuring that all windows remained within chromosome boundaries. These 200 bp regions served as the core genomic units for model input.
Before training, regions overlapping the ENCODE Drosophila blacklist were filtered out to remove genomic artifacts and low-quality signal regions. The resulting processed dataset contained approximately 450,000 genomic windows across all chromosomes.
Two separate gReLU convolutional multilayer perceptron (Conv-MLP) models were trained — one using ATAC-seq signals to predict chromatin accessibility, and the other using ChIP-nexus signals to model transcription factor binding profiles.

### Model Implementation
### Model Interpretation with Integrated Gradients


## Results
### Model Performance
### Sequence-to-function identifies key regulatory motifs for *Drosophila melanogaster* development

## Discussion

## Quick Start
### 1. Clone the repository
### 2. Navigate to the project directory:
### 3. Create a virtual environment: 
### 4. Install required packages
### 5. Running the pipeline

## References
The ATAC model was trained using the [train_atac_model.py](scripts/model_training/train_atac_model.py) script located in the [model training](scripts/modeltraining/) folder under [scripts/](scripts/).
