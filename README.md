# 🐉 Drugon

**A physics-inspired fingerprint that beats pretrained chemical transformers in low-data affinity prediction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Drugon is a lightweight molecular affinity predictor built on **Phase-Encoded Morgan Fingerprints** and **LightGBM**. It uses complex-valued atom embeddings where each substructure is assigned a phase derived from its chemical context (atomic number, aromaticity, formal charge).

Despite having **~15,000× fewer parameters** than pretrained chemical transformers (ChemBERTa, MoLFormer), Drugon outperforms them on target-specific pIC50 prediction in the low-data regime (n < 2000).

## Key Results

| Target | n | Binary Morgan | **Drugon** | ChemBERTa | MoLFormer |
|--------|---|---------------|-----------|-----------|-----------|
| c-Abl (CHEMBL4550) | 1378 | 0.723 | **0.749** | 0.605 | 0.551 |
| AChE (CHEMBL220) | 2000 | 0.834 | **0.844** | — | — |
| COX-2 (CHEMBL230) | 2000 | 0.741 | **0.742** | — | — |

*Reported as Pearson r under 5-fold cross-validation.*

## Model Comparison (c-Abl)

| Model | Params | Size | Training Time | Pearson r |
|-------|--------|------|---------------|-----------|
| **Drugon** | ~3K | **5 MB** | **30 s** | **0.749** |
| ChemBERTa | 46 M | 180 MB | 15 min | 0.605 |
| MoLFormer | 47 M | 200 MB | 20 min | 0.551 |

## How It Works

Drugon extends the standard binary Morgan fingerprint with **complex phase encoding**:

$$
\text{Drugon}[i] = \sum_{a \in \text{atoms}(i)} \exp\left(i \cdot \phi(a)\right)
$$

where the phase $\phi(a)$ is derived from the local chemical environment:

$$
\phi(a) = \pi \cdot \left(0.15 \cdot Z + 1.5 \cdot \mathbb{1}_{\text{aromatic}} + 1.2 \cdot q\right)
$$

with $Z$ = atomic number, $q$ = formal charge.

The real and imaginary components are concatenated into a **2048-dimensional** real vector, which is then fed to LightGBM.

## Installation

```bash
pip install rdkit lightgbm scikit-learn numpy pandas
