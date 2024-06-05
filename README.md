# Bayesian Inference for Consistent Predictions in Overparameterized Nonlinear Regression

## Overview

This project outlines the methodology and prerequisites for experiments of Bayesian overparameterized nonlinear regression, discussed in the following paper:

**Tomoya Wakayama (2024). [Bayesian Inference for Consistent Predictions in Overparameterized Nonlinear Regression](https://arxiv.org/abs/2404.04498)**.

## Prerequisites 📋

### System Requirements 🖥

- **Julia Version**: 1.10
- **Python Version**: 3.11
- **R Version**: 4.3

### Required Packages 📦

To execute the project scripts, the following libraries and packages for Python, Julia, and R must be installed:

**Python Libraries:**

```python
pandas scikit-learn optuna lightgbm torch pyro
```

**Julia Packages:**

```julia
LinearAlgebra, Distributions, MultivariateStats, Statistics, StatsBase, Random, SparseArrays, Plots, CSV, DataFrames
```

**R Libraries:**

```r
data.table rstan pROC horseshoenlm pgdraw
```

### Julia-Python Dependency 🐍

The Julia code requires Python's `scikit-learn` library for certain functionalities. Install it via Conda within Julia using the following commands:
```julia
using PyCall, Conda
Conda.add("scikit-learn")
```

After installation, import the `roc_auc_score` function from `scikit-learn` for model evaluation:

```julia
roc_auc_score = pyimport("sklearn.metrics")["roc_auc_score"] # Import sklearn.metrics.roc_auc_score
```

## Dataset 📊

The experiments utilize the ARCENE dataset from the UCI Machine Learning Repository, available at [https://archive.ics.uci.edu/ml/datasets/arcene](https://archive.ics.uci.edu/ml/datasets/arcene).

## License 📄

The source code and documentation are licensed under the MIT License.