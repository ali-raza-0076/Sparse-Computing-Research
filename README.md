# Sparse Matrix Operations: Problem Statement and Achievements

[![Python 3.13+](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20WSL2-lightgrey.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **Authors**: Aymane Khaldi, Ali Raza | University of Houston

---

## Problem Statement

Sparse matrices, which are characterized by having a high proportion of zero elements (90-99.9% sparsity), are fundamental in fields such as graph analytics, recommendation systems, and scientific computing. However, performing operations like addition and multiplication on these matrices efficiently poses significant challenges due to:

1. The computational overhead of dense representations.
2. The need for optimized memory usage.
3. The trade-offs between CPU and GPU performance for sparse workloads.

This project explores and benchmarks various approaches to address these challenges by combining sparse storage formats (CSR/CSC and COO) with memory models (in-memory and database-driven).

---

## Motivation

The motivation for this project stems from the need to:

- Identify the most efficient methods for sparse matrix operations across different hardware configurations.
- Optimize performance for real-world applications such as graph neural networks (GNNs) and dynamic graph updates.
- Evaluate the trade-offs between computational speed, memory efficiency, and database I/O overhead.

---

## Key Achievements

| **Achievement**            | **Description**                                                                 |
|-----------------------------|---------------------------------------------------------------------------------|
| **Faster Computations**     | Achieved up to **870× speedup** for specific matrix operations on CPUs.         |
| **Efficient Memory Usage**  | Reduced memory usage by up to **500×** compared to dense matrix representations.|
| **Graph Processing Gains**  | Improved graph analysis tasks by up to **718×** on real-world datasets.         |
| **GPU Acceleration**        | Leveraged GPUs to achieve up to **57× faster processing** for large matrices.   |
| **Dynamic Updates**         | Enabled faster updates to graphs, improving performance by up to **7×**.       |
| **Database Integration**    | Demonstrated efficient database usage with minimal overhead for large datasets. |
| **Hardware Utilization**    | Fully utilized modern hardware like multi-core CPUs and GPUs for better results.|

---

## Getting Started

For detailed setup instructions, system requirements, and how to run the benchmarks, please refer to the [instructions.md](instructions.md) file. It includes:

- **Installation Instructions**: Step-by-step setup for Windows, Linux, and WSL2
- **System Requirements**: Minimum and recommended hardware specifications
- **Running Benchmarks**: How to execute individual or all benchmarks
- **Troubleshooting Guide**: Common issues and their solutions

### Quick Start

```powershell
# Install dependencies
python install_requirements.py

# Run all benchmarks
python run_all_benchmarks.py
```

For complete instructions, see [instructions.md](instructions.md).

---
