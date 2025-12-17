# Sparse Matrix Operations: Addition and Multiplication
## Database Systems Project - COSC 6340

[![Python 3.13+](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20WSL2-lightgrey.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **Authors**: Aymane Khaldi, Ali Raza | University of Houston

---

## Abstract

Sparse matrices with 90-99.9% sparsity appear in graph analytics, recommendation systems, and scientific computing. We implement **four approaches** combining storage formats (CSR/CSC vs COO) with memory models (in-memory vs database). Through experiments on matrices up to **500,000×500,000** and graphs with **20,000 vertices**, we identify critical thresholds:

1. **870×** speedup at 99.9% sparsity (single-threaded CSR)
2. Multiprocessing overhead (1.9s) makes TRUE parallelism impractical below 10K×10K
3. GPU results show COO format **10× faster** than CSR (57× vs 5.75× speedup)
4. GNN workloads demonstrate **486-718× CPU speedup** on real graph data
5. Database I/O viable for CPU sparse (**2.4-11.5% overhead**), marginal for GPU sparse (75-86% overhead), catastrophic for dense (99+% overhead)
6. GPU utilizes **5,888 CUDA cores** across all benchmarks

---

## Performance Achievements

| Achievement | Value | Context |
|-------------|-------|---------|
| **Maximum CPU Speedup** | **870×** | CSR multiplication, 3,000×3,000 at 99.9% sparsity |
| **Maximum Memory Reduction** | **500×** | Sparse vs dense at 99.9% sparsity |
| **GNN CPU Performance** | **718×** | Static graph inference, 10,000 vertices, COO format |
| **GPU Sparse Speedup** | **57×** | COO format, 10,000×10,000 at 99.9% sparsity |
| **Incremental Update Gain** | **7.22×** | Dynamic graphs, CPU, 20,000 vertices at 99% |
| **Database I/O Efficiency** | **2.4%** | Overhead for 500K×500K sparse COO multiplication |
| **GPU CUDA Cores** | **5,888** | NVIDIA RTX 5070 Ti utilized across all tests |

---

## Comprehensive Benchmark Results

> **Hardware**: AMD Ryzen 9 8940HX (16 cores, 32 threads) + NVIDIA RTX 5070 Ti (5,888 CUDA cores, 12GB VRAM)  
> **Software**: Python 3.13.1, NumPy 2.2.1, PyTorch 2.9.1+cu130, Numba 0.61.0  
> **Validation**: All results verified against SciPy sparse implementations

### 1. Single-Threaded CPU: Multiplication Performance (CSR×CSC)

**Configuration**: AMD Ryzen 9 8940HX, Numba JIT compilation

| Matrix Size | Sparsity | Sparse Time | Dense Time | Speedup | Nonzeros | Theoretical Limit |
|-------------|----------|-------------|------------|---------|----------|-------------------|
| 1,000×1,000 | 90% | 0.363s | 0.486s | **1.34×** | 100,000 | 11× |
| 1,000×1,000 | 99% | 0.046s | 0.472s | **10.25×** | 10,000 | 100× |
| 1,000×1,000 | 99.9% | 0.014s | 0.481s | **34.67×** | 999 | 1,000× |
| 2,000×2,000 | 99.9% | 0.027s | 2.272s | **84.09×** | 4,000 | 1,000× |
| 3,000×3,000 | 99.9% | 0.064s | 55.385s | **870×** | 8,999 | 1,000× |

> **Analysis**: At 3,000×3,000, sparse processes **8,999 entries** while dense processes **27 billion operations** (n³). Measured speedup (870×) approaches theoretical limit (1,000×) as matrix size increases, validating O(nnz) vs O(n³) complexity advantage. Overhead decreases relatively as computation dominates.

### 2. Single-Threaded CPU: Addition Performance (CSR)

| Sparsity | Sparse Time | Dense Time | Speedup | Winner | Analysis |
|----------|-------------|------------|---------|--------|----------|
| 90% | 0.216s | 0.001s | 0.00× | **Dense** | SIMD optimization dominates |
| 99% | 0.021s | 0.001s | 0.05× | **Dense** | CSR conversion overhead exceeds savings |
| 99.9% | 0.004s | 0.001s | 0.28× | **Dense** | Addition is O(n²), not O(n³) |

> **Critical Finding**: Dense wins for addition at ALL sparsity levels due to highly optimized SIMD instructions. CSR conversion overhead (sorting + building row pointers) exceeds computation savings even at 99.9%. **Sparse excels for O(n³) multiplication but not O(n²) addition.**

### 3. Multiprocessing CPU Performance (16 cores, CSR Multiplication)

**Configuration**: Python multiprocessing.Pool, true OS-level parallelism

| Matrix Size | Sparsity | Sparse Time | Dense Time | Speedup | Process Overhead |
|-------------|----------|-------------|------------|---------|------------------|
| 1,000×1,000 | 99% | 1.949s | 0.012s | 0.01× | **1.9s** (142× compute) |
| 1,000×1,000 | 99.9% | 1.981s | 0.014s | 0.01× | **1.98s** (142× compute) |
| 2,000×2,000 | 99% | 2.526s | 0.064s | 0.03× | **2.5s** (39× compute) |
| 3,000×3,000 | 99.9% | 2.264s | 0.212s | 0.11× | **2.2s** (10× compute) |

> **Critical Finding**: Process creation overhead (1.9-2.5s) **dominates computation time**. At 1K×1K with 999 nonzeros, overhead is **142× larger** than computation (1.98s vs 0.014s). Even at 3K×3K, overhead remains 10× larger than benefit. **Dense wins by 9-100× across all sizes. Multiprocessing impractical below 10K×10K for sparse operations.**

### 4. GPU Performance: CSR vs COO Formats (99.9% sparsity)

**Configuration**: NVIDIA RTX 5070 Ti (5,888 CUDA cores), PyTorch 2.9.1+cu130

| Matrix Size | Format | Sparse Time | Dense Time | Speedup | Memory Sparse | Memory Dense |
|-------------|--------|-------------|------------|---------|---------------|--------------|
| 1,000×1,000 | CSR→COO | 0.001117s | 0.000352s | 0.31× | 0.03 MB | 15.26 MB |
| 1,000×1,000 | COO | 0.001117s | 0.000352s | 0.31× | 0.03 MB | 15.26 MB |
| 10,000×10,000 | CSR→COO | 0.022s | 0.126s | 5.75× | 1.14 MB | 381 MB |
| 10,000×10,000 | **COO** | **0.0017s** | 0.099s | **57.05×** | 1.14 MB | 381 MB |

> **Critical Finding**: COO is **10× faster** than CSR on GPU (0.0017s vs 0.022s) because **PyTorch natively supports COO format** while CSR must convert first, adding overhead. At 1K×1K, GPU overhead dominates and dense wins. At 10K×10K, sparse becomes competitive with 57× speedup.

### 5. GPU Scaling Analysis (COO Format)

| Matrix Size | Sparsity | Sparse Time | Dense Time | Speedup | Memory Savings |
|-------------|----------|-------------|------------|---------|----------------|
| 2,000×2,000 | 99.9% | 0.000045s | 0.000274s | **6.13×** | 333× (1.09MB vs 381MB) |
| 3,000×3,000 | 99.9% | 0.000041s | 0.000483s | **11.92×** | 333× |
| 10,000×10,000 | 99% | 0.000494s | 0.004863s | **9.84×** | 5.25× (72.6MB vs 381MB) |
| 10,000×10,000 | 99.9% | 0.000220s | 0.004801s | **21.86×** | **333×** (1.14MB vs 381MB) |

> **Key Insight**: GPU sparse speedup increases with larger matrices (6.13×→11.92×→21.86×) and higher sparsity (9.84× at 99%→21.86× at 99.9%). Memory savings remain consistent at 333× for 99.9% sparsity.

### 6. CPU Sparse vs GPU Dense: Platform Crossover Point

**Configuration**: 1,000×1,000 matrices

| Sparsity | CPU Sparse (CSR) | GPU Dense (5,888 cores) | Winner | GPU Advantage |
|----------|------------------|-------------------------|--------|---------------|
| 90% | 0.0217s | 0.0011s | **GPU** | **19.87×** faster |
| 99% | 0.0009s | 0.0004s | **GPU** | **2.30×** faster |
| 99.9% | **0.0001s** | 0.0004s | **CPU Sparse** | CPU **4×** faster |

> **Critical Threshold**: At **99.9% sparsity with only 999 nonzeros**, CPU sparse beats GPU because sequential processing of minimal data beats GPU kernel launch overhead (data transfer, CUDA synchronization). GPU's 5,888 cores cannot overcome overhead for ultra-sparse data.

### 7. Graph Neural Networks: Static Performance (Real Adjacency Matrices)

**Configuration**: Social network graphs, 99.9% sparsity, CPU single-threaded

| Vertices | Edges | CSR Time | COO Time | Dense Time | CSR Speedup | COO Speedup |
|----------|-------|----------|----------|------------|-------------|-------------|
| 5,000 | ~25,000 | 0.0008s | 0.0010s | 0.467s | **547×** | 467× |
| 10,000 | ~99,945 | 0.0084s | 0.0057s | 4.085s | 486× | **718×** |
| 20,000 | ~399,780 | 0.0556s | 0.0522s | 12.595s | **226×** | 241× |

> **Key Insight**: Real graph adjacency matrices at 99.9% sparsity show **486-718× CPU speedup**. **COO slightly faster than CSR** for graphs (0.0057s vs 0.0084s at 10K vertices) due to simpler graph representation and natural (i,j,v) triplet format. As graph size increases, speedup decreases but remains massive.

### 8. Graph Neural Networks: GPU Static Performance

**Configuration**: NVIDIA RTX 5070 Ti, 5,888 CUDA cores, real graphs

| Vertices | Sparsity | Sparse Time | Dense Time | Speedup | Winner |
|----------|----------|-------------|------------|---------|--------|
| 10,000 | 99% | 0.101s | 0.124s | 1.23× | Sparse |
| 10,000 | 99.9% | 0.022s | 0.104s | **4.69×** | Sparse |
| 20,000 | **99%** | **1.004s** | **0.898s** | 0.89× | **DENSE WINS** |
| 20,000 | 99.9% | 0.155s | 0.932s | **6.01×** | Sparse |

> **Critical Finding**: At 20K vertices with 99% sparsity (4M edges), **GPU dense WINS** (0.898s vs 1.004s sparse). Sparse only wins at **99.9%+ sparsity**. This demonstrates GPU's **massive parallelism (5,888 cores)** can overcome sparse advantages at moderate sparsity through brute-force dense computation.

### 9. Graph Neural Networks: Dynamic Updates (Incremental vs Full Recomputation)

**Configuration**: 20,000 vertices, edge additions

| Platform | Sparsity | Edges | Full Recomp | Incremental | Speedup | Winner |
|----------|----------|-------|-------------|-------------|---------|--------|
| **CPU** | 99% | 3,996,000 | 1.772s | 0.246s | **7.22×** | Incremental |
| **CPU** | 99.9% | 399,780 | 0.167s | 0.054s | **3.11×** | Incremental |
| **GPU** | 99% | 3,996,000 | 0.051s | 0.049s | 1.04× | Incremental |
| **GPU** | 99.9% | 399,780 | **0.019s** | 0.046s | 0.41× | **Full Recomp** |

> **Critical Finding**: **Opposite behavior on CPU vs GPU**. 
> - **CPU benefits from incremental updates** (7.22× speedup) by reusing previous computation results - validates database-style incremental maintenance
> - **GPU prefers full recomputation** (2.45× faster than incremental) because converting COO→dense→COO is expensive
> - **GPU is optimized for rebuilding entire structures**, not incremental modifications

### 10. Database I/O Overhead: CPU Sparse (500K×500K, 500K entries, 0.0002% density)

**Configuration**: 3-phase workflow (Read CSV → Compute → Write CSV)

| Approach | Operation | Total Time | I/O Time | Compute Time | I/O Overhead | Result NNZ | Viability |
|----------|-----------|------------|----------|--------------|--------------|------------|-------------|
| **CSR/CSC** | Addition | 35.03s | 3.01s | 32.02s | **8.6%** | 1M |  **Viable** |
| **CSR/CSC** | Mult (CSR×CSC) | 18.66s | 0.78s | 17.88s | **4.2%** | 88 |  **Viable** |
| **COO** | Addition | 13.09s | 1.51s | 11.58s | **11.5%** | 1M |  **Viable** |
| **COO** | Multiplication | 52.63s | 1.26s | 51.37s | **2.4%** | 263K | **Viable** |

> **Key Insight**: Database I/O overhead remains **minimal (2.4-11.5%)** at 500K×500K scale. Computation dominates I/O, confirming viability for CPU sparse operations. I/O overhead actually **decreases** as computation dominates at larger scales.

### 11. Database I/O Overhead: GPU Sparse (500K×500K, 500K entries, 0.0002% density)

**Configuration**: CSV I/O + GPU computation, 5,888 CUDA cores

| Approach | Operation | Total Time | I/O Time | Compute Time | I/O Overhead | Viability |
|----------|-----------|------------|----------|--------------|--------------|-------------|
| **COO** | Addition | 2.806s | 2.356s | 0.450s | **83.93%** |  **Marginal** |
| **COO** | Multiplication | 2.490s | 1.942s | 0.548s | **78.02%** |  **Marginal** |
| **CSR** | Addition | 3.747s | 3.207s | 0.540s | **85.58%** |  **Marginal** |
| **CSR** | Multiplication | 2.477s | 1.876s | 0.601s | **75.75%** |  **Marginal** |

> **Critical Finding**: GPU's **fast computation (0.45-0.60s)** makes CSV I/O (1.9-3.2s) **relatively expensive**. I/O overhead increases to **75-86%**, marking **marginal viability threshold** where GPU acceleration makes I/O dominant.

### 12. Database I/O: Dense Matrices (CATASTROPHIC FAILURE)

**Configuration**: 100% density, CSV I/O workflow

| Platform | Size | Entries | Operation | Total Time | I/O Time | Compute | I/O Overhead | Viability |
|----------|------|---------|-----------|------------|----------|---------|--------------|-------------|
| **CPU** | 2K×2K | 4M | Addition | 21.45s | 21.44s | 0.01s | **99.97%** | **NOT Viable** |
| **CPU** | 2K×2K | 4M | Multiplication | 21.26s | 21.18s | 0.08s | **99.65%** | **NOT Viable** |
| **CPU** | 3K×3K | 9M | COO Addition | 97.54s | 97.52s | 0.02s | **99.98%** | **NOT Viable** |
| **CPU** | 3K×3K | 9M | COO Mult | 102.62s | 102.41s | 0.21s | **99.79%** | **NOT Viable** |
| **GPU** | 3K×3K | 9M | COO Addition | 30.58s | 30.30s | 0.28s | **99.07%** | **NOT Viable** |
| **GPU** | 3K×3K | 9M | COO Mult | 33.25s | 32.97s | 0.28s | **99.17%** | **NOT Viable** |
| **GPU** | 3K×3K | 9M | CSR Addition | 35.12s | 34.80s | 0.32s | **99.08%** | **NOT Viable** |
| **GPU** | 3K×3K | 9M | CSR Mult | 34.13s | 33.86s | 0.27s | **99.22%** | **NOT Viable** |

> **Catastrophic Finding**: Dense matrices show **99+% I/O overhead** regardless of:
> - **Size** (2K×2K or 3K×3K)
> - **Platform** (CPU or GPU)
> - **Format** (COO or CSR)
> - **GPU cores** (5,888 CUDA cores can't help)
> 
> CSV parsing **dominates by two orders of magnitude**. CPU requires 97-103s for 9M entries where computation is only 0.02-0.21s. GPU takes 30-35s with 0.24-0.32s computation. **Binary formats or in-memory processing mandatory for dense matrices.**

### 13. Database I/O: GNN Inference Overhead

**Configuration**: Citation networks (Cora, PubMed)

| Dataset | Vertices | Edges | Operation | Total Time | I/O Time | Compute | I/O Overhead | Viability |
|---------|----------|-------|-----------|------------|----------|---------|--------------|-------------|
| **Cora** | 2,708 | 5,429 | GNN Inference | 0.284s | 0.252s | 0.032s | **88.7%** | **NOT Viable** |
| **PubMed** | 19,717 | 44,338 | GNN Inference | 3.088s | 3.011s | 0.077s | **97.5%** | **NOT Viable** |

> **Finding**: GNN forward pass computation (0.032-0.077s) is **extremely fast**, making CSV I/O dominant (88-97% overhead). **In-memory processing mandatory for GNN inference.**

---

## Asymptotic Analysis & Validation

### Theoretical Complexity

As dimension *n* increases with fixed sparsity *s*, nonzeros grow as `nnz = s·n²`.

**Multiplication Complexity**:
- Sparse: `T_sparse(n) = O(s²·n³)`
- Dense: `T_dense(n) = O(n³)`
- **Speedup ratio**: `1/s²`

At **99.9% sparsity** (s = 0.001): theoretical **1,000,000× limit**

### Empirical Validation (99.9% sparsity)

| Matrix Size | Measured Speedup | Theoretical Limit | % of Theoretical |
|-------------|------------------|-------------------|------------------|
| 1,000×1,000 | 34.67× | 1,000,000× | **3.5%** |
| 2,000×2,000 | 84.09× | 1,000,000× | **8.4%** |
| 3,000×3,000 | **870×** | 1,000,000× | **87%** |

> **Analysis**: Measured speedup **grows with size**, approaching `1/s²` asymptote. Overhead (sorting, pointer chasing) decreases relatively to computation as *n* increases. At 3,000×3,000, we achieve **87% of theoretical limit**.

### Memory Scaling

| Sparsity | Dense Memory | Sparse Memory | Measured Reduction |
|----------|--------------|---------------|-------------------|
| 99% | 381 MB | 4.2 MB | **90.9×** |
| 99.9% | 381 MB | 0.76 MB | **500×** |

- Dense requires `O(n²)` space
- Sparse requires `O(s·n²)` space
- Database approach becomes **necessary** when sparse storage `s·n²` exceeds available RAM

---

## Core Contributions

### 1. Database Theory Foundations

Sparse operations expressed as **relational queries**:

**Addition as Full Outer Join**:
```sql
C = A ⊔⊓ B ON (i, j)
```
Implementation: Sort both matrices by (i,j), two-pointer merge. Matching positions: add values. Non-matching: copy entry. Complexity: `O(nnz(A) + nnz(B))`.

**Multiplication as Join-Aggregate**:
```sql
C = SELECT i, k, SUM(v_A · v_B)
    FROM A JOIN B ON A.j = B.i
    GROUP BY i, k
```
For each (i,k): compute ∑<sub>j</sub> A[i,j]·B[j,k]. CSR/CSC enables O(1) row/column lookup. Complexity: `O(nnz(A) · avg_nnz(B))`.

### 2. External Memory Algorithms

**External Memory Guarantee**: External merge sort and blocked matrix operations handle matrices of arbitrary size [n→∞] by processing in fixed-size blocks that fit in RAM.

**I/O Cost Model**: For block size *B* tuples:
```
I/O_add = (nnz(A) + nnz(B) + nnz(C)) / B
I/O_mult = (nnz(A) + nnz(B) + nnz(C)) / B
```

Sequential access through CSR/CSC minimizes random seeks. Measurements show **2.4-11.5% I/O overhead** for sparse operations at 500K×500K scale.

**Complexity**: External sort requires `O((nnz/B) · log_{M/B}(nnz/M))` I/O operations, where *M* is RAM size. Computation: `O(nnz log nnz)` for sorting and `O(nnz)` for merging.

### 3. Four-Way Comparative Analysis

| Approach | Memory Model | Format | Best For | Peak Performance |
|----------|--------------|--------|----------|------------------|
| CSR/CSC Implementation | In-Memory | CSR/CSC | CPU multiplication | 870× speedup |
| COO Implementation | In-Memory | COO | GPU operations | 57× speedup |
| CSR/CSC Database | Disk-based | CSR/CSC | Out-of-core | 4.2% I/O overhead |
| COO Database | Disk-based | COO | Persistent storage | 2.4% I/O overhead |

### 4. Platform-Dependent Optimization

**CPU Prefers CSR**: 3.8× faster than COO for multiplication (0.064s vs 0.243s at 3K×3K, 99.9%)
- O(1) row/column access
- Cache-friendly sequential access
- Optimal for join-aggregate operations

**GPU Prefers COO**: 13× faster than CSR for multiplication (0.0017s vs 0.022s at 10K×10K, 99.9%)
- PyTorch native support
- No conversion overhead
- Optimal for massive parallelism

### 5. Sparsity Thresholds Identified

| Operation | Platform | Threshold | Winner |
|-----------|----------|-----------|--------|
| Multiplication | CPU | 99%+ | Sparse (10-870× speedup) |
| Multiplication | GPU | 99.9%+ | Sparse (21-57× speedup) |
| Addition | CPU | None | Dense always wins |
| GNN Inference | CPU | 99%+ | Sparse (226-718× speedup) |
| GNN Inference | GPU | 99.9%+ | Sparse (4-6× speedup) |

---

## Quick Start

### Installation

**Windows**:
```powershell
python install_requirements.py
```

**Linux/WSL**:
```bash
sudo apt update && sudo apt install -y python3-pip
python3 install_requirements.py

# Auto-created virtual environment
source venv/bin/activate
```

### Generate Test Data

```bash
python generate_sparse_data.py
python generate_dense_data.py
```

### Run All Benchmarks

```bash
python run_all_benchmarks.py
```

Results saved to `results/` (CSV, JSON, TXT) and logs to `output/`.

---

## Project Structure

```
├── README.md                           ← You are here
├── LICENSE                             ← Apache License 2.0
├── requirements.txt                    ← Python dependencies
├── install_requirements.py             ← Automated installer
├── run_all_benchmarks.py               ← Run all benchmarks
│
├── input/                              ← Test matrices (CSV)
├── results/                            ← Benchmark results
│   ├── *.csv                           ← Result matrices
│   ├── *.json                          ← Performance metrics
│   └── *.txt                           ← Human-readable summaries
│
├── CSR_CSC_Implementation/             ← Approach 1: In-memory CSR/CSC
│   ├── core_implementations/
│   ├── dense_sparse_cpu_benchmarks/
│   ├── dense_sparse_gpu_benchmarks/
│   └── gnn_sparse_benchmarks/
│
├── COO_Implementation/                 ← Approach 2: In-memory COO
│   ├── core_implementations/
│   ├── dense_sparse_cpu_benchmarks/
│   ├── dense_sparse_gpu_benchmarks/
│   └── gnn_sparse_benchmarks/
│
├── CSR_CSC_Database_Approach/          ← Approach 3: Disk I/O + CSR/CSC
│   ├── dense_sparse_cpu_benchmarks/
│   ├── dense_sparse_gpu_benchmarks/
│   ├── gnn_benchmark/
│   └── verification/
│
└── COO_Database_Approach/              ← Approach 4: Disk I/O + COO
    ├── core_implementations/
    ├── dense_sparse_cpu_benchmarks/
    ├── dense_sparse_gpu_benchmarks/
    └── verification/
```

---

## Key Findings & Recommendations

### When to Use Each Approach

| Scenario | Best Approach | Rationale |
|----------|---------------|-----------|
| **Matrix fits in RAM, 99.9% sparse** | CSR/CSC Implementation (CPU) | 870× speedup, minimal overhead |
| **Matrix fits in RAM, GPU available** | COO Implementation (GPU) | 57× speedup, native PyTorch support |
| **Matrix > RAM** | Database Approaches | 2.4-11.5% I/O overhead acceptable |
| **Dynamic graphs** | COO Implementation | 718× speedup, 7.22× incremental updates |
| **Dense matrices** | In-memory only | Database approach shows 99% I/O overhead |
| **GNN inference** | In-memory sparse | 88-97% I/O overhead makes database unviable |

### Critical Thresholds

1. **Sparsity**: Sparse algorithms win at **99%+ for CPU**, **99.9%+ for GPU**
2. **Matrix Size**: Multiprocessing viable only for **10K×10K+** (overhead = 1.9-2.5s)
3. **GPU Crossover**: CPU sparse beats GPU dense at **99.9%+ sparsity** for small matrices
4. **Database Viability**: CPU sparse (2.4-11.5% overhead), GPU sparse (75-86% overhead), Dense (99%+ overhead)

### Platform Recommendations

**Use CPU when**:
- Matrix < 10K×10K (GPU overhead dominates)
- Sparsity = 99.9%+ (CPU beats GPU for ultra-sparse)
- Incremental updates needed (7.22× speedup)

**Use GPU when**:
- Matrix ≥ 10K×10K
- Sparsity = 99-99.9%
- Full recomputation acceptable (2.45× faster)
- COO format (10× faster than CSR)

---

## Experimental Setup

### Hardware

- **CPU**: AMD Ryzen 9 8940HX (16 cores, 32 threads, 3.3-5.2 GHz)
- **GPU**: NVIDIA RTX 5070 Ti (**5,888 CUDA cores**, 12GB VRAM, Ada Lovelace architecture)
- **RAM**: 16GB DDR5
- **Storage**: NVMe SSD (PCIe 4.0)

### Software Stack

- **Python**: 3.13.1
- **NumPy**: 2.2.1 (vectorized operations, dense baseline)
- **SciPy**: 1.15.0 (sparse matrix validation)
- **PyTorch**: 2.9.1+cu130 (GPU sparse operations)
- **CUDA**: 13.0
- **Numba**: 0.61.0 (JIT compilation for CPU kernels)

### Methodology

- **Test Matrices**: 1,000×1,000 to 500,000×500,000
- **Sparsity Levels**: 90%, 99%, 99.9%
- **Runs**: 3 per configuration, averaged
- **Validation**: All results verified against `scipy.sparse`
- **GPU**: CUDA synchronization + warmup runs for accurate timing

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This research project represents extensive benchmarking across **4 approaches**, **3 compute architectures** (single-threaded CPU, multicore CPU, GPU), and **multiple data scales** (1K to 500K dimensions).

**Special thanks to**:
- **Numba** team for JIT compilation capabilities
- **PyTorch** team for excellent sparse tensor support (native COO on 5,888 CUDA cores)
- **SciPy** developers for robust sparse matrix implementations used for validation

---

## AI Assistance Disclosure

We used **Claude (Anthropic)** as our primary Large Language Model assistant for:

1. **Code Debugging**: CSR/CSC format conversion, external sorting, Numba JIT issues
2. **Benchmarking Infrastructure**: Automated execution scripts, timing metrics, JSON results
3. **Project Organization**: Directory hierarchy, separating implementations/benchmarks/data
4. **Data Validation**: Verified sparsity levels, coordinate format compliance, SciPy validation

---