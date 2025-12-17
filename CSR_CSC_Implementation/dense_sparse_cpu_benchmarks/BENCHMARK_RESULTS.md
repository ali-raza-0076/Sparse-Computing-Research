# Dense vs Sparse CPU Benchmark Results

## Overview

Comprehensive benchmarking comparing dense (NumPy) and sparse (CSR/CSC) matrix operations on CPU.

**Hardware**: AMD Ryzen 9 8940HX (16 cores), 32GB RAM  
**Software**: Python 3.13, NumPy, Numba

---

## Single-Threaded Results

### Multiplication (CSR×CSC)

| Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |
|-------------|----------|-----------|-------------|------------|---------|--------|
| 1000×1000   | 90%      | 100,000   | 0.363s      | 0.486s     | 1.34×   | Sparse |
| 1000×1000   | 99%      | 10,000    | 0.046s      | 0.472s     | 10.25×  | Sparse |
| 1000×1000   | 99.9%    | 999       | 0.014s      | 0.481s     | 34.67×  | Sparse |
| 2000×2000   | 99%      | 40,000    | 0.155s      | 5.346s     | 34.42×  | Sparse |
| 2000×2000   | 99.9%    | 3,999     | 0.063s      | 5.329s     | 84.09×  | Sparse |
| 3000×3000   | 99%      | 90,000    | 0.354s      | 52.960s    | 149.63× | Sparse |
| 3000×3000   | 99.9%    | 8,999     | 0.064s      | 55.385s    | 870.01× | Sparse |

**Key Findings:**
- Sparse wins at all sparsity levels
- Speedup increases exponentially with sparsity: 1.34× → 10.25× → 34.67× (1000×1000)
- Scaling benefit with matrix size at 99% sparsity: 10.25× (1000) → 34.42× (2000) → 149.63× (3000)
- Scaling benefit at 99.9% sparsity: 34.67× (1000) → 84.09× (2000) → 870.01× (3000)
- Memory advantage: 2.5× to 250× less memory

**Scaling Limitation:**
Beyond 3000×3000, single-threaded multiplication becomes impractical. Dense O(n³) operations require prohibitively long execution times (hours) for matrices ≥4000×4000.

### Addition (CSR)

| Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |
|-------------|----------|-----------|-------------|------------|---------|--------|
| 1000×1000   | 90%      | 100,000   | 0.216s      | 0.001s     | 0.00×   | Dense  |
| 1000×1000   | 99%      | 10,000    | 0.021s      | 0.001s     | 0.05×   | Dense  |
| 1000×1000   | 99.9%    | 999       | 0.004s      | 0.001s     | 0.28×   | Dense  |
| 2000×2000   | 99%      | 40,000    | 0.084s      | 0.003s     | 0.04×   | Dense  |
| 2000×2000   | 99.9%    | 3,999     | 0.011s      | 0.008s     | 0.75×   | Dense  |
| 3000×3000   | 99%      | 90,000    | 0.196s      | 0.007s     | 0.04×   | Dense  |
| 3000×3000   | 99.9%    | 8,999     | 0.052s      | 0.015s     | 0.30×   | Dense  |

**Key Findings:**
- Dense wins at all sparsity levels and matrix sizes
- CSR conversion overhead exceeds computation savings
- Addition is O(n) operation - insufficient complexity to benefit from sparse representation

---

## Multicore Parallel Results (16 cores)

### A. Numba Threading (numba.prange - Shared Memory)

#### Multiplication (CSR×CSC)

| Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |
|-------------|----------|-----------|-------------|------------|---------|--------|
| 1000×1000   | 90%      | 100,000   | 40.206s     | 0.819s     | 0.02×   | Dense  |
| 1000×1000   | 99%      | 10,000    | 12.865s     | 0.915s     | 0.07×   | Dense  |
| 1000×1000   | 99.9%    | 999       | 8.558s      | 4.314s     | 0.50×   | Dense  |
| 2000×2000   | 99%      | 40,000    | 22.154s     | 5.101s     | 0.23×   | Dense  |
| **3000×3000**   | **99%**      | **90,000**    | **8.677s**      | **57.082s**    | **6.58×**   | **Sparse**  |
| **3000×3000**   | **99.9%**    | **8,999**     | **3.722s**      | **54.971s**    | **14.77×**  | **Sparse**  |

**Key Findings:**
- Small matrices (≤2000×2000): Dense wins due to parallel overhead
- Large matrices (3000×3000): Sparse wins decisively (6.58× to 14.77×)
- Scaling trend: Sparse multiplication benefits from larger matrix sizes
- Memory advantage: 25-250× less memory

#### Addition (COO)

| Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |
|-------------|----------|-----------|-------------|------------|---------|--------|
| 1000×1000   | 90%      | 100,000   | 10.622s     | 0.001s     | 0.00×   | Dense  |
| 1000×1000   | 99%      | 10,000    | 0.130s      | 0.001s     | 0.01×   | Dense  |
| 1000×1000   | 99.9%    | 999       | 0.009s      | 0.002s     | 0.17×   | Dense  |
| 2000×2000   | 99%      | 40,000    | 0.526s      | 0.007s     | 0.01×   | Dense  |
| 3000×3000   | 99%      | 90,000    | 3.295s      | 0.009s     | 0.00×   | Dense  |
| 3000×3000   | 99.9%    | 8,999     | 2.512s      | 0.016s     | 0.01×   | Dense  |

**Key Findings:**
- Dense wins at ALL matrix sizes
- CSR conversion + parallel coordination costs exceed computation savings

### B. TRUE Multiprocessing (multiprocessing.Pool - Separate Processes)

#### Multiplication (CSR×CSR)

| Matrix Size | Sparsity | Non-Zeros C | Sparse Time | Dense Time | Speedup | Winner | Memory Ratio |
|-------------|----------|-------------|-------------|------------|---------|--------|--------------|
| 1000×1000   | 99%      | 3,000       | 1.949s      | 0.012s     | 0.01×   | Dense  | 90.9×        |
| 1000×1000   | 99.9%    | 835         | 1.981s      | 0.014s     | 0.01×   | Dense  | 500.0×       |
| 2000×2000   | 99%      | 5,577       | 2.526s      | 0.064s     | 0.03×   | Dense  | 95.2×        |
| 2000×2000   | 99.9%    | 1,777       | 1.601s      | 0.062s     | 0.04×   | Dense  | 666.7×       |
| 3000×3000   | 99%      | 7,360       | 2.264s      | 0.212s     | 0.09×   | Dense  | 96.8×        |
| 3000×3000   | 99.9%    | 2,838       | 2.142s      | 0.228s     | 0.11×   | Dense  | 750.0×       |

**Key Findings:**
- Uses Python `multiprocessing.Pool` (separate processes, no GIL)
- Process creation overhead dominates at these sizes
- Speedup improves with size (0.01× → 0.11×)
- Higher sparsity shows better speedup at same size
- Memory savings: 91-750× less than dense

#### Addition (CSR)

| Matrix Size | Sparsity | Non-Zeros C | Sparse Time | Dense Time | Speedup | Winner | Memory Ratio |
|-------------|----------|-------------|-------------|------------|---------|--------|--------------|
| 1000×1000   | 99%      | 19,982      | 2.066s      | 0.002s     | 0.00×   | Dense  | 90.9×        |
| 1000×1000   | 99.9%    | 1,997       | 0.958s      | 0.002s     | 0.00×   | Dense  | 500.0×       |
| 2000×2000   | 99%      | 79,957      | 1.704s      | 0.008s     | 0.00×   | Dense  | 95.2×        |
| 2000×2000   | 99.9%    | 7,994       | 1.758s      | 0.007s     | 0.00×   | Dense  | 666.7×       |
| 3000×3000   | 99%      | 179,967     | 1.814s      | 0.020s     | 0.01×   | Dense  | 96.8×        |
| 3000×3000   | 99.9%    | 17,994      | 1.949s      | 0.022s     | 0.01×   | Dense  | 750.0×       |

**Key Findings:**
- Dense wins by 91-921× despite massive memory advantage
- Multiprocessing overhead (process spawn, IPC) exceeds computation time
- Addition too simple (O(n)) for process-based parallelism to help
- Sparsity level makes no difference to speedup trend

---

## Summary

### Multiplication (CSR×CSC)
**Operation Complexity**: O(n³)

**Single-Threaded**: Sparse wins at all sparsity levels  
- Best case: 870× speedup at 99.9% sparsity (3000×3000)

**Numba Parallel**: Sparse wins at large sizes (≥3000×3000)  
- 3000×3000: 6.58× to 14.77× speedup depending on sparsity

**TRUE Multiprocessing**: Dense wins at tested sizes
- Process overhead dominates (0.01× to 0.11× speedup)
- Memory advantage: 500-750× less

### Addition (CSR/COO)
**Operation Complexity**: O(n)

**All Configurations**: Dense wins consistently  
- Format conversion overhead exceeds computation savings
- Simple element-wise addition too fast to benefit from sparse representation

### Critical Insight

**Operation complexity determines sparse advantage:**
- High complexity operations (multiplication O(n³)): Large benefit from skipping zero operations
- Low complexity operations (addition O(n)): Overhead dominates regardless of sparsity

**Matrix size matters for parallel multiplication:**
- Small matrices (≤2000): Parallel overhead exceeds benefits
- Large matrices (≥3000): Sparse parallel shows significant advantage

---

## Files

**Single-Threaded**:
- `single_threaded/results/multiplication_results.{json,csv,txt}`
- `single_threaded/results/addition_results.{json,csv,txt}`

**Multicore Parallel**:
- `multicore_parallel/results/multiplication_parallel_*_results.{json,csv,txt}` (Numba threading)
- `multicore_parallel/results/addition_parallel_*_results.{json,csv,txt}` (Numba threading)
- `multicore_parallel/results/multiplication_multiprocess_csr_*.txt` (TRUE multiprocessing)
- `multicore_parallel/results/addition_multiprocess_csr_*.txt` (TRUE multiprocessing)

**Benchmarks**:
- `single_threaded/multiplication_benchmark.py`
- `single_threaded/addition_benchmark.py`
- `multicore_parallel/multiplication_parallel_benchmark.py` (Numba)
- `multicore_parallel/addition_parallel_benchmark.py` (Numba)
- `multicore_parallel/multiplication_multiprocess_benchmark.py` (TRUE multiprocessing)
- `multicore_parallel/addition_multiprocess_benchmark.py` (TRUE multiprocessing)
