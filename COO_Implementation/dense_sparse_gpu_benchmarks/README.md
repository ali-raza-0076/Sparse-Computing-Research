================================================================================
    GPU SPARSE MATRIX BENCHMARKS - COO FORMAT
================================================================================


OVERVIEW
-----------------

Evaluation of GPU acceleration for sparse matrix operations using Coordinate
(COO) format. Compares PyTorch sparse tensors against dense implementations on
CUDA-enabled GPUs.


HARDWARE CONFIGURATION
-----------------

  * GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
  * CUDA: Version 13.0
  * PyTorch: Version 2.9.1+cu130


METHODOLOGY
-----------------

Format:
  PyTorch native COO sparse tensors (row indices, column indices, values)

Timing Protocol:
  * CUDA synchronization before and after each operation
  * Warmup run to compile CUDA kernels
  * Multiple runs averaged for statistical stability

Test Parameters:
  * Matrix sizes: 1000x1000, 2000x2000, 3000x3000
  * Sparsity levels: 90%, 99%, 99.9%
  * Operations: Addition (COO + COO), Multiplication (COO * COO)

Technical Note - Matrix Multiplication Order:
  When using non-square test matrices (e.g., 50,001 x 50,000), multiplication
  may use B * A instead of A * B to ensure dimension compatibility. For A * B to
  be valid, A.cols must equal B.rows. If this condition is not met, B * A is
  computed instead (where B.cols must equal A.rows). This ensures benchmarks run
  successfully with the available test data.


Results


Matrix Addition (COO + COO)
----------------------------

1000x1000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 100,000   | 0.000150s   | 0.000039s  | 0.26x   | Dense  | 1.14 MB    | 3.81 MB
99.0%    | 10,000    | 0.000298s   | 0.000058s  | 0.19x   | Dense  | 0.11 MB    | 3.81 MB
99.9%    | 999       | 0.000045s   | 0.000032s  | 0.72x   | Dense  | 0.01 MB    | 3.81 MB

2000x2000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 400,000   | 0.000272s   | 0.000269s  | 0.99x   | Dense  | 4.58 MB    | 15.26 MB
99.0%    | 40,000    | 0.000142s   | 0.000281s  | 1.98x   | Sparse | 0.46 MB    | 15.26 MB
99.9%    | 3,999     | 0.000045s   | 0.000274s  | 6.13x   | Sparse | 0.05 MB    | 15.26 MB

3000x3000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 900,000   | 0.000351s   | 0.000611s  | 1.74x   | Sparse | 10.30 MB   | 34.33 MB
99.0%    | 90,000    | 0.000127s   | 0.000492s  | 3.88x   | Sparse | 1.03 MB    | 34.33 MB
99.9%    | 8,999     | 0.000041s   | 0.000483s  | 11.92x  | Sparse | 0.10 MB    | 34.33 MB

10000x10000 Matrices:

Sparsity | Non-Zeros   | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-------------|-------------|------------|---------|--------|------------|------------
90.0%    | 10,000,000  | 0.003800s   | 0.006486s  | 1.71x   | Sparse | 114.44 MB  | 381.47 MB
99.0%    | 1,000,000   | 0.000494s   | 0.004863s  | 9.84x   | Sparse | 11.44 MB   | 381.47 MB
99.9%    | 99,999      | 0.000220s   | 0.004801s  | 21.86x  | Sparse | 1.14 MB    | 381.47 MB

Key Findings:
- Small matrices (1000x1000): Dense operations faster due to GPU overhead costs
- Crossover point: 2000x2000 at 99% sparsity
- Large matrices (10000x10000): Sparse achieves up to 21.86x speedup at 99.9% sparsity
- Memory efficiency: 3x to 333x reduction with sparse format
- Performance scales strongly with both matrix size and sparsity level


Matrix Multiplication (COO * COO)
----------------------------------

1000x1000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 100,000   | 0.002910s   | 0.000183s  | 0.06x   | Dense  | 1.14 MB    | 3.81 MB
99.0%    | 10,000    | 0.001626s   | 0.000330s  | 0.20x   | Dense  | 0.11 MB    | 3.81 MB
99.9%    | 999       | 0.001117s   | 0.000352s  | 0.31x   | Dense  | 0.01 MB    | 3.81 MB

2000x2000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 400,000   | 0.021122s   | 0.000971s  | 0.05x   | Dense  | 4.58 MB    | 15.26 MB
99.0%    | 40,000    | 0.001764s   | 0.001041s  | 0.59x   | Dense  | 0.46 MB    | 15.26 MB
99.9%    | 3,999     | 0.001139s   | 0.001244s  | 1.09x   | Sparse | 0.05 MB    | 15.26 MB

3000x3000 Matrices:

Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-----------|-------------|------------|---------|--------|------------|-----------
90.0%    | 900,000   | 0.484675s   | 0.003095s  | 0.01x   | Dense  | 10.30 MB   | 34.33 MB
99.0%    | 90,000    | 0.002930s   | 0.003010s  | 1.03x   | Sparse | 1.03 MB    | 34.33 MB
99.9%    | 8,999     | 0.001521s   | 0.003697s  | 2.43x   | Sparse | 0.10 MB    | 34.33 MB

10000x10000 Matrices:

Sparsity | Non-Zeros   | Sparse Time | Dense Time | Speedup | Winner | Sparse Mem | Dense Mem
---------|-------------|-------------|------------|---------|--------|------------|------------
90.0%    | 10,000,000  | CUDA Error  | N/A        | N/A     | Failed | 114.44 MB  | 381.47 MB
99.0%    | 1,000,000   | 0.041691s   | 0.099935s  | 2.40x   | Sparse | 11.44 MB   | 381.47 MB
99.9%    | 99,999      | 0.001740s   | 0.099280s  | 57.06x  | Sparse | 1.14 MB    | 381.47 MB

Key Findings:
- Multiplication more sensitive to sparsity than addition
- Crossover point: 2000x2000 at 99.9% sparsity (later than addition)
- Large matrices (10000x10000): Sparse achieves up to 57.06x speedup at 99.9% sparsity
- 90% sparsity at 10000x10000 exceeds GPU memory workspace limits (CUDA error)
- Best results require extreme sparsity (99.9%) at large scales


Critical Analysis

Key Insight - Why Sparse Can Win on GPU:
- Dense operations process ALL elements (including zeros): 10000^2 = 100M operations regardless of sparsity
- Sparse operations process ONLY non-zeros: 99.9% sparsity = 100K operations (1000x less work)
- Tradeoff: Sparse saves computation but suffers from irregular memory access
- Crossover occurs when computation savings outweigh memory access penalties

Addition vs Multiplication Performance:
- Addition: Sparse wins earlier (2000x2000 at 99%)
- Multiplication: Sparse wins later (2000x2000 at 99.9%)
- Multiplication has higher computational complexity making sparse less efficient at lower sparsity

GPU Sparse Limitations:
- Irregular memory access patterns reduce GPU efficiency
- Kernel launch overhead significant for small matrices
- PyTorch COO operations less optimized than cuBLAS dense kernels
- Sparse multiplication requires temporary workspace causing memory pressure

When Sparse Wins:
- Addition: 99%+ sparsity with 2000x2000+ matrices
- Multiplication: 99.9% sparsity with 3000x3000+ matrices
- Memory-constrained scenarios where 3-333x savings critical

Practical Implications:
- Dense GPU dominant for most workloads (under 99% sparsity)
- Sparse GPU beneficial only at extreme sparsity with large scale
- Multiplication benefits more dramatically at 99.9% (57x vs 22x speedup)
- Memory savings consistent regardless of operation type

Why Neural Networks Use Dense GPU Operations:
- Neural network weights typically 0-50% sparse (even after pruning)
- At low sparsity, sparse format overhead exceeds cost of computing zeros
- Dense cuBLAS operations are among the most optimized GPU kernels
- Regular memory access patterns enable coalesced GPU memory access
- Only extreme sparsity (99%+) like graph neural networks benefit from sparse GPU
- Results confirm: 90% sparsity = 20x slower sparse, 99.9% sparsity = 57x faster sparse


Usage

``bash
# Addition benchmarks
python addition_gpu_coo_benchmark.py --size 3000 --sparsity 99.9 --num-runs 3

# Multiplication benchmarks  
python multiplication_gpu_coo_benchmark.py --size 3000 --sparsity 99.9 --num-runs 3
`

Arguments: --size (matrix dimension), --sparsity (90/99/99.9), --num-runs (averaging iterations)

Results saved to: results/{operation}_gpu_coo_{size}x{size}_{sparsity}pct_results.{json,txt,csv}`

