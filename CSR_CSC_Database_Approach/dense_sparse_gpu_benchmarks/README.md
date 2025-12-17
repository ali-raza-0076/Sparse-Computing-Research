================================================================================
    CSR/CSC GPU BENCHMARKS - DATABASE I/O WORKFLOW
================================================================================


OVERVIEW
-----------------

Sparse matrix operations using PyTorch CSR format with database I/O workflow.

Hardware: Google Colab T4 GPU


COMMANDS
-----------------

Run from project root:

Sparse benchmarks (50K x 50K, 100K entries):
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_gpu_benchmarks\addition_gpu_benchmark.py
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_gpu_benchmarks\multiplication_gpu_benchmark.py

Dense benchmarks (2000 x 2000, 4M entries):
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_gpu_benchmarks\addition_dense_gpu_benchmark.py
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_gpu_benchmarks\multiplication_dense_gpu_benchmark.py


================================================================================
RESULTS
================================================================================


Sparse Benchmarks (50K x 50K, 100K entries, 0.004% dense, T4 GPU)
------------------------------------------------------------------

Operation             Total Time    I/O Read    GPU Compute    I/O Write    I/O %    Throughput    Result NNZ
--------------------------------------------------------------------------------
Addition (CSR)        0.513s        0.081s      0.177s         0.256s       65.6%    195K/s        100K
Multiplication        0.736s        0.077s      0.229s         0.430s       68.9%    330K/s        243K


Dense Benchmarks (2000 x 2000, 4M entries, 100% dense, T4 GPU)
---------------------------------------------------------------

Operation             Total Time    I/O Read    GPU Compute    I/O Write    I/O %    Throughput
--------------------------------------------------------------------------------
Addition              15.49s        12.14s      0.155s         3.20s        99.0%    773K/s
Multiplication        15.82s        12.24s      0.189s         3.39s        98.8%    759K/s


KEY FINDINGS
-----------------

Sparse (T4 GPU):
  I/O overhead 65.6-68.9%, GPU compute 31.1-34.4% (I/O dominated even for sparse)

Dense (T4 GPU):
  I/O overhead 98.8-99.0%, GPU compute 1.0-1.2% (I/O dominates completely)

Platform:
  All benchmarks run successfully on Google Colab T4 GPU

PyTorch behavior:
  Multiplication produces 1000x denser results (243K vs 242 nonzeros) compared
  to Numba implementation due to different algorithm

Database workflow:
  Not viable for GPU due to high I/O overhead even for sparse matrices


TECHNICAL NOTES
-----------------

  * Input: Matrix A (50,001 x 50,000), Matrix B (50,001 x 50,001), 100K non-zeros each
  * Format: CSR (Compressed Sparse Row) for PyTorch operations
  * Device: Google Colab T4 GPU
  * Why B * A? The test matrices have non-square dimensions. For matrix multiplication
    A * B requires A.cols == B.rows (50,000 != 50,001 - incompatible), while B * A
    requires B.cols == A.rows (50,001 == 50,001 - compatible). This ensures GPU
    benchmarks run successfully with the available test data.
  * Phase 1: Read COO format CSV from disk
  * Phase 2: Convert to CSR and run PyTorch sparse operations on GPU
  * Phase 3: Convert back to COO and write result CSV to disk
  * Algorithm difference: PyTorch sparse multiplication has different numerical
    behavior than Numba CSR * CSC


OUTPUT FILES
-----------------

  * results/addition_gpu_csr_.csv - Addition result matrix
  * results/multiplication_gpu_sparse_.csv - Multiplication result matrix
  * results/metrics_addition_gpu_csr.json - Detailed timing metrics
  * results/metrics_multiplication_gpu_sparse.json - Detailed timing metrics

