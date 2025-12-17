================================================================================
    COO GPU BENCHMARKS - DATABASE I/O WORKFLOW
================================================================================


OVERVIEW
-----------------

Sparse matrix operations using PyTorch COO format with database I/O workflow.

Hardware: Google Colab T4 GPU


COMMANDS
-----------------

Run from project root:

Sparse benchmarks (50K x 50K, 100K entries):
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_gpu_benchmarks\addition_gpu_benchmark.py
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_gpu_benchmarks\multiplication_gpu_benchmark.py

Dense benchmarks (2000 x 2000, 4M entries):
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_gpu_benchmarks\addition_dense_gpu_benchmark.py
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_gpu_benchmarks\multiplication_dense_gpu_benchmark.py


================================================================================
RESULTS
================================================================================


Sparse Benchmarks (50K x 50K, 100K entries, 0.004% dense, T4 GPU)
------------------------------------------------------------------

Operation         Total Time    I/O Read    GPU Compute    I/O Write    I/O %    Throughput
--------------------------------------------------------------------------------
Addition          0.544s        0.142s      0.246s         0.156s       54.8%    735K/s
Multiplication    0.580s        0.137s      0.285s         0.158s       50.9%    688K/s


Dense Benchmarks (2000 x 2000, 4M entries, 100% dense, T4 GPU)
---------------------------------------------------------------

Operation         Total Time    I/O Read    GPU Compute    I/O Write    I/O %    Throughput
--------------------------------------------------------------------------------
Addition          15.83s        12.61s      0.003s         3.21s        99.98%   757K/s
Multiplication    15.92s        12.45s      0.036s         3.43s        99.77%   754K/s


KEY FINDINGS
-----------------

Sparse (T4 GPU):
  I/O overhead 50.9-54.8%, GPU compute 45.2-49.1% (balanced workload)

Dense (T4 GPU):
  I/O overhead 99.8%, GPU compute 0.02-0.23% (I/O dominates completely)

Platform:
  All benchmarks run successfully on Google Colab T4 GPU

Dense matrices:
  I/O completely dominates due to 100% density (4M entries)


TECHNICAL NOTES
-----------------

  * Input: Matrix A (50,001 x 50,000), Matrix B (50,001 x 50,001), 100K non-zeros each
  * Dimension handling: Addition pads to common shape; Multiplication uses B * A for compatibility
  * Why B * A? The test matrices have non-square dimensions. For matrix multiplication
    A * B requires A.cols == B.rows (50,000 != 50,001 - incompatible), while B * A
    requires B.cols == A.rows (50,001 == 50,001 - compatible). This ensures GPU
    benchmarks run successfully with the available test data.
  * Format: COO (Coordinate List)
  * Device: Google Colab T4 GPU
  * Phase 1: Read CSV from disk
  * Phase 2: PyTorch sparse tensor operations on GPU
  * Phase 3: Write result CSV to disk


OUTPUT FILES
-----------------

  * results/addition_gpu_coo_.csv - Addition result matrix
  * results/multiplication_gpu_coo_BxA_.csv - Multiplication result matrix
  * results/metrics_addition_gpu.json - Detailed timing metrics
  * results/metrics_multiplication_gpu.json - Detailed timing metrics


