================================================================================
    CSR/CSC DATABASE APPROACH
================================================================================

Sparse matrix operations using CSR/CSC formats with database I/O workflow
(read CSV, compute, write CSV).


TEST MATRICES
-----------------

Sparse: 50,001 x 50,000 with 100K nonzeros (0.004% density), ~2MB CSV files
Dense: 2000 x 2000 with 4M entries (100% density), ~51MB CSV files


DATA FLOW
-----------------

1. Phase 1: Read COO format from CSV
2. Phase 2: Convert to CSR/CSC and compute (Addition: CSR+CSR, Multiplication: CSR * CSC)
3. Phase 3: Write result back to CSV

Conversion uses custom Numba JIT implementations with external merge sort for CSC.


================================================================================
SPARSE MATRIX BENCHMARKS (100K ENTRIES)
================================================================================

System: Python 3.13, NumPy 2.2.1, Numba 0.61.0


CPU Single-Threaded (Custom Numba)
-----------------------------------

Operation             Time      I/O      Compute    Result
--------------------------------------------------------------------------------
Addition (CSR)        1.161s    16%      84%        65K nonzeros
Multiplication        1.173s    9%       91%        242 nonzeros
  (CSR * CSC)


CPU Single-Threaded (Custom Numba) - 500K x 500K (500K entries, 0.0002% density)
---------------------------------------------------------------------------------

Operation             Time      I/O      Compute    Result
--------------------------------------------------------------------------------
Addition (CSR)        35.03s    8.6%     91.4%      1M nonzeros
Multiplication        18.66s    4.2%     95.8%      88 nonzeros
  (CSR * CSC)


GPU (PyTorch on Google Colab T4 GPU)
-------------------------------------

Sparse (100K entries, 50K x 50K):

Operation             Time      I/O      Compute    Result
--------------------------------------------------------------------------------
Addition              0.513s    66%      34%        100K nonzeros
Multiplication        0.736s    69%      31%        243K nonzeros

Sparse (500K entries, 500K x 500K):

Operation             Time      I/O      Compute    Result
--------------------------------------------------------------------------------
Addition (CSR)        3.747s    85.58%   14.42%     1M nonzeros
Multiplication        2.477s    75.75%   24.25%     500K nonzeros
  (Sparse)

PyTorch multiplication produces 1000x denser results (243K vs 242) due to
different algorithm.


GNN BENCHMARKS (CSR FORMAT)
---------------------------

Dataset             Vertices    Edges      Time      I/O Overhead
--------------------------------------------------------------------------------
Cora                2,708       10,832     0.284s    89%
CiteSeer            3,327       13,308     0.391s    98%
PubMed              19,717      98,584     3.088s    98%


KEY FINDINGS (SPARSE)
---------------------

  * CSR/CSC is 2.22x faster than COO for multiplication
  * Custom Numba: Low I/O overhead (9-16%), optimal for database workflows
  * PyTorch GPU: Fast absolute times but higher I/O ratio (66-69%) even on T4 GPU
  * GNN workloads: I/O dominated (88-98%), database approach not viable for inference


================================================================================
DENSE MATRIX BENCHMARKS
================================================================================

Note: Dense matrices use NumPy arrays, not CSR/CSC format.

Dense (4M entries, 2000 x 2000):

Operation             Device         Time      I/O Overhead    Compute
--------------------------------------------------------------------------------
Addition              CPU Single     21.45s    99.97%          0.03%
                      CPU Parallel   21.15s    99.98%          0.02%
                      GPU (T4)       15.49s    99.00%          1.00%

Multiplication        CPU Single     21.26s    99.65%          0.35%
                      CPU Parallel   21.21s    99.74%          0.26%
                      GPU (T4)       15.82s    98.81%          1.19%

Dense (9M entries, 3000 x 3000):

Operation             Device         Time      I/O Overhead    Compute
--------------------------------------------------------------------------------
Addition              GPU (T4)       35.12s    99.08%          0.92%
Multiplication        GPU (T4)       34.13s    99.22%          0.78%


KEY FINDINGS (DENSE)
--------------------

  * Database I/O approach NOT viable: 99%+ I/O overhead
  * GPU advantage negated by I/O bottleneck
  * Multicore provides no benefit (I/O bound)
  * CSV I/O for 4M entries takes 15-21s regardless of compute device


================================================================================
CONCLUSION
================================================================================

Sparse matrices (100K entries):
  Database approach viable with 9-16% I/O overhead for custom CPU
  implementations. CSR/CSC is 2.22x faster than COO for multiplication.

Dense matrices (4M entries):
  Database approach NOT viable with 99%+ I/O overhead across all
  implementations. CSV I/O bottleneck makes compute device irrelevant.

Verdict:
  Database approach with CSV storage only effective for sparse matrices where
  computation significantly exceeds I/O time.


