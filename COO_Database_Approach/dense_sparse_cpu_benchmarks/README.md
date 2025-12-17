================================================================================
    COO DATABASE I/O BENCHMARKS
================================================================================


OVERVIEW
-----------------

COO sparse matrix operations with database I/O workflow:
  * Phase 1: Read from disk (CSV)
  * Phase 2: Process in RAM (COO algorithms)
  * Phase 3: Write to disk (CSV)


COMMANDS
-----------------

Run from project root (DB_Project_MatMul/):

Single-threaded:
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_cpu_benchmarks\single_threaded\addition_benchmark.py
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_cpu_benchmarks\single_threaded\multiplication_benchmark.py

Parallel (32 cores):
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_cpu_benchmarks\multicore_parallel\addition_parallel_benchmark.py --num-runs 1
  .\venv313\Scripts\python.exe COO_Implementation\dense_sparse_cpu_benchmarks\multicore_parallel\multiplication_parallel_benchmark.py --num-runs 1


================================================================================
RESULTS
================================================================================

Test System: AMD Ryzen 9 8940HX (32 cores), Python 3.13/3.14


Sparse Benchmarks (50K x 50K, 100K entries, 0.004% dense)
Operation         Mode       Total Time    I/O Overhead         Compute           Throughput    Speedup
--------------------------------------------------------------------------------
Addition          Single     1.341s        0.238s (18%)         1.104s (82%)      298K/s        1.0x
Addition          Parallel   1.010s        0.223s (22%)         0.788s (78%)      396K/s        1.33x
Multiplication    Single     2.600s        0.226s (9%)          2.373s (91%)      154K/s        1.0x
Multiplication    Parallel   2.420s        0.227s (9%)          2.193s (91%)      165K/s        1.07x


Sparse Benchmarks (500K x 500K, 500K entries, 0.0002% dense)
-------------------------------------------------------------

Operation         Mode       Total Time    I/O Overhead         Compute           Throughput    Speedup
--------------------------------------------------------------------------------
Addition          Single     13.09s        1.51s (11.51%)       11.59s (88.49%)   153K/s        1.0x
Addition          Parallel   20.18s        1.42s (7.04%)        18.76s (92.96%)   99K/s         0.65x
Multiplication    Single     52.63s        1.24s (2.35%)        51.40s (97.65%)   29K/s         1.0x
Multiplication    Parallel   52.87s        1.19s (2.24%)        51.68s (97.76%)   28K/s         1.00x


Dense Benchmarks (2000 x 2000, 4M entries, 100% dense)
-------------------------------------------------------

Operation         Mode       Total Time    I/O Time    Compute    I/O Overhead    Throughput
--------------------------------------------------------------------------------
Addition          Single     23.01s        22.99s      0.006s     99.97%          521K/s
Addition          Parallel   21.04s        21.03s      0.006s     99.97%          569K/s
Multiplication    Single     21.56s        21.49s      0.069s     99.68%          557K/s
Multiplication    Parallel   21.24s        21.18s      0.057s     99.73%          565K/s


Dense Benchmarks (3000 x 3000, 9M entries, 100% dense)
-------------------------------------------------------

Operation         Mode       Total Time    I/O Time    Compute    I/O Overhead    Throughput
--------------------------------------------------------------------------------
Addition          Single     97.54s        97.52s      0.024s     99.98%          276K/s
Addition          Parallel   96.09s        96.08s      0.015s     99.98%          281K/s
Multiplication    Single     102.62s       102.41s     0.209s     99.80%          263K/s
Multiplication    Parallel   96.30s        96.10s      0.199s     99.79%          280K/s


KEY FINDINGS
-----------------

Sparse (0.004% dense): I/O overhead 9-22%, compute dominates
Dense (100% dense): I/O overhead 99.7%, database I/O approach inefficient
Parallel speedup: 1.07-1.33x for sparse, minimal for dense
Conclusion: Database I/O viable only for sparse matrices (less than 1% dense)


OUTPUT FILES
-----------------

Each benchmark generates in results/ folder:
  * Result CSV: operation_coo_dims_config.csv (i,j,v format)
  * Metrics JSON: metrics_operation.json (timing breakdown)


TECHNICAL NOTES
-----------------

  * Format: Pure COO (i,j,v triplets), no CSR/CSC conversion
  * Algorithm: Addition uses merge-sort, multiplication uses hash-based lookup
  * Optimization: Numba JIT compilation



