================================================================================
    COO DATABASE APPROACH
================================================================================

Pure COO (Coordinate) format implementation using simple (row, col, value)
triplets with database I/O workflow.


STRUCTURE
-----------------

COO_Database_Approach/
  * core_implementations/          - Addition & multiplication algorithms
  * dense_sparse_cpu_benchmarks/   - CPU benchmarks (single/parallel)
  * dense_sparse_gpu_benchmarks/   - GPU benchmarks (PyTorch)
  * gnn_benchmark/                 - GNN inference benchmarks
  * verification/                  - Correctness verification scripts
  * requirements.txt


================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================

System: AMD Ryzen 9 8940HX (32 cores), Python 3.13/3.14, PyTorch 2.9.1+cpu


CPU BENCHMARKS - SPARSE (50K x 50K, 100K entries, 0.004% dense)
----------------------------------------------------------------

Operation         Single-threaded    Parallel (32 cores)    I/O Overhead    Speedup
--------------------------------------------------------------------------------
Addition          1.341s             1.010s                 18-22%          1.33x
Multiplication    2.600s             2.420s                 9%              1.07x


CPU BENCHMARKS - SPARSE (500K x 500K, 500K entries, 0.0002% dense)
-------------------------------------------------------------------

Operation         Single-threaded    Parallel (32 cores)    I/O Overhead    Speedup
--------------------------------------------------------------------------------
Addition          13.09s             20.18s                 11.51%          0.65x
Multiplication    52.63s             52.87s                 2.35%           1.00x


CPU BENCHMARKS - DENSE (2000 x 2000, 4M entries, 100% dense)
-------------------------------------------------------------

Operation         Single-threaded    Parallel (32 cores)    I/O Overhead    Speedup
--------------------------------------------------------------------------------
Addition          23.01s             21.04s                 99.97%          1.09x
Multiplication    21.56s             21.24s                 99.68-99.77%    1.02x


CPU BENCHMARKS - DENSE (3000 x 3000, 9M entries, 100% dense)
-------------------------------------------------------------

Operation         Single-threaded    Parallel (32 cores)    I/O Overhead    Speedup
--------------------------------------------------------------------------------
Addition          97.54s             96.09s                 99.98%          1.02x
Multiplication    102.62s            96.30s                 99.79-99.80%    1.07x


GPU BENCHMARKS (PYTORCH ON GOOGLE COLAB T4 GPU)
------------------------------------------------

Matrix Type           Addition    Multiplication    I/O Overhead
--------------------------------------------------------------------------------
Sparse (50K x 50K,    0.544s      0.580s            54.8%
  100K)

Sparse (500K x 500K,  2.806s      2.490s            78.02-83.93%
  500K)

Dense (2000 x 2000,   15.83s      15.92s            99.77-99.98%
  4M)

Dense (3000 x 3000,   30.58s      33.25s            99.07-99.17%
  9M)


GNN BENCHMARKS (2-LAYER GCN)
-----------------------------

Graph         Vertices    Edges      Total Time    I/O Overhead    GNN Compute
--------------------------------------------------------------------------------
Cora          2,708       10,832     0.289s        95.7%           0.012s (4%)
CiteSeer      3,327       13,308     0.407s        96.8%           0.013s (3%)
PubMed        19,717      98,584     3.973s        97.3%           0.107s (3%)


KEY FINDINGS
-----------------

Sparse matrices (0.004% dense):
  * I/O overhead: 9-22% for CPU, 54.8% for GPU
  * GPU execution successful on T4 hardware
  * Compute dominates: Database I/O workflow efficient
  * Parallel speedup: 1.07-1.33x on 32 cores

Dense matrices (100% dense):
  * I/O overhead: 99.7-99.98% (completely dominates)
  * Database I/O approach inefficient for dense matrices
  * Minimal benefit from parallelization

GNN inference:
  * I/O overhead: 95-97% (disk operations dominate)
  * GNN compute very fast: 0.012-0.107s per forward pass
  * Database workflow adds significant overhead for small compute tasks

Conclusion:
  Database I/O approach viable only for sparse matrices (less than 1% dense)
  with compute-intensive operations


RUNNING BENCHMARKS
-----------------

See individual README files in each folder for detailed results and analysis.


CPU Benchmarks:

  Sparse matrices (50K x 50K, 100K entries)
  python dense_sparse_cpu_benchmarks/single_threaded/addition_benchmark.py
  python dense_sparse_cpu_benchmarks/single_threaded/multiplication_benchmark.py
  python dense_sparse_cpu_benchmarks/multicore_parallel/addition_parallel_benchmark.py
  python dense_sparse_cpu_benchmarks/multicore_parallel/multiplication_parallel_benchmark.py

  Dense matrices (2000 x 2000, 4M entries)
  python dense_sparse_cpu_benchmarks/single_threaded/addition_dense_benchmark.py
  python dense_sparse_cpu_benchmarks/single_threaded/multiplication_dense_benchmark.py
  python dense_sparse_cpu_benchmarks/multicore_parallel/addition_dense_parallel_benchmark.py
  python dense_sparse_cpu_benchmarks/multicore_parallel/multiplication_dense_parallel_benchmark.py


GPU Benchmarks:

  Sparse matrices
  python dense_sparse_gpu_benchmarks/addition_gpu_benchmark.py
  python dense_sparse_gpu_benchmarks/multiplication_gpu_benchmark.py

  Dense matrices
  python dense_sparse_gpu_benchmarks/addition_dense_gpu_benchmark.py
  python dense_sparse_gpu_benchmarks/multiplication_dense_gpu_benchmark.py


GNN Benchmarks:

  Generate synthetic graph data (run once)
  python gnn_benchmark/generate_data.py

  Run GNN inference benchmarks
  python gnn_benchmark/gnn_static_benchmark.py --graph all


TECHNICAL NOTES
-----------------

  * Format: COO (i,j,value) triplets - simpler than CSR/CSC but 2-10x slower
  * Algorithms: Addition uses merge-sort, multiplication uses hash-based lookup
  * Optimization: Numba JIT compilation
  * GPU: Google Colab T4 GPU (PyTorch)
  * GNN: 2-layer GCN with symmetric normalization


