CSR/CSC Database I/O Benchmarks


Overview

CSR/CSC sparse matrix operations with database I/O workflow:
- Phase 1: Read from disk (CSV in COO format)
- Phase 2: Convert to CSR/CSC and process in RAM
- Phase 3: Write result to disk (CSV)


Commands

Run from project root (DB_Project_MatMul/):

``bash
# Single-threaded
.\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_cpu_benchmarks\single_threaded\addition_benchmark.py
.\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_cpu_benchmarks\single_threaded\multiplication_benchmark.py

# Parallel (NumPy multicore for dense)
.\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_cpu_benchmarks\multicore_parallel\addition_dense_parallel_benchmark.py
.\venv313\Scripts\python.exe CSR_CSC_Database_Approach\dense_sparse_cpu_benchmarks\multicore_parallel\multiplication_dense_parallel_benchmark.py
`


Results

Test System: AMD Ryzen 9 8940HX (32 cores), Python 3.13, NumPy 2.2.1, Numba 0.61.0


Sparse Benchmarks (50K×50K, 100K entries, 0.004% dense) | Operation | Mode | Total Time | I/O Overhead | Compute | Throughput | Result NNZ |  | ----------- | ------ | ------------ | -------------- | --------- | ------------ | ------------ |  | Addition (CSR) | Single | 1.161s | 16.4% | 83.6% | 86K/s | 100K |  | Multiplication (CSR×CSC) | Single | 1.173s | 8.7% | 91.3% | 206/s | 242 | Sparse Benchmarks (500K×500K, 500K entries, 0.0002% dense) | Operation | Mode | Total Time | I/O Overhead | Compute | Throughput | Result NNZ |  | ----------- | ------ | ------------ | -------------- | --------- | ------------ | ------------ |  | Addition (CSR) | Single | 35.03s | 8.58% | 91.42% | 28.6K/s | 1M |  | Multiplication (CSR×CSC) | Single | 18.66s | 4.20% | 95.80% | 5/s | 88 | Dense Benchmarks (2000×2000, 4M entries, 100% dense) | Operation | Mode | Total Time | I/O Time | Compute | I/O Overhead | Throughput |  | ----------- | ------ | ------------ | ---------- | --------- | -------------- | ------------ |  | Addition | Single | 21.45s | 21.44s | 0.006s | 99.97% | 558K/s |  | Addition | Parallel | 21.15s | 21.15s | 0.005s | 99.98% | 566K/s |  | Multiplication | Single | 21.26s | 21.19s | 0.075s | 99.65% | 564K/s |  | Multiplication | Parallel | 21.21s | 21.16s | 0.056s | 99.74% | 566K/s | Dense Benchmarks (3000×3000, 9M entries, 100% dense) | Operation | Mode | Total Time | I/O Time | Compute | I/O Overhead | Throughput |  | ----------- | ------ | ------------ | ---------- | --------- | -------------- | ------------ |  | Addition | Single | 104.20s | 104.15s | 0.051s | 99.95% | 259K/s |  | Multiplication | Single | 99.06s | 98.48s | 0.585s | 99.41% | 273K/s | Key Findings

Sparse (0.004% dense): I/O overhead 8.7-16.4%, compute dominates
Sparse (0.0002% dense, 500K×500K): I/O overhead 4.2-8.6%, even better scaling  
Dense (100% dense): I/O overhead 99.65-99.98%, database I/O approach inefficient  
Dense (100% dense, 3K×3K): I/O overhead 99.41-99.95%, scales poorly with size
CSR/CSC vs COO: CSR×CSC multiplication 2.22× faster than COO for sparse  
Parallel speedup: Minimal for dense (I/O bound)  
Conclusion: Database I/O viable only for sparse matrices where compute >> I/O


Output Files

Each benchmark generates in results/ folder:
- Result CSV: <operation>_<format>_<dims>.csv (i,j,v format)
- Metrics JSON: metrics_<operation>_<format>.json` (timing breakdown)


Technical Notes

- Format: CSR for addition (CSR+CSR), CSR×CSC for multiplication
- Conversion: COO ? CSR/CSC using custom Numba JIT with external merge sort
- Algorithm: CSR uses direct indexing (faster than COO hash lookup)
- Optimization: Numba JIT compilation for CSR/CSC operations

