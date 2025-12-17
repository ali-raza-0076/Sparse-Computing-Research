"""
COO Matrix Multiplication Benchmark (Single-Threaded, Database I/O Workflow)

Database I/O Workflow:
- Phase 1: Read from secondary storage (CSV files)
- Phase 2: Process in RAM (COO multiplication algorithm)
- Phase 3: Write to secondary storage (CSV result)

Features:
- Format: Pure COO (coordinate list: i,j,v triplets)
- Parallelism: Single-threaded (no multicore)
- Metrics: I/O time breakdown, compute time, throughput, overhead percentages

Output:
- results/multiplication_coo_<dims>_<nnz>.csv (result matrix)
- results/metrics_multiplication.json (detailed performance metrics)

Note: Computes B×A due to dimension compatibility (B: 50001×50001, A: 50001×50000)
"""
import numpy as np
import time
from scipy import sparse as sp
import csv
import os
from tabulate import tabulate
import json
from tqdm import tqdm
import sys
import argparse

import importlib.util
core_impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core_implementations/sparse_multiplication_coo.py'))
spec = importlib.util.spec_from_file_location('sparse_multiplication_coo', core_impl_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec for sparse_multiplication_coo from {core_impl_path}")
sparse_multiplication_coo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_multiplication_coo)
COOMatrix = sparse_multiplication_coo.COOMatrix
sparse_multiply_coo = sparse_multiplication_coo.sparse_multiply_coo


def read_coo_csv(filepath, transpose=False):
    data = []
    max_i = 0
    max_j = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
                if transpose:
                    data.append((j, i, v))
                    if j > max_i:
                        max_i = j
                    if i > max_j:
                        max_j = i
                else:
                    data.append((i, j, v))
                    if i > max_i:
                        max_i = i
                    if j > max_j:
                        max_j = j
    shape = (max_i + 1, max_j + 1)
    return COOMatrix(shape=shape, data=data)


def run_comparison_from_files(input_a, input_b, num_runs=1):
    """
    Run multiplication using two input CSV files in i,j,v format.
    Records detailed I/O and computation metrics following database I/O workflow:
    Phase 1: Read from secondary storage → Phase 2: Process in RAM → Phase 3: Write to secondary storage
    """
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: COO Multiplication Benchmark")
    print(f"{'='*70}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    coo_A = read_coo_csv(input_a)
    coo_B = read_coo_csv(input_b)
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    nnz_A = coo_A.count_nnz()
    nnz_B = coo_B.count_nnz()
    
    print(f"  Matrix A: {input_a}")
    print(f"    Shape: {coo_A.shape}, {nnz_A:,} non-zeros")
    print(f"  Matrix B: {input_b}")
    print(f"    Shape: {coo_B.shape}, {nnz_B:,} non-zeros")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    if coo_B.shape[1] != coo_A.shape[0]:
        raise ValueError(f"Incompatible dimensions: B is {coo_B.shape}, A is {coo_A.shape}. Need B.cols == A.rows")
    
    print(f"  Computing B×A (compatible dimensions)")
    
    print(f"\n[PHASE 2] Processing in RAM...")
    compute_start = time.perf_counter()
    times = []
    for _ in range(num_runs):
        run_start = time.perf_counter()
        result = sparse_multiply_coo(coo_B, coo_A)
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    nnz_result = len(result.data)
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result shape: {result.shape}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"multiplication_coo_BxA_{result.shape[0]}x{result.shape[1]}_result.csv"
    out_path = os.path.join(results_dir, out_name)
    
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in result.data:
            writer.writerow([i, j, v])
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    print(f"  I/O Write Time: {write_time:.6f}s")
    print(f"  Result written to: {out_path}")
    
    total_time = read_time + compute_time + write_time
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):  {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (Compute): {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write):   {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {read_time + write_time:.6f}s ({(read_time + write_time)/total_time*100:.2f}%)")
    print(f"Throughput:       {(nnz_A + nnz_B + nnz_result)/total_time:,.0f} entries/sec")
    print(f"{'='*70}")
    
    metrics = {
        "operation": "multiplication",
        "execution_mode": "single_threaded",
        "matrix_a": {
            "file": input_a,
            "shape": list(coo_A.shape),
            "nnz": nnz_A
        },
        "matrix_b": {
            "file": input_b,
            "shape": list(coo_B.shape),
            "nnz": nnz_B
        },
        "result": {
            "operation": "B×A",
            "shape": list(result.shape),
            "nnz": nnz_result,
            "file": out_path
        },
        "timing": {
            "phase1_read_time_sec": read_time,
            "phase2_compute_time_sec": compute_time,
            "phase3_write_time_sec": write_time,
            "total_time_sec": total_time,
            "io_overhead_sec": read_time + write_time,
            "compute_avg_sec": avg_time,
            "compute_std_sec": std_time
        },
        "percentages": {
            "read": round(read_time/total_time*100, 2),
            "compute": round(compute_time/total_time*100, 2),
            "write": round(write_time/total_time*100, 2),
            "io_overhead": round((read_time + write_time)/total_time*100, 2)
        },
        "throughput_entries_per_sec": round((nnz_A + nnz_B + nnz_result)/total_time, 2),
        "num_runs": num_runs
    }
    
    metrics_path = os.path.join(results_dir, "metrics_multiplication.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return out_path


def generate_sparse_matrix(size, sparsity_percent, seed):
    """
    Generate sparse matrix with given sparsity.
    
    Args:
        size: Matrix dimension (size × size)
        sparsity_percent: Percentage of zeros (90, 99, 99.9)
        seed: Random seed for reproducibility
    
    Returns:
        rows, cols, values: NumPy arrays for COO format
    """
    np.random.seed(seed)
    
    total_elements = size * size
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(total_elements * density)
    
    rows = np.random.randint(0, size, size=num_entries)
    cols = np.random.randint(0, size, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)
    
    return rows, cols, values


def benchmark_sparse_multiplication(coo_A, coo_B, num_runs=3):
    """
    Benchmark sparse COO×COO multiplication (single-threaded).
    
    Args:
        coo_A, coo_B: COOMatrix objects
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result: Average time, std deviation, result matrix
    """
    _ = sparse_multiply_coo(coo_A, coo_B)
    
    times = []
    for _ in tqdm(range(num_runs), desc="  Sparse COO×COO (single-threaded)", leave=False):
        start = time.perf_counter()
        result = sparse_multiply_coo(coo_A, coo_B)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result


def benchmark_dense_multiplication(A_dense, B_dense, num_runs=3):
    """
    Benchmark dense NumPy matrix multiplication.
    
    Args:
        A_dense, B_dense: NumPy dense arrays
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result: Average time, std deviation, result matrix
    """
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense NumPy (np.matmul)", leave=False):
        start = time.perf_counter()
        result = np.matmul(A_dense, B_dense)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result


def run_comparison(size, sparsity_percent, num_runs=3):
    """
    Run single comparison for one sparsity level.
    
    Args:
        size: Matrix dimension
        sparsity_percent: Sparsity percentage
        num_runs: Number of test iterations
    
    Returns:
        dict: Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Testing: {size}×{size} matrix, {sparsity_percent}% sparsity")
    print(f"{'='*70}")
    
    print("Generating sparse matrices...")
    rows_A, cols_A, vals_A = generate_sparse_matrix(size, sparsity_percent, seed=42)
    rows_B, cols_B, vals_B = generate_sparse_matrix(size, sparsity_percent, seed=123)
    
    data_A = list(zip(rows_A.tolist(), cols_A.tolist(), vals_A.tolist()))
    data_B = list(zip(rows_B.tolist(), cols_B.tolist(), vals_B.tolist()))
    
    coo_A = COOMatrix(shape=(size, size), data=data_A)
    coo_B = COOMatrix(shape=(size, size), data=data_B)
    
    actual_nnz_A = len(data_A)
    actual_nnz_B = len(data_B)
    actual_sparsity_A = 100 * (1 - actual_nnz_A / (size * size))
    actual_sparsity_B = 100 * (1 - actual_nnz_B / (size * size))
    
    print(f"Matrix A: {actual_nnz_A:,} non-zeros ({actual_sparsity_A:.2f}% sparse)")
    print(f"Matrix B: {actual_nnz_B:,} non-zeros ({actual_sparsity_B:.2f}% sparse)")
    
    A_scipy = sp.coo_matrix((vals_A, (rows_A, cols_A)), shape=(size, size))
    B_scipy = sp.coo_matrix((vals_B, (rows_B, cols_B)), shape=(size, size))
    A_dense = A_scipy.toarray()
    B_dense = B_scipy.toarray()
    
    sparse_memory = (actual_nnz_A + actual_nnz_B) * (2 * 4 + 8)
    dense_memory = A_dense.nbytes + B_dense.nbytes
    
    print(f"\nMemory Usage:")
    print(f"  Sparse (COO):  {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"  Dense (NumPy): {dense_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory Ratio:  {dense_memory / sparse_memory:.2f}×")
    
    print(f"\nBenchmarking SPARSE (COO×COO, single-threaded)...")
    sparse_time, sparse_std, C_sparse = benchmark_sparse_multiplication(coo_A, coo_B, num_runs)
    print(f"  Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    
    print(f"\nBenchmarking DENSE (NumPy matmul)...")
    dense_time, dense_std, C_dense = benchmark_dense_multiplication(A_dense, B_dense, num_runs)
    print(f"  Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    print(f"\n{'='*70}")
    print(f"RESULT: {winner} wins with {abs(speedup):.2f}× speedup")
    print(f"{'='*70}")
    
    return {
        "sparsity_percent": sparsity_percent,
        "actual_sparsity_A": actual_sparsity_A,
        "actual_sparsity_B": actual_sparsity_B,
        "matrix_size": size,
        "nnz_A": actual_nnz_A,
        "nnz_B": actual_nnz_B,
        "sparse_time": sparse_time,
        "sparse_std": sparse_std,
        "dense_time": dense_time,
        "dense_std": dense_std,
        "speedup": speedup,
        "winner": winner,
        "memory_sparse_mb": sparse_memory / 1024 / 1024,
        "memory_dense_mb": dense_memory / 1024 / 1024,
        "memory_ratio": dense_memory / sparse_memory,
        "algorithm": "COO×COO (single-threaded)",
        "num_runs": num_runs
    }


def main():
    parser = argparse.ArgumentParser(description='COO multiplication: multiply two input CSVs (i,j,v) and write result to results/.')
    parser.add_argument('--input_a', type=str, default='input/matrix_a_5k.csv', help='Path to matrix_a.csv')
    parser.add_argument('--input_b', type=str, default='input/matrix_b_5k.csv', help='Path to matrix_b.csv')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs (default: 1)')
    args = parser.parse_args()
    run_comparison_from_files(args.input_a, args.input_b, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
