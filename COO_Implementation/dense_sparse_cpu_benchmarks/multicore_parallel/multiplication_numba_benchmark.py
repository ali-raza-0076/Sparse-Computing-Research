"""  
Dense vs Sparse Matrix Multiplication Benchmark (Numba Threading)

Compares performance of:
- Dense CPU (NumPy matrix multiplication - single-threaded)
- Sparse CPU (COO×COO Numba threading using set_num_threads)

Test Configuration:
- Matrix Size: Configurable (default: 1,000×1,000)
- Sparsity Levels: 90%, 99%, 99.9%
- Format: COO (Coordinate List)
- Parallelism: Numba threading with set_num_threads
- Runs: 3 iterations per test (averaged)

Command-line Options:
  --size SIZE        Matrix size (default: 1000)
  --sparsity LEVEL   Sparsity level (default: 90)
  --num-runs N       Number of runs (default: 3)
  --num-cores N      Number of cores (default: 16)

Output:
- results/multiplication_numba_{size}x{size}_results.txt
- results/multiplication_numba_{size}x{size}_results.json
- results/multiplication_numba_{size}x{size}_results.csv
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'core_implementations'))
from sparse_multiplication_coo import sparse_multiply_coo


def generate_sparse_matrix(size, sparsity_percent, seed):
    """Generate sparse matrix with given sparsity."""
    np.random.seed(seed)
    total_elements = size * size
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(total_elements * density)
    
    rows = np.random.randint(0, size, size=num_entries)
    cols = np.random.randint(0, size, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)
    
    return rows, cols, values


def benchmark_multiplication(matrix_size, sparsity_percent, num_cores, num_runs=3):
    """
    Benchmark sparse COO multiplication vs dense NumPy multiplication.
    
    Args:
        matrix_size: Size of square matrix (size × size)
        sparsity_percent: Percentage of zero elements
        num_cores: Number of cores for Numba threading
        num_runs: Number of iterations to average
    """
    print(f"\n{'='*70}")
    print(f"Testing: {matrix_size}×{matrix_size} matrix, {sparsity_percent}% sparsity, {num_cores} cores")
    print(f"{'='*70}")
    
    print("Generating sparse matrices...")
    rows_A, cols_A, vals_A = generate_sparse_matrix(matrix_size, sparsity_percent, seed=42)
    rows_B, cols_B, vals_B = generate_sparse_matrix(matrix_size, sparsity_percent, seed=123)
    
    actual_nnz_A = len(rows_A)
    actual_nnz_B = len(rows_B)
    actual_sparsity_A = 100 * (1 - actual_nnz_A / (matrix_size * matrix_size))
    actual_sparsity_B = 100 * (1 - actual_nnz_B / (matrix_size * matrix_size))
    
    print(f"Matrix A: {actual_nnz_A:,} non-zeros ({actual_sparsity_A:.2f}% sparse)")
    print(f"Matrix B: {actual_nnz_B:,} non-zeros ({actual_sparsity_B:.2f}% sparse)")
    
    A_scipy = sp.coo_matrix((vals_A, (rows_A, cols_A)), shape=(matrix_size, matrix_size))
    B_scipy = sp.coo_matrix((vals_B, (rows_B, cols_B)), shape=(matrix_size, matrix_size))
    A_dense = A_scipy.toarray()
    B_dense = B_scipy.toarray()
    
    sparse_memory = (actual_nnz_A + actual_nnz_B) * (2 * 4 + 8)
    dense_memory = A_dense.nbytes + B_dense.nbytes
    print(f"\nMemory Usage:")
    print(f"  Sparse (COO): {sparse_memory / 1024**2:.2f} MB")
    print(f"  Dense (NumPy):    {dense_memory / 1024**2:.2f} MB")
    print(f"  Memory Ratio:     {dense_memory / sparse_memory:.2f}×")
    
    print(f"\nBenchmarking SPARSE (COO×COO, {num_cores}-core Numba threading)...")
    
    import numba
    numba.set_num_threads(num_cores)
    
    _ = sparse_multiply_coo(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, 
                            (matrix_size, matrix_size), (matrix_size, matrix_size))
    
    sparse_times = []
    for _ in tqdm(range(num_runs), desc=f"  Sparse COO×COO ({num_cores}-core Numba)", leave=False):
        start = time.perf_counter()
        result_rows, result_cols, result_vals = sparse_multiply_coo(
            rows_A, cols_A, vals_A, rows_B, cols_B, vals_B,
            (matrix_size, matrix_size), (matrix_size, matrix_size)
        )
        elapsed = time.perf_counter() - start
        sparse_times.append(elapsed)
    
    sparse_time = np.mean(sparse_times)
    sparse_std = np.std(sparse_times)
    result_nnz = len(result_vals)
    
    print(f"\n Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    print(f" Result non-zeros:  {result_nnz:,}")
    
    print(f"\nBenchmarking DENSE (NumPy, single-threaded baseline)...")
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    _ = np.dot(A_dense, B_dense)
    
    dense_times = []
    for _ in tqdm(range(num_runs), desc="  Dense NumPy", leave=False):
        start = time.perf_counter()
        C_dense = np.dot(A_dense, B_dense)
        elapsed = time.perf_counter() - start
        dense_times.append(elapsed)
    
    dense_time = np.mean(dense_times)
    dense_std = np.std(dense_times)
    
    print(f"\n Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1.0 else "Dense"
    
    print(f"\n{'='*70}")
    print(f"RESULT: {winner} wins with {speedup:.2f}× speedup")
    print(f"{'='*70}")
    
    return {
        'matrix_size': matrix_size,
        'sparsity_percent': sparsity_percent,
        'nnz_A': actual_nnz_A,
        'num_cores': num_cores,
        'sparse_time': sparse_time,
        'sparse_std': sparse_std,
        'dense_time': dense_time,
        'dense_std': dense_std,
        'speedup': speedup,
        'winner': winner,
        'result_nnz': result_nnz,
        'sparse_memory_mb': sparse_memory / 1024**2,
        'dense_memory_mb': dense_memory / 1024**2,
        'algorithm': f'COO×COO ({num_cores}-core Numba threading)'
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse vs dense matrix multiplication (Numba threading)')
    parser.add_argument('--size', type=int, default=1000, help='Matrix size (default: 1000)')
    parser.add_argument('--sparsity', type=float, nargs='+', default=[90.0, 99.0, 99.9],
                        help='Sparsity levels (default: 90.0 99.0 99.9)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per test (default: 3)')
    parser.add_argument('--num-cores', type=int, default=16, help='Number of cores (default: 16)')
    
    args = parser.parse_args()
    
    matrix_size = args.size
    sparsity_levels = args.sparsity if isinstance(args.sparsity, list) else [args.sparsity]
    num_runs = args.num_runs
    num_cores = args.num_cores
    
    print("\n" + "="*70)
    print("MULTIPLICATION BENCHMARK: Dense vs Sparse (Numba Threading)")
    print("="*70)
    print(f"\nAlgorithm: COO×COO (Coordinate Format)")
    print(f"Parallelism: {num_cores}-core Numba threading (set_num_threads)")
    print(f"Available CPU cores: {os.cpu_count()}")
    print("="*70)
    
    results = []
    for sparsity in tqdm(sparsity_levels, desc="Running Benchmarks", unit="test"):
        result = benchmark_multiplication(matrix_size, sparsity, num_cores, num_runs)
        results.append(result)
    
    print("\n\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['sparsity_percent']:.1f}%",
            f"{r['nnz_A']:,}",
            f"{r['num_cores']} cores",
            f"{r['sparse_time']:.6f}s",
            f"{r['dense_time']:.6f}s",
            f"{r['speedup']:.2f}×",
            r['winner'],
            f"{r['sparse_memory_mb']:.2f} MB",
            f"{r['dense_memory_mb']:.2f} MB"
        ])
    
    headers = ["Sparsity", "Non-Zeros", "Cores", "Sparse Time", "Dense Time", 
               "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    sparse_wins = sum(1 for r in results if r['winner'] == 'Sparse')
    dense_wins = sum(1 for r in results if r['winner'] == 'Dense')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Matrix Size: {matrix_size}×{matrix_size}")
    print(f"Algorithm: COO×COO ({num_cores}-core Numba threading)")
    print(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels")
    print(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if len(sparsity_levels) == 1:
        sparsity_str = f"sparsity{sparsity_levels[0]:.1f}pct".replace(".", "_")
    else:
        sparsity_list = "_".join([f"{s:.1f}pct".replace(".", "_") for s in sparsity_levels])
        sparsity_str = f"sparsity{sparsity_list}"
    
    base_name = f"multiplication_numba_{matrix_size}x{matrix_size}_{sparsity_str}_results"
    
    json_path = os.path.join(results_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    txt_path = os.path.join(results_dir, f"{base_name}.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("MULTIPLICATION BENCHMARK: Dense vs Sparse (Numba Threading)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Algorithm: COO×COO ({num_cores}-core Numba threading)\n")
        f.write(f"Matrix Size: {matrix_size}×{matrix_size}\n")
        f.write(f"Runs per test: {num_runs}\n\n")
        f.write(table + "\n\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Sparse wins: {sparse_wins}/{len(results)}\n")
        f.write(f"Dense wins:  {dense_wins}/{len(results)}\n")
    
    csv_path = os.path.join(results_dir, f"{base_name}.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sparsity%", "NonZeros", "Cores", "SparseTime(s)",
            "DenseTime(s)", "Speedup", "Winner", "Algorithm"
        ])
        for r in results:
            writer.writerow([
                r['sparsity_percent'], r['nnz_A'], r['num_cores'],
                f"{r['sparse_time']:.6f}",
                f"{r['dense_time']:.6f}", f"{r['speedup']:.2f}", r['winner'], r['algorithm']
            ])
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - {base_name}.json")
    print(f"  - {base_name}.txt")
    print(f"  - {base_name}.csv")


if __name__ == "__main__":
    main()
