"""
Dense vs Sparse Matrix Multiplication Benchmark (Multicore Parallel)

Compares performance of:
- Dense CPU (NumPy matrix multiplication - single-threaded)
- Sparse CPU (CSR×CSC multicore parallel algorithm using 16 cores)

Test Configuration:
- Matrix Size: Configurable (default: 1,000×1,000)
- Sparsity Levels: 90%, 99%, 99.9%
- Format: CSR×CSC (Compressed Sparse Row × Compressed Sparse Column)
- Parallelism: 16-core (AMD Ryzen 9 8940HX)
- Runs: 3 iterations per test (averaged)

Command-line Options:
  --size SIZE        Matrix size (default: 1000)
  --sparsity LEVEL   Sparsity level (default: 90)
  --num-runs N       Number of runs (default: 3)
  --num-cores N      Number of cores (default: 16)

Output:
- results/multiplication_parallel_{size}x{size}_results.txt
- results/multiplication_parallel_{size}x{size}_results.json
- results/multiplication_parallel_{size}x{size}_results.csv
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
import multiprocessing as mp
import tempfile
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'core_implementations'))
from sparse_multiplication_parallel import multiply_matrices_parallel
from matrix_formats import COOMatrix


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


def save_coo_to_csv(rows, cols, vals, filepath):
    """Save COO matrix to CSV file sorted by (row, col)."""
    entries = sorted(zip(rows, cols, vals), key=lambda x: (x[0], x[1]))
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row, col, val in entries:
            writer.writerow([row, col, val])


def load_coo_from_csv(filepath):
    """Load COO matrix from CSV file."""
    rows, cols, vals = [], [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 3:
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                vals.append(float(line[2]))
    return rows, cols, vals


def benchmark_sparse_parallel(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, size, num_cores, num_runs=3):
    """
    Benchmark sparse CSR×CSC multiplication with multicore parallelism using file-based API.
    
    Args:
        rows_A, cols_A, vals_A: Matrix A in COO format
        rows_B, cols_B, vals_B: Matrix B in COO format
        size: Matrix dimension
        num_cores: Number of CPU cores to use
        num_runs: Number of iterations to average
    
    Returns:
        avg_time_total, std_time_total, avg_time_compute, std_time_compute, result_nnz
    """
    times_total = []
    times_compute = []
    result_nnz = 0
    
    for _ in tqdm(range(num_runs), desc=f"  Sparse CSR×CSC ({num_cores}-core parallel)", leave=False):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f_a:
            temp_a = f_a.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f_b:
            temp_b = f_b.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f_out:
            temp_out = f_out.name
        
        try:
            start_total = time.perf_counter()
            
            save_coo_to_csv(rows_A, cols_A, vals_A, temp_a)
            save_coo_to_csv(rows_B, cols_B, vals_B, temp_b)
            
            start_compute = time.perf_counter()
            multiply_matrices_parallel(temp_a, temp_b, temp_out,
                                     shape_a=(size, size),
                                     shape_b=(size, size),
                                     num_workers=num_cores)
            end_compute = time.perf_counter()
            
            result_rows, result_cols, result_vals = load_coo_from_csv(temp_out)
            result_nnz = len(result_rows)
            
            end_total = time.perf_counter()
            
            times_total.append(end_total - start_total)
            times_compute.append(end_compute - start_compute)
            
        finally:
            for f in [temp_a, temp_b, temp_out]:
                if os.path.exists(f):
                    os.remove(f)
    
    return np.mean(times_total), np.std(times_total), np.mean(times_compute), np.std(times_compute), result_nnz


def benchmark_dense_multiplication(A_dense, B_dense, num_runs=3):
    """Benchmark dense NumPy matrix multiplication (single-threaded baseline)."""
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense NumPy (single-threaded)", leave=False):
        start = time.perf_counter()
        result = np.matmul(A_dense, B_dense)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result


def run_comparison(size, sparsity_percent, num_cores, num_runs=3):
    """Run comparison for one sparsity level with multicore sparse."""
    print(f"\n{'='*70}")
    print(f"Testing: {size}×{size} matrix, {sparsity_percent}% sparsity, {num_cores} cores")
    print(f"{'='*70}")
    
    print("Generating sparse matrices...")
    rows_A, cols_A, vals_A = generate_sparse_matrix(size, sparsity_percent, seed=42)
    rows_B, cols_B, vals_B = generate_sparse_matrix(size, sparsity_percent, seed=123)
    
    actual_nnz_A = len(rows_A)
    actual_nnz_B = len(rows_B)
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
    print(f"  Sparse (CSR+CSC): {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"  Dense (NumPy):    {dense_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory Ratio:     {dense_memory / sparse_memory:.2f}×")
    
    print(f"\nBenchmarking SPARSE (CSR×CSC, {num_cores}-core parallel)...")
    sparse_time_total, sparse_std_total, sparse_time_compute, sparse_std_compute, result_nnz = \
        benchmark_sparse_parallel(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, size, num_cores, num_runs)
    print(f"  Total (with I/O):  {sparse_time_total:.6f}s ± {sparse_std_total:.6f}s")
    print(f"  Compute only:      {sparse_time_compute:.6f}s ± {sparse_std_compute:.6f}s")
    print(f"  Result non-zeros:  {result_nnz:,}")
    
    print(f"\nBenchmarking DENSE (NumPy, single-threaded baseline)...")
    dense_time, dense_std, C_dense = benchmark_dense_multiplication(A_dense, B_dense, num_runs)
    print(f"  Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    speedup = dense_time / sparse_time_compute
    winner = "Sparse" if speedup > 1 else "Dense"
    
    print(f"\n{'='*70}")
    print(f"RESULT: {winner} wins with {abs(speedup):.2f}× speedup (compute-only)")
    print(f"{'='*70}")
    
    return {
        "sparsity_percent": sparsity_percent,
        "actual_sparsity_A": actual_sparsity_A,
        "actual_sparsity_B": actual_sparsity_B,
        "matrix_size": size,
        "nnz_A": actual_nnz_A,
        "nnz_B": actual_nnz_B,
        "result_nnz": result_nnz,
        "num_cores": num_cores,
        "sparse_time_total": sparse_time_total,
        "sparse_std_total": sparse_std_total,
        "sparse_time_compute": sparse_time_compute,
        "sparse_std_compute": sparse_std_compute,
        "dense_time": dense_time,
        "dense_std": dense_std,
        "speedup": speedup,
        "winner": winner,
        "memory_sparse_mb": sparse_memory / 1024 / 1024,
        "memory_dense_mb": dense_memory / 1024 / 1024,
        "memory_ratio": dense_memory / sparse_memory,
        "algorithm": f"CSR×CSC ({num_cores}-core parallel)",
        "num_runs": num_runs
    }


def main():
    parser = argparse.ArgumentParser(description='Dense vs Sparse Matrix Multiplication Benchmark (Parallel)')
    parser.add_argument('--size', type=int, default=1000, help='Matrix size (default: 1000)')
    parser.add_argument('--sparsity', type=float, default=None, help='Single sparsity level to test (default: test all)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs to average (default: 3)')
    parser.add_argument('--num-cores', type=int, default=None, help='Number of cores to use (default: auto-detect, max 16)')
    args = parser.parse_args()
    
    print("="*70)
    print("MULTIPLICATION BENCHMARK: Dense vs Sparse (Multicore Parallel)")
    print("="*70)
    
    available_cores = mp.cpu_count()
    num_cores = args.num_cores if args.num_cores else min(16, available_cores)
    
    print(f"\nAlgorithm: CSR×CSC (Compressed Sparse Row × Compressed Sparse Column)")
    print(f"Parallelism: {num_cores}-core multiprocessing")
    print(f"Available CPU cores: {available_cores}")
    print("="*70)
    
    matrix_size = args.size
    sparsity_levels = [args.sparsity] if args.sparsity else [90, 99, 99.9]
    num_runs = args.num_runs
    
    results = []
    for sparsity in tqdm(sparsity_levels, desc="Running Benchmarks", unit="test"):
        result = run_comparison(matrix_size, sparsity, num_cores, num_runs)
        results.append(result)
    
    print("\n\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['sparsity_percent']}%",
            f"{r['nnz_A']:,}",
            f"{r['num_cores']} cores",
            f"{r['sparse_time_compute']:.6f}s",
            f"{r['dense_time']:.6f}s",
            f"{r['speedup']:.2f}×",
            r['winner'],
            f"{r['memory_sparse_mb']:.2f} MB",
            f"{r['memory_dense_mb']:.2f} MB"
        ])
    
    headers = ["Sparsity", "Non-Zeros", "Cores", "Sparse Time", "Dense Time", "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    sparse_wins = sum(1 for r in results if r['winner'] == 'Sparse')
    dense_wins = sum(1 for r in results if r['winner'] == 'Dense')
    
    print(f"Matrix Size: {matrix_size}×{matrix_size}")
    print(f"Algorithm: CSR×CSC ({num_cores}-core parallel)")
    print(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels")
    print(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = f"multiplication_parallel_{matrix_size}x{matrix_size}_results"
    
    json_path = os.path.join(results_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    txt_path = os.path.join(results_dir, f"{base_name}.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("MULTIPLICATION BENCHMARK: Dense vs Sparse (Multicore Parallel)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Algorithm: CSR×CSC ({num_cores}-core parallel)\n")
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
            "Sparsity%", "NonZeros", "Cores", "SparseTimeCompute(s)", "SparseTimeTotal(s)",
            "DenseTime(s)", "Speedup", "Winner", "Algorithm"
        ])
        for r in results:
            writer.writerow([
                r['sparsity_percent'], r['nnz_A'], r['num_cores'],
                f"{r['sparse_time_compute']:.6f}", f"{r['sparse_time_total']:.6f}",
                f"{r['dense_time']:.6f}", f"{r['speedup']:.2f}", r['winner'], r['algorithm']
            ])
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - {base_name}.json")
    print(f"  - {base_name}.txt")
    print(f"  - multiplication_parallel_1000x1000_results.csv")


if __name__ == "__main__":
    main()
