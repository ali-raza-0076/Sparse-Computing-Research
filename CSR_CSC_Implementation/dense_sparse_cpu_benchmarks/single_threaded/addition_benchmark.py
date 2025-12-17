"""
Dense vs Sparse Matrix Addition Benchmark (Single-Threaded)

Compares performance of:
- Dense CPU (NumPy array addition)
- Sparse CPU (CSR+CSR single-threaded algorithm)

Test Configuration:
- Matrix Size: 1000×1000
- Sparsity Levels: 90%, 99%, 99.9%
- Format: CSR (Compressed Sparse Row)
- Parallelism: Single-threaded (no multicore)
- Runs: 3 iterations per test (averaged)

Output:
- results/addition_1000x1000_results.txt (human-readable table)
- results/addition_1000x1000_results.json (structured data)
- results/addition_1000x1000_results.csv (spreadsheet format)
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
from sparse_addition_csr import sparse_add_coo
from matrix_formats import COOMatrix


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


def benchmark_sparse_addition(coo_A, coo_B, num_runs=3):
    """
    Benchmark sparse CSR addition (single-threaded).
    
    Args:
        coo_A, coo_B: COOMatrix objects
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result: Average time, std deviation, result matrix
    """
    _ = sparse_add_coo(coo_A, coo_B)
    
    times = []
    for _ in tqdm(range(num_runs), desc="  Sparse CSR (single-threaded)", leave=False):
        start = time.perf_counter()
        result = sparse_add_coo(coo_A, coo_B)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result


def benchmark_dense_addition(A_dense, B_dense, num_runs=3):
    """
    Benchmark dense NumPy array addition.
    
    Args:
        A_dense, B_dense: NumPy dense arrays
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result: Average time, std deviation, result matrix
    """
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense NumPy (+)", leave=False):
        start = time.perf_counter()
        result = A_dense + B_dense
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
    print(f"  Sparse (CSR):  {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"  Dense (NumPy): {dense_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory Ratio:  {dense_memory / sparse_memory:.2f}×")
    
    print(f"\nBenchmarking SPARSE (CSR, single-threaded)...")
    sparse_time, sparse_std, C_sparse = benchmark_sparse_addition(coo_A, coo_B, num_runs)
    print(f"  Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    
    print(f"\nBenchmarking DENSE (NumPy +)...")
    dense_time, dense_std, C_dense = benchmark_dense_addition(A_dense, B_dense, num_runs)
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
        "algorithm": "CSR (single-threaded)",
        "num_runs": num_runs
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse vs dense addition (single-threaded)')
    parser.add_argument('--size', type=int, default=1000, help='Matrix size (default: 1000)')
    parser.add_argument('--sparsity', type=float, default=None, help='Single sparsity level to test (e.g., 99 or 99.9)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per test (default: 3)')
    args = parser.parse_args()
    
    print("="*70)
    print("ADDITION BENCHMARK: Dense vs Sparse (Single-Threaded)")
    print("="*70)
    print("\nAlgorithm: CSR (Compressed Sparse Row)")
    print("Parallelism: Single-threaded (no multicore)")
    print("="*70)
    
    matrix_size = args.size
    sparsity_levels = [args.sparsity] if args.sparsity is not None else [90, 99, 99.9]
    num_runs = args.num_runs
    
    results = []
    for sparsity in tqdm(sparsity_levels, desc="Running Benchmarks", unit="test"):
        result = run_comparison(matrix_size, sparsity, num_runs)
        results.append(result)
    
    print("\n\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['sparsity_percent']}%",
            f"{r['nnz_A']:,}",
            f"{r['sparse_time']:.6f}s",
            f"{r['dense_time']:.6f}s",
            f"{r['speedup']:.2f}×",
            r['winner'],
            f"{r['memory_sparse_mb']:.2f} MB",
            f"{r['memory_dense_mb']:.2f} MB"
        ])
    
    headers = ["Sparsity", "Non-Zeros", "Sparse Time", "Dense Time", "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    sparse_wins = sum(1 for r in results if r['winner'] == 'Sparse')
    dense_wins = sum(1 for r in results if r['winner'] == 'Dense')
    
    print(f"Matrix Size: {matrix_size}×{matrix_size}")
    print(f"Algorithm: CSR (single-threaded)")
    print(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels")
    print(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels")
    print()
    
    sparse_winning_sparsities = [r['sparsity_percent'] for r in results if r['winner'] == 'Sparse']
    dense_winning_sparsities = [r['sparsity_percent'] for r in results if r['winner'] == 'Dense']
    
    if sparse_winning_sparsities:
        print(f"Sparse wins at: {', '.join(map(str, sparse_winning_sparsities))}% sparsity")
    if dense_winning_sparsities:
        print(f"Dense wins at:  {', '.join(map(str, dense_winning_sparsities))}% sparsity")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if len(sparsity_levels) == 1:
        sparsity_str = f"{sparsity_levels[0]}pct"
    else:
        sparsity_str = "multi"
    
    json_path = os.path.join(results_dir, f"addition_{matrix_size}x{matrix_size}_{sparsity_str}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    txt_path = os.path.join(results_dir, f"addition_{matrix_size}x{matrix_size}_{sparsity_str}_results.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("ADDITION BENCHMARK: Dense vs Sparse (Single-Threaded)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Algorithm: CSR (single-threaded)\n")
        f.write(f"Matrix Size: {matrix_size}×{matrix_size}\n")
        f.write(f"Runs per test: {num_runs}\n\n")
        f.write(table + "\n\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels\n")
        f.write(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels\n\n")
        if sparse_winning_sparsities:
            f.write(f"Sparse wins at: {', '.join(map(str, sparse_winning_sparsities))}% sparsity\n")
        if dense_winning_sparsities:
            f.write(f"Dense wins at:  {', '.join(map(str, dense_winning_sparsities))}% sparsity\n")
    
    csv_path = os.path.join(results_dir, f"addition_{matrix_size}x{matrix_size}_{sparsity_str}_results.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Sparsity%", "NonZeros", "SparseTime(s)", "DenseTime(s)", 
            "Speedup", "Winner", "SparseMem(MB)", "DenseMem(MB)", "Algorithm"
        ])
        for r in results:
            writer.writerow([
                r['sparsity_percent'],
                r['nnz_A'],
                f"{r['sparse_time']:.6f}",
                f"{r['dense_time']:.6f}",
                f"{r['speedup']:.2f}",
                r['winner'],
                f"{r['memory_sparse_mb']:.2f}",
                f"{r['memory_dense_mb']:.2f}",
                r['algorithm']
            ])
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - addition_{matrix_size}x{matrix_size}_results.json")
    print(f"  - addition_{matrix_size}x{matrix_size}_results.txt")
    print(f"  - addition_{matrix_size}x{matrix_size}_results.csv")


if __name__ == "__main__":
    main()
