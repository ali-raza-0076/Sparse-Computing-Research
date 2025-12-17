"""
COO Matrix Addition Benchmark (Single-Threaded, Database I/O Workflow)

Database I/O Workflow:
- Phase 1: Read from secondary storage (CSV files)
- Phase 2: Process in RAM (COO addition algorithm)
- Phase 3: Write to secondary storage (CSV result)

Features:
- Format: Pure COO (coordinate list: i,j,v triplets)
- Parallelism: Single-threaded (no multicore)
- Metrics: I/O time breakdown, compute time, throughput, overhead percentages

Output:
- results/addition_coo_<dims>_<nnz>.csv (result matrix)
- results/metrics_addition.json (detailed performance metrics)
"""
import numpy as np
import time
import csv
import sys
import os
import argparse
import json
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
import importlib.util
core_impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core_implementations/sparse_addition_coo.py'))
spec = importlib.util.spec_from_file_location('sparse_addition_coo', core_impl_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec for sparse_addition_coo from {core_impl_path}")
sparse_addition_coo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_addition_coo)
COOMatrix = sparse_addition_coo.COOMatrix
sparse_add_coo = sparse_addition_coo.sparse_add_coo


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
    Benchmark sparse COO×COO addition (single-threaded).
    
    Args:
        coo_A, coo_B: COOMatrix objects
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result: Average time, std deviation, result matrix
    """
    _ = sparse_add_coo(coo_A, coo_B)
    
    times = []
    for _ in tqdm(range(num_runs), desc="  Sparse COO×COO (single-threaded)", leave=False):
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


def read_coo_csv(filepath):
    """Read a COO matrix from a CSV file with i,j,v columns."""
    data = []
    max_i = 0
    max_j = 0
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            i, j, v = int(row[0]), int(row[1]), float(row[2])
            data.append((i, j, v))
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j
    return data, max_i + 1, max_j + 1

def run_comparison_from_files(file_a, file_b, num_runs=3):
    """
    Run addition using two input CSV files in i,j,v format.
    Records detailed I/O and computation metrics following database I/O workflow:
    Phase 1: Read from secondary storage → Phase 2: Process in RAM → Phase 3: Write to secondary storage
    """
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: COO Addition Benchmark")
    print(f"{'='*70}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    data_A, nrows_A, ncols_A = read_coo_csv(file_a)
    data_B, nrows_B, ncols_B = read_coo_csv(file_b)
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    shape = (max(nrows_A, nrows_B), max(ncols_A, ncols_B))
    print(f"  Matrix A: {file_a}")
    print(f"    Shape: {nrows_A}×{ncols_A}, {len(data_A):,} non-zeros")
    print(f"  Matrix B: {file_b}")
    print(f"    Shape: {nrows_B}×{ncols_B}, {len(data_B):,} non-zeros")
    print(f"  Result shape: {shape}")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    print(f"\n[PHASE 2] Processing in RAM...")
    compute_start = time.perf_counter()
    data_A.sort(key=lambda x: (x[0], x[1]))
    data_B.sort(key=lambda x: (x[0], x[1]))
    coo_A = COOMatrix(shape=shape, data=data_A)
    coo_B = COOMatrix(shape=shape, data=data_B)
    sparse_time, sparse_std, C_sparse = benchmark_sparse_addition(coo_A, coo_B, num_runs)
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    
    print(f"  Computation Time: {compute_time:.6f}s (avg: {sparse_time:.6f}s ± {sparse_std:.6f}s)")
    print(f"  Result: {len(C_sparse.data):,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"addition_coo_{shape[0]}x{shape[1]}_{len(data_A)}A_{len(data_B)}B.csv"
    out_path = os.path.join(results_dir, out_name)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in C_sparse.data:
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
    print(f"Throughput:       {(len(data_A) + len(data_B) + len(C_sparse.data))/total_time:,.0f} entries/sec")
    print(f"{'='*70}")
    
    metrics = {
        "operation": "addition",
        "execution_mode": "single_threaded",
        "matrix_a": {
            "file": file_a,
            "shape": [nrows_A, ncols_A],
            "nnz": len(data_A)
        },
        "matrix_b": {
            "file": file_b,
            "shape": [nrows_B, ncols_B],
            "nnz": len(data_B)
        },
        "result": {
            "shape": list(shape),
            "nnz": len(C_sparse.data),
            "file": out_path
        },
        "timing": {
            "phase1_read_time_sec": read_time,
            "phase2_compute_time_sec": compute_time,
            "phase3_write_time_sec": write_time,
            "total_time_sec": total_time,
            "io_overhead_sec": read_time + write_time,
            "compute_avg_sec": sparse_time,
            "compute_std_sec": sparse_std
        },
        "percentages": {
            "read": round(read_time/total_time*100, 2),
            "compute": round(compute_time/total_time*100, 2),
            "write": round(write_time/total_time*100, 2),
            "io_overhead": round((read_time + write_time)/total_time*100, 2)
        },
        "throughput_entries_per_sec": round((len(data_A) + len(data_B) + len(C_sparse.data))/total_time, 2),
        "num_runs": num_runs
    }
    
    metrics_path = os.path.join(results_dir, "metrics_addition.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return out_path


def main():

    parser = argparse.ArgumentParser(description='COO addition: add two input CSVs (i,j,v) and write result to results/.')
    parser.add_argument('--input_a', type=str, default='input/matrix_a_5k.csv', help='Path to matrix_a.csv')
    parser.add_argument('--input_b', type=str, default='input/matrix_b_5k.csv', help='Path to matrix_b.csv')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs (default: 1)')
    args = parser.parse_args()
    run_comparison_from_files(args.input_a, args.input_b, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
