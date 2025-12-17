"""
Dense Matrix Multiplication Benchmark (Multicore Parallel, Database I/O Workflow)

Database I/O Workflow:
- Phase 1: Read from secondary storage (CSV files in COO format)
- Phase 2: Process in RAM (Dense multiplication using NumPy with multicore BLAS)
- Phase 3: Write to secondary storage (CSV result in COO format)

Features:
- Format: Dense (stored as COO in CSV, converted to dense for computation)
- Parallelism: Multicore (NumPy with OpenBLAS/MKL)
- Metrics: I/O time breakdown, compute time, throughput, overhead percentages

Output:
- results/multiplication_dense_parallel_<dims>_<cores>cores_result.csv (result matrix)
- results/metrics_multiplication_dense_parallel.json (detailed performance metrics)
"""
import numpy as np
import time
import csv
import os
import argparse
import json
from multiprocessing import cpu_count


def read_coo_to_dense(filepath):
    """Read COO format CSV and convert to dense NumPy array."""
    data = []
    max_i = 0
    max_j = 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
                data.append((i, j, v))
                if i > max_i:
                    max_i = i
                if j > max_j:
                    max_j = j
    
    shape = (max_i + 1, max_j + 1)
    dense = np.zeros(shape, dtype=np.float64)
    for i, j, v in data:
        dense[i, j] = v
    
    return dense, shape, len(data)


def write_dense_to_coo(filepath, dense_array):
    """Write dense array to COO format CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        rows, cols = dense_array.shape
        count = 0
        for i in range(rows):
            for j in range(cols):
                v = dense_array[i, j]
                if v != 0:
                    writer.writerow([i, j, v])
                    count += 1
    return count


def dense_multiplication_parallel(A, B, num_runs=3):
    """
    Benchmark dense matrix multiplication using NumPy (inherently parallel with BLAS).
    
    Args:
        A, B: Dense NumPy arrays
        num_runs: Number of iterations to average
    
    Returns:
        avg_time, std_time, result
    """
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = A @ B
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result


def run_parallel_benchmark(input_a, input_b, num_workers=None, num_runs=1):
    """
    Run parallel dense multiplication benchmark with database I/O workflow.
    Records detailed I/O and computation metrics following database I/O workflow:
    Phase 1: Read from secondary storage → Phase 2: Process in RAM → Phase 3: Write to secondary storage
    
    Note: NumPy automatically uses multicore BLAS (OpenBLAS/MKL) for dense operations.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: Dense Multiplication Benchmark (Multicore, {num_workers} cores)")
    print(f"{'='*70}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    A_dense, shape_A, nnz_A = read_coo_to_dense(input_a)
    B_dense, shape_B, nnz_B = read_coo_to_dense(input_b)
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    print(f"  Matrix A: {input_a}")
    print(f"    Shape: {shape_A[0]}×{shape_A[1]}, {nnz_A:,} non-zeros in COO")
    print(f"  Matrix B: {input_b}")
    print(f"    Shape: {shape_B[0]}×{shape_B[1]}, {nnz_B:,} non-zeros in COO")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    if A_dense.shape[1] != B_dense.shape[0]:
        raise ValueError(f"Incompatible dimensions for A×B: A is {A_dense.shape}, B is {B_dense.shape}. Need A.cols == B.rows")
    
    print(f"  Computing A×B (compatible dimensions)")
    
    print(f"\n[PHASE 2] Processing in RAM (Dense NumPy with BLAS parallelism)...")
    compute_start = time.perf_counter()
    avg_time, std_time, result = dense_multiplication_parallel(A_dense, B_dense, num_runs=num_runs)
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    
    nnz_result = np.count_nonzero(result)
    
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result shape: {result.shape[0]}×{result.shape[1]}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"multiplication_dense_parallel_{result.shape[0]}x{result.shape[1]}_{num_workers}cores_result.csv"
    out_path = os.path.join(results_dir, out_name)
    
    written_entries = write_dense_to_coo(out_path, result)
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    print(f"  I/O Write Time: {write_time:.6f}s")
    print(f"  Result written to: {out_path} ({written_entries:,} entries)")
    
    total_time = read_time + compute_time + write_time
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):  {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (Compute): {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write):   {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {read_time + write_time:.6f}s ({(read_time + write_time)/total_time*100:.2f}%)")
    print(f"Throughput:       {(nnz_A + nnz_B + written_entries)/total_time:,.0f} entries/sec")
    print(f"{'='*70}")
    
    metrics = {
        "operation": "multiplication",
        "execution_mode": "multicore_parallel_dense",
        "num_workers": num_workers,
        "matrix_a": {
            "file": input_a,
            "shape": list(shape_A),
            "nnz": nnz_A,
            "total_elements": shape_A[0] * shape_A[1]
        },
        "matrix_b": {
            "file": input_b,
            "shape": list(shape_B),
            "nnz": nnz_B,
            "total_elements": shape_B[0] * shape_B[1]
        },
        "result": {
            "shape": [result.shape[0], result.shape[1]],
            "nnz": int(nnz_result),
            "total_elements": result.shape[0] * result.shape[1],
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
        "throughput_entries_per_sec": round((nnz_A + nnz_B + written_entries)/total_time, 2),
        "num_runs": num_runs
    }
    
    metrics_path = os.path.join(results_dir, "metrics_multiplication_dense_parallel.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Dense parallel matrix multiplication using NumPy BLAS')
    parser.add_argument('--input_a', type=str, default='input/dense_matrix_a_300x300.csv', help='Path to dense matrix A')
    parser.add_argument('--input_b', type=str, default='input/dense_matrix_b_300x300.csv', help='Path to dense matrix B')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of cores (default: CPU count)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs (default: 1)')
    args = parser.parse_args()
    
    run_parallel_benchmark(args.input_a, args.input_b, num_workers=args.num_workers, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
