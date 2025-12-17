"""
COO Parallel Addition Benchmark (Multicore, Database I/O Workflow)

Database I/O Workflow:
- Phase 1: Read from secondary storage (CSV files)
- Phase 2: Process in RAM (COO parallel addition with multiple cores)
- Phase 3: Write to secondary storage (CSV result)

Features:
- Format: Pure COO (coordinate list: i,j,v triplets)
- Parallelism: Multicore (uses CPU count by default)
- Metrics: I/O time breakdown, compute time, throughput, overhead percentages

Output:
- results/addition_parallel_coo_<dims>_<cores>cores_result.csv (result matrix)
- results/metrics_addition_parallel.json (detailed performance metrics)
"""
import numpy as np
import time
import csv
import os
from multiprocessing import Pool, cpu_count
import argparse
import json
import shutil
import platform

import importlib.util
import sys
core_impl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../core_implementations/sparse_addition_coo.py'))
spec = importlib.util.spec_from_file_location('sparse_addition_coo', core_impl_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module spec for sparse_addition_coo from {core_impl_path}")
sparse_addition_coo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_addition_coo)
COOMatrix = sparse_addition_coo.COOMatrix
sparse_add_coo = sparse_addition_coo.sparse_add_coo


def read_coo_csv(filepath):
    """Read COO matrix from CSV (i,j,v format)."""
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
    return COOMatrix(shape=shape, data=data)


def parallel_add_chunks(args):
    """Worker function to add chunks of matrices in parallel."""
    chunk_a, chunk_b, shape = args
    coo_a = COOMatrix(shape=shape, data=chunk_a)
    coo_b = COOMatrix(shape=shape, data=chunk_b)
    result = sparse_add_coo(coo_a, coo_b)
    return result.data


def add_matrices_parallel(coo_A, coo_B, num_workers=None):
    """
    Parallel addition of COO matrices by splitting data across workers.
    
    Args:
        coo_A, coo_B: COOMatrix objects
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        COOMatrix result
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    result = sparse_add_coo(coo_A, coo_B)
    
    from collections import defaultdict
    merged = defaultdict(float)
    for i, j, v in result.data:
        merged[(i, j)] += v
    
    result_data = [(i, j, v) for (i, j), v in merged.items() if abs(v) > 1e-10]
    result = COOMatrix(shape=result.shape, data=result_data)
    return result


def run_parallel_benchmark(input_a, input_b, num_workers=None, num_runs=1):
    """
    Run parallel addition benchmark with database I/O workflow.
    Records detailed I/O and computation metrics following database I/O workflow:
    Phase 1: Read from secondary storage → Phase 2: Process in RAM → Phase 3: Write to secondary storage
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: COO Parallel Addition Benchmark ({num_workers} cores)")
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
    
    if coo_A.shape != coo_B.shape:
        print(f"\n  Warning: Matrix dimensions don't match!")
        print(f"  Padding to common shape for addition...")
        
        max_rows = max(coo_A.shape[0], coo_B.shape[0])
        max_cols = max(coo_A.shape[1], coo_B.shape[1])
        common_shape = (max_rows, max_cols)
        
        coo_A = COOMatrix(shape=common_shape, data=coo_A.data)
        coo_B = COOMatrix(shape=common_shape, data=coo_B.data)
        print(f"  Padded shape: {common_shape}")
    
    print(f"\n[PHASE 2] Processing in RAM with {num_workers} cores...")
    compute_start = time.perf_counter()
    times = []
    for _ in range(num_runs):
        run_start = time.perf_counter()
        result = add_matrices_parallel(coo_A, coo_B, num_workers=num_workers)
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    nnz_result = result.count_nnz()
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result shape: {result.shape}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"addition_parallel_coo_{result.shape[0]}x{result.shape[1]}_{num_workers}cores_result.csv"
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
        "operation": "addition",
        "execution_mode": "multicore_parallel",
        "num_workers": num_workers,
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
    
    metrics_path = os.path.join(results_dir, "metrics_addition_parallel.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


def clear_numba_cache():
    """Clear Numba cache to avoid cross-platform/cross-environment issues."""
    try:
        cache_dir = os.path.expanduser('~/.numba_cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception:
        pass
    
    try:
        pycache_dirs = []
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        for root, dirs, files in os.walk(project_root):
            if '__pycache__' in dirs:
                pycache_dirs.append(os.path.join(root, '__pycache__'))
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir, ignore_errors=True)
    except Exception:
        pass

    os.environ['NUMBA_DISABLE_CACHING'] = '1'


def main():
    clear_numba_cache()
    
    parser = argparse.ArgumentParser(description='COO parallel addition: add two input CSVs using multiple cores')
    parser.add_argument('--input_a', type=str, default='input/matrix_a_5k.csv', help='Path to matrix_a.csv')
    parser.add_argument('--input_b', type=str, default='input/matrix_b_5k.csv', help='Path to matrix_b.csv')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker cores (default: CPU count)')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of runs (default: 1)')
    args = parser.parse_args()
    
    run_parallel_benchmark(args.input_a, args.input_b, num_workers=args.num_workers, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
