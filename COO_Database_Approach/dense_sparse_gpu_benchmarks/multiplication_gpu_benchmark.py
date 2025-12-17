"""
COO GPU Multiplication Benchmark - Database I/O Workflow

Benchmarks sparse matrix multiplication on GPU using PyTorch with database I/O:
- Phase 1: Read from disk (CSV)
- Phase 2: Transfer to GPU and compute
- Phase 3: Transfer back and write to disk
"""
import numpy as np
import torch
import time
import csv
import os
import argparse
import json


def read_coo_csv(filepath):
    """Read COO matrix from CSV (i,j,v format)."""
    data = []
    max_i, max_j = 0, 0
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
                data.append((i, j, v))
                max_i, max_j = max(max_i, i), max(max_j, j)
    return data, (max_i + 1, max_j + 1)


def write_coo_csv(filepath, rows, cols, vals):
    """Write COO matrix to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in zip(rows, cols, vals):
            writer.writerow([i, j, v])


def run_gpu_benchmark(input_a, input_b, num_runs=1):
    """Run GPU multiplication benchmark with database I/O workflow."""
    
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: COO GPU Multiplication Benchmark")
    print(f"{'='*70}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    data_A, shape_A = read_coo_csv(input_a)
    data_B, shape_B = read_coo_csv(input_b)
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    nnz_A = len(data_A)
    nnz_B = len(data_B)
    
    print(f"  Matrix A: {input_a}")
    print(f"    Shape: {shape_A}, {nnz_A:,} non-zeros")
    print(f"  Matrix B: {input_b}")
    print(f"    Shape: {shape_B}, {nnz_B:,} non-zeros")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    if shape_B[1] != shape_A[0]:
        raise ValueError(f"Incompatible dimensions: B is {shape_B}, A is {shape_A}")
    
    print(f"  Computing B×A (compatible dimensions)")
    
    print(f"\n[PHASE 2] Processing on GPU...")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This benchmark requires a GPU. Run on Google Colab with T4 GPU.")
    
    device = torch.device("cuda")
    print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Device: {device}")
    
    compute_start = time.perf_counter()
    
    rows_A = np.array([d[0] for d in data_A], dtype=np.int64)
    cols_A = np.array([d[1] for d in data_A], dtype=np.int64)
    vals_A = np.array([d[2] for d in data_A], dtype=np.float32)
    
    rows_B = np.array([d[0] for d in data_B], dtype=np.int64)
    cols_B = np.array([d[1] for d in data_B], dtype=np.int64)
    vals_B = np.array([d[2] for d in data_B], dtype=np.float32)
    
    times = []
    for _ in range(num_runs):
        run_start = time.perf_counter()
        
        indices_A = torch.from_numpy(np.vstack([rows_A, cols_A])).to(device)
        values_A = torch.from_numpy(vals_A).to(device)
        sparse_A = torch.sparse_coo_tensor(indices_A, values_A, shape_A, device=device)
        
        indices_B = torch.from_numpy(np.vstack([rows_B, cols_B])).to(device)
        values_B = torch.from_numpy(vals_B).to(device)
        sparse_B = torch.sparse_coo_tensor(indices_B, values_B, shape_B, device=device)
        
        result_sparse = torch.sparse.mm(sparse_B, sparse_A)
        result_sparse = result_sparse.coalesce()
        
        indices_result = result_sparse.indices().cpu().numpy()
        values_result = result_sparse.values().cpu().numpy()
        rows_result = indices_result[0]
        cols_result = indices_result[1]
        vals_result = values_result
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    result_shape = (shape_B[0], shape_A[1])
    nnz_result = len(vals_result)
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result shape: {result_shape}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"multiplication_gpu_coo_BxA_{result_shape[0]}x{result_shape[1]}.csv"
    out_path = os.path.join(results_dir, out_name)
    
    write_coo_csv(out_path, rows_result, cols_result, vals_result)
    
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    print(f"  I/O Write Time: {write_time:.6f}s")
    print(f"  Result written to: {out_path}")
    
    total_time = read_time + compute_time + write_time
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):    {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (GPU):     {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write):   {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {read_time + write_time:.6f}s ({(read_time + write_time)/total_time*100:.2f}%)")
    print(f"Throughput:       {(nnz_A + nnz_B + nnz_result)/total_time:,.0f} entries/sec")
    print(f"{'='*70}")
    
    metrics = {
        "operation": "multiplication",
        "device": "GPU" if torch.cuda.is_available() else "CPU_fallback",
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "matrix_a": {"file": input_a, "shape": list(shape_A), "nnz": nnz_A},
        "matrix_b": {"file": input_b, "shape": list(shape_B), "nnz": nnz_B},
        "result": {"operation": "B×A", "shape": list(result_shape), "nnz": nnz_result, "file": out_path},
        "timing": {
            "phase1_read_time_sec": read_time,
            "phase2_gpu_time_sec": compute_time,
            "phase3_write_time_sec": write_time,
            "total_time_sec": total_time,
            "io_overhead_sec": read_time + write_time,
            "compute_avg_sec": avg_time,
            "compute_std_sec": std_time
        },
        "percentages": {
            "read": round(read_time/total_time*100, 2),
            "gpu_compute": round(compute_time/total_time*100, 2),
            "write": round(write_time/total_time*100, 2),
            "io_overhead": round((read_time + write_time)/total_time*100, 2)
        },
        "throughput_entries_per_sec": round((nnz_A + nnz_B + nnz_result)/total_time, 2),
        "num_runs": num_runs
    }
    
    metrics_path = os.path.join(results_dir, "metrics_multiplication_gpu.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='COO GPU multiplication benchmark')
    parser.add_argument('--input_a', type=str, default='../../input/matrix_a_5k.csv')
    parser.add_argument('--input_b', type=str, default='../../input/matrix_b_5k.csv')
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    
    run_gpu_benchmark(args.input_a, args.input_b, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
