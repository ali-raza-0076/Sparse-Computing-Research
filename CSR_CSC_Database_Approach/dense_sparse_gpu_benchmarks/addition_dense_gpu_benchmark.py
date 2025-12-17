"""
Dense Matrix Addition GPU Benchmark - Database I/O Workflow

Benchmarks dense matrix addition on GPU using PyTorch with database I/O:
- Phase 1: Read from disk (CSV in COO format)
- Phase 2: Convert to dense, transfer to GPU and compute
- Phase 3: Transfer back and write to disk (COO format)
"""
import numpy as np
import torch
import time
import csv
import os
import argparse
import json


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
    dense = np.zeros(shape, dtype=np.float32)
    for i, j, v in data:
        dense[i, j] = v
    
    return dense, shape, len(data)


def write_dense_to_coo(filepath, dense_array):
    """Write dense array to COO format CSV (only non-zeros)."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        rows, cols = dense_array.shape
        count = 0
        for i in range(rows):
            for j in range(cols):
                v = dense_array[i, j]
                if abs(v) > 1e-10:
                    writer.writerow([i, j, float(v)])
                    count += 1
    return count


def run_gpu_benchmark(input_a, input_b, num_runs=1):
    """Run GPU dense addition benchmark with database I/O workflow."""
    
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: Dense GPU Addition Benchmark")
    print(f"{'='*70}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU. "
                         "In Google Colab, go to Runtime > Change runtime type > Select T4 GPU.")
    
    device = torch.device("cuda:0")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    
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
    
    if A_dense.shape != B_dense.shape:
        max_rows = max(A_dense.shape[0], B_dense.shape[0])
        max_cols = max(A_dense.shape[1], B_dense.shape[1])
        
        A_padded = np.zeros((max_rows, max_cols), dtype=np.float32)
        B_padded = np.zeros((max_rows, max_cols), dtype=np.float32)
        
        A_padded[:A_dense.shape[0], :A_dense.shape[1]] = A_dense
        B_padded[:B_dense.shape[0], :B_dense.shape[1]] = B_dense
        
        A_dense = A_padded
        B_dense = B_padded
        print(f"  Padded to common shape: {max_rows}×{max_cols}")
    
    print(f"\n[PHASE 2] Processing on GPU...")
    print(f"  Using device: {device}")
    
    compute_start = time.perf_counter()
    
    times = []
    for _ in range(num_runs):
        run_start = time.perf_counter()
        
        A_tensor = torch.from_numpy(A_dense).to(device)
        B_tensor = torch.from_numpy(B_dense).to(device)
        
        result_tensor = A_tensor + B_tensor
        
        torch.cuda.synchronize()
        
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    
    result_dense = result_tensor.cpu().numpy()
    
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    nnz_result = np.count_nonzero(result_dense)
    
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result shape: {result_dense.shape[0]}×{result_dense.shape[1]}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_name = f"addition_dense_gpu_{result_dense.shape[0]}x{result_dense.shape[1]}_result.csv"
    out_path = os.path.join(results_dir, out_name)
    
    written_entries = write_dense_to_coo(out_path, result_dense)
    
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    print(f"  I/O Write Time: {write_time:.6f}s")
    print(f"  Result written to: {out_path} ({written_entries:,} entries)")
    
    total_time = read_time + compute_time + write_time
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):    {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (GPU):     {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write):   {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {read_time + write_time:.6f}s ({(read_time + write_time)/total_time*100:.2f}%)")
    print(f"Throughput:       {(nnz_A + nnz_B + written_entries)/total_time:,.0f} entries/sec")
    print(f"{'='*70}")
    
    metrics = {
        "operation": "addition",
        "device": "GPU",
        "gpu_name": torch.cuda.get_device_name(0),
        "format": "dense",
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
            "shape": [result_dense.shape[0], result_dense.shape[1]],
            "nnz": int(nnz_result),
            "total_elements": result_dense.shape[0] * result_dense.shape[1],
            "file": out_path
        },
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
        "throughput_entries_per_sec": round((nnz_A + nnz_B + written_entries)/total_time, 2),
        "num_runs": num_runs
    }
    
    metrics_path = os.path.join(results_dir, "metrics_addition_dense_gpu.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Dense GPU addition benchmark')
    parser.add_argument('--input_a', type=str, default='../../input/dense_matrix_a_300x300.csv')
    parser.add_argument('--input_b', type=str, default='../../input/dense_matrix_b_300x300.csv')
    parser.add_argument('--num-runs', type=int, default=1)
    args = parser.parse_args()
    
    run_gpu_benchmark(args.input_a, args.input_b, num_runs=args.num_runs)


if __name__ == "__main__":
    main()
