"""
GPU Multiplication Benchmark - Sparse Format with Database I/O

Benchmarks sparse matrix multiplication using PyTorch on GPU with database I/O:
- Phase 1: Read from disk (CSV)
- Phase 2: Convert to sparse tensors, transfer to GPU, multiply, transfer back
- Phase 3: Write result to disk (CSV)
"""
import torch
import numpy as np
import time
import csv
import os
import json
import argparse


def read_coo_from_csv(filepath):
    """Read COO matrix from CSV file."""
    rows, cols, vals = [], [], []
    max_row = max_col = 0
    with open(filepath, 'r') as f:
        for line in f:
            i, j, v = line.strip().split(',')
            i, j, v = int(i), int(j), float(v)
            rows.append(i)
            cols.append(j)
            vals.append(v)
            max_row = max(max_row, i)
            max_col = max(max_col, j)
    shape = (max_row + 1, max_col + 1)
    return rows, cols, vals, shape


def coo_to_sparse_torch(rows, cols, vals, shape, device='cuda'):
    """Convert COO to PyTorch sparse tensor."""
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    return sparse_tensor


def sparse_to_coo_lists(sparse_tensor):
    """Convert PyTorch sparse tensor back to COO lists for CSV writing."""
    coo_tensor = sparse_tensor.to_sparse_coo().cpu()
    indices = coo_tensor.indices()
    values = coo_tensor.values()
    rows = indices[0].numpy().tolist()
    cols = indices[1].numpy().tolist()
    vals = values.numpy().tolist()
    return rows, cols, vals


def write_coo_to_csv(rows, cols, vals, filepath):
    """Write COO format to CSV, filtering zeros and duplicates."""
    seen = set()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in zip(rows, cols, vals):
            if v != 0 and (i, j) not in seen:
                writer.writerow([i, j, int(v)])
                seen.add((i, j))


def run_gpu_multiplication_benchmark(input_a, input_b, num_runs=1):
    """Run GPU sparse multiplication benchmark with database I/O workflow."""
    print(f"\n{'='*70}")
    print(f"GPU DATABASE I/O WORKFLOW: Sparse Multiplication Benchmark")
    print(f"{'='*70}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU. "
                         "In Google Colab, go to Runtime > Change runtime type > Select T4 GPU.")
    
    device = 'cuda:0'
    print(f"\nGPU Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    
    rows_A, cols_A, vals_A, shape_A = read_coo_from_csv(input_a)
    rows_B, cols_B, vals_B, shape_B = read_coo_from_csv(input_b)
    
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    print(f"  Matrix A: {input_a}")
    print(f"    Shape: {shape_A}, {len(vals_A):,} non-zeros")
    print(f"  Matrix B: {input_b}")
    print(f"    Shape: {shape_B}, {len(vals_B):,} non-zeros")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    print(f"\n[PHASE 2] Processing on {device.upper()} (Sparse format)...")
    compute_start = time.perf_counter()
    
    print(f"  Converting COO -> Sparse and transferring to {device.upper()}...")
    sparse_A = coo_to_sparse_torch(rows_A, cols_A, vals_A, shape_A, device)
    sparse_B = coo_to_sparse_torch(rows_B, cols_B, vals_B, shape_B, device)
    
    _ = torch.sparse.mm(sparse_A, sparse_B)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        run_start = time.perf_counter()
        
        result_sparse = torch.sparse.mm(sparse_A, sparse_B)
        
        torch.cuda.synchronize()
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    
    result_sparse = result_sparse.cpu()
    
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    rows_out, cols_out, vals_out = sparse_to_coo_lists(result_sparse)
    nnz_result = len(vals_out)
    
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s +- {std_time:.6f}s)")
    print(f"  Result: {result_sparse.shape}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    
    os.makedirs('results', exist_ok=True)
    output_file = f'results/multiplication_gpu_sparse_{result_sparse.shape[0]}x{result_sparse.shape[1]}.csv'
    write_coo_to_csv(rows_out, cols_out, vals_out, output_file)
    
    write_end = time.perf_counter()
    write_time = write_end - write_start
    print(f"  I/O Write Time: {write_time:.6f}s")
    print(f"  Result written to: {os.path.abspath(output_file)}")
    
    total_time = read_time + compute_time + write_time
    io_overhead = read_time + write_time
    io_percent = (io_overhead / total_time) * 100
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Device:           {device.upper()}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):  {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (GPU Compute): {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write): {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {io_overhead:.6f}s ({io_percent:.2f}%)")
    print(f"Throughput:       {nnz_result/total_time:,.0f} entries/sec")
    print(f"{'='*70}\n")
    
    metrics = {
        'operation': 'multiplication',
        'format': 'Sparse',
        'device': device,
        'total_time': float(total_time),
        'io_read_time': float(read_time),
        'compute_time': float(compute_time),
        'io_write_time': float(write_time),
        'io_overhead': float(io_overhead),
        'io_percent': float(io_percent),
        'throughput': float(nnz_result/total_time),
        'input_nnz_a': len(vals_A),
        'input_nnz_b': len(vals_B),
        'result_nnz': nnz_result,
        'result_shape': list(result_sparse.shape)
    }
    
    metrics_file = 'results/metrics_multiplication_gpu_sparse.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='GPU Sparse Multiplication Benchmark - Database I/O')
    parser.add_argument('--input_a', type=str, default='../../input/matrix_a_5k.csv',
                       help='Path to matrix A CSV file')
    parser.add_argument('--input_b', type=str, default='../../input/matrix_b_5k.csv',
                       help='Path to matrix B CSV file')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of runs for timing')
    args = parser.parse_args()
    
    run_gpu_multiplication_benchmark(args.input_a, args.input_b, args.num_runs)


if __name__ == '__main__':
    main()
