"""
CSR Addition Benchmark - Database I/O Workflow

Benchmarks sparse matrix addition using CSR format with database I/O:
- Phase 1: Read from disk (CSV in COO format)
- Phase 2: Convert to CSR, add, compute in memory
- Phase 3: Write result to disk (CSV)
"""
import numpy as np
import time
import csv
import os
import json
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../CSR_CSC_Implementation/core_implementations')))
from matrix_formats import COOMatrix, CSRMatrix, build_csr_from_coo
from sparse_addition_csr import sparse_add_csr


def read_coo_from_csv(filepath):
    """Read COO matrix from CSV file."""
    data = []
    max_row = max_col = 0
    with open(filepath, 'r') as f:
        for line in f:
            i, j, v = line.strip().split(',')
            i, j, v = int(i), int(j), float(v)
            data.append((i, j, v))
            max_row = max(max_row, i)
            max_col = max(max_col, j)
    shape = (max_row + 1, max_col + 1)
    return data, shape


def write_csr_to_csv(csr_matrix, filepath):
    """Write CSR matrix to CSV in COO format, filtering out zeros and ensuring no duplicates."""
    seen = set()
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in range(csr_matrix.shape[0]):
            start = csr_matrix.row_ptr[row]
            end = csr_matrix.row_ptr[row + 1]
            for idx in range(start, end):
                col = csr_matrix.col_idx[idx]
                val = csr_matrix.values[idx]
                if val != 0 and (row, col) not in seen:
                    writer.writerow([row, col, int(val)])
                    seen.add((row, col))


def run_addition_benchmark(input_a, input_b, num_runs=1):
    """Run CSR addition benchmark with database I/O workflow."""
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: CSR Addition Benchmark")
    print(f"{'='*70}")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    read_start = time.perf_counter()
    
    data_A, shape_A = read_coo_from_csv(input_a)
    data_B, shape_B = read_coo_from_csv(input_b)
    
    read_end = time.perf_counter()
    read_time = read_end - read_start
    
    print(f"  Matrix A: {input_a}")
    print(f"    Shape: {shape_A}, {len(data_A):,} non-zeros")
    print(f"  Matrix B: {input_b}")
    print(f"    Shape: {shape_B}, {len(data_B):,} non-zeros")
    print(f"  I/O Read Time: {read_time:.6f}s")
    
    print(f"\n[PHASE 2] Processing in RAM (CSR format)...")
    compute_start = time.perf_counter()
    
    print(f"  Converting COO -> CSR...")
    data_A_list = [(r, c, v) for r, c, v in data_A]
    data_B_list = [(r, c, v) for r, c, v in data_B]
    coo_A = COOMatrix(shape_A, data=data_A_list)
    coo_B = COOMatrix(shape_B, data=data_B_list)
    csr_A = build_csr_from_coo(coo_A)
    csr_B = build_csr_from_coo(coo_B)
    
    times = []
    for _ in range(num_runs):
        run_start = time.perf_counter()
        result_csr = sparse_add_csr(csr_A, csr_B)
        run_end = time.perf_counter()
        times.append(run_end - run_start)
    
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    nnz_result = len(result_csr.values)
    print(f"  Computation Time: {compute_time:.6f}s (avg: {avg_time:.6f}s ± {std_time:.6f}s)")
    print(f"  Result: {result_csr.shape}, {nnz_result:,} non-zeros")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    write_start = time.perf_counter()
    
    os.makedirs('results', exist_ok=True)
    output_file = f'results/addition_csr_{result_csr.shape[0]}x{result_csr.shape[1]}.csv'
    write_csr_to_csv(result_csr, output_file)
    
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
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (Read):  {read_time:.6f}s ({read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (Compute): {compute_time:.6f}s ({compute_time/total_time*100:.2f}%)")
    print(f"  Phase 3 (Write): {write_time:.6f}s ({write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {io_overhead:.6f}s ({io_percent:.2f}%)")
    print(f"Throughput:       {nnz_result/total_time:,.0f} entries/sec")
    print(f"{'='*70}\n")
    
    metrics = {
        'operation': 'addition',
        'format': 'CSR',
        'total_time': float(total_time),
        'io_read_time': float(read_time),
        'compute_time': float(compute_time),
        'io_write_time': float(write_time),
        'io_overhead': float(io_overhead),
        'io_percent': float(io_percent),
        'throughput': float(nnz_result/total_time),
        'input_nnz_a': len(data_A),
        'input_nnz_b': len(data_B),
        'result_nnz': nnz_result,
        'result_shape': result_csr.shape
    }
    
    metrics_file = 'results/metrics_addition_csr.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description='CSR Addition Benchmark - Database I/O')
    parser.add_argument('--input_a', type=str, default='input/matrix_a_5k.csv',
                       help='Path to matrix A CSV file')
    parser.add_argument('--input_b', type=str, default='input/matrix_a_5k.csv',
                       help='Path to matrix B CSV file (default: same as A for dimension compatibility)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs for timing')
    args = parser.parse_args()
    
    run_addition_benchmark(args.input_a, args.input_b, args.num_runs)


if __name__ == '__main__':
    main()
