"""
Benchmark script for TRUE multiprocessing COO sparse matrix addition vs dense NumPy addition.
Uses Python multiprocessing with separate processes (not Numba threading).
Tests at different matrix sizes to show overhead vs computation tradeoff.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sparse_addition_multiprocess_coo import sparse_add_coo_multiprocess


def coo_to_dense(coo_matrix, rows, cols):
    """Convert COO format to dense NumPy matrix."""
    row_indices, col_indices, values = coo_matrix
    dense = np.zeros((rows, cols), dtype=np.float64)
    for i in range(len(values)):
        dense[row_indices[i], col_indices[i]] += values[i]
    return dense


def generate_sparse_coo(rows, cols, sparsity_percent):
    """Generate random sparse matrix in COO format with specified sparsity."""
    total_elements = rows * cols
    num_nonzeros = int(total_elements * (100 - sparsity_percent) / 100)
    
    row_indices = np.random.randint(0, rows, size=num_nonzeros, dtype=np.int32)
    col_indices = np.random.randint(0, cols, size=num_nonzeros, dtype=np.int32)
    values = np.random.randn(num_nonzeros).astype(np.float64)
    
    return (row_indices, col_indices, values)


def benchmark_addition(size, sparsity_percent=99.9, num_cores=16):
    """
    Benchmark sparse COO addition (multiprocess) vs dense NumPy addition.
    
    Args:
        size: Matrix dimension (size x size)
        sparsity_percent: Percentage of zero elements
        num_cores: Number of cores for multiprocessing
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking {size}x{size} Matrix Addition (Sparsity: {sparsity_percent}%)")
    print(f"{'='*80}\n")
    
    print("Generating random sparse matrices A and B...")
    A_coo = generate_sparse_coo(size, size, sparsity_percent)
    B_coo = generate_sparse_coo(size, size, sparsity_percent)
    
    A_row, A_col, A_val = A_coo
    B_row, B_col, B_val = B_coo
    
    print(f"Matrix A: {len(A_val):,} non-zero entries")
    print(f"Matrix B: {len(B_val):,} non-zero entries")
    
    sparse_memory_mb = (A_row.nbytes + A_col.nbytes + A_val.nbytes + 
                        B_row.nbytes + B_col.nbytes + B_val.nbytes) / (1024**2)
    print(f"Sparse memory (A + B): {sparse_memory_mb:.2f} MB")
    
    print(f"\n--- Sparse COO Addition (TRUE Multiprocessing, {num_cores} cores) ---")
    
    _ = sparse_add_coo_multiprocess(A_row, A_col, A_val, B_row, B_col, B_val, num_cores=num_cores)
    
    start = time.perf_counter()
    C_row, C_col, C_val = sparse_add_coo_multiprocess(A_row, A_col, A_val, B_row, B_col, B_val, num_cores=num_cores)
    sparse_time = time.perf_counter() - start
    
    C_row, C_col, C_val = C_row, C_col, C_val
    print(f"Time: {sparse_time:.6f} seconds")
    print(f"Result C: {len(C_val):,} non-zero entries")
    print(f"Memory (result): {(C_row.nbytes + C_col.nbytes + C_val.nbytes) / (1024**2):.2f} MB")
    
    print(f"\n--- Dense NumPy Addition (Single-threaded) ---")
    
    print("Converting to dense matrices...")
    A_dense = coo_to_dense((A_row, A_col, A_val), size, size)
    B_dense = coo_to_dense((B_row, B_col, B_val), size, size)
    
    dense_memory_mb = (A_dense.nbytes + B_dense.nbytes) / (1024**2)
    print(f"Dense memory (A + B): {dense_memory_mb:.2f} MB")
    
    original_threads = os.environ.get('OMP_NUM_THREADS', None)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    _ = A_dense + B_dense
    
    start = time.perf_counter()
    C_dense = A_dense + B_dense
    dense_time = time.perf_counter() - start
    
    if original_threads:
        os.environ['OMP_NUM_THREADS'] = original_threads
    else:
        os.environ.pop('OMP_NUM_THREADS', None)
    
    print(f"Time: {dense_time:.6f} seconds")
    print(f"Memory (result): {C_dense.nbytes / (1024**2):.2f} MB")
    
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    print(f"Sparse (multiprocess, {num_cores} cores): {sparse_time:.6f} seconds")
    print(f"Dense (single-thread):                    {dense_time:.6f} seconds")
    print(f"Speedup (Sparse/Dense):                   {dense_time/sparse_time:.2f}x")
    print(f"Memory savings:                           {dense_memory_mb/sparse_memory_mb:.2f}x")
    print(f"{'='*80}\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sparsity_str = f"{sparsity_percent:.1f}".replace('.', '_')
    result_filename = f"addition_multiprocess_{size}x{size}_sparsity{sparsity_str}pct_results.txt"
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, result_filename)
    
    with open(result_path, 'w') as f:
        f.write(f"COO Sparse Matrix Addition Benchmark - TRUE Multiprocessing\n")
        f.write(f"{'='*80}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Matrix Size: {size} x {size}\n")
        f.write(f"Sparsity: {sparsity_percent}%\n")
        f.write(f"Number of Cores: {num_cores}\n")
        f.write(f"\n")
        f.write(f"Matrix A: {len(A_val):,} non-zero entries\n")
        f.write(f"Matrix B: {len(B_val):,} non-zero entries\n")
        f.write(f"Result C: {len(C_val):,} non-zero entries\n")
        f.write(f"\n")
        f.write(f"Sparse COO Addition (multiprocess, {num_cores} cores): {sparse_time:.6f} seconds\n")
        f.write(f"Dense NumPy Addition (single-thread):                  {dense_time:.6f} seconds\n")
        f.write(f"\n")
        f.write(f"Speedup (Dense/Sparse): {dense_time/sparse_time:.2f}x\n")
        f.write(f"Memory Savings: {dense_memory_mb/sparse_memory_mb:.2f}x\n")
        f.write(f"\n")
        f.write(f"Sparse Memory (A+B): {sparse_memory_mb:.2f} MB\n")
        f.write(f"Dense Memory (A+B): {dense_memory_mb:.2f} MB\n")
        f.write(f"{'='*80}\n")
    
    print(f"Results saved to: {result_filename}")
    
    return {
        'size': size,
        'sparsity': sparsity_percent,
        'sparse_time': sparse_time,
        'dense_time': dense_time,
        'speedup': dense_time / sparse_time,
        'sparse_nnz': len(C_val),
        'sparse_memory_mb': sparse_memory_mb,
        'dense_memory_mb': dense_memory_mb
    }


if __name__ == "__main__":
    size = 1000
    sparsity = 99.9
    num_cores = 16
    
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) > 2:
        sparsity = float(sys.argv[2])
    if len(sys.argv) > 3:
        num_cores = int(sys.argv[3])
    
    print("\n" + "="*80)
    print("COO SPARSE MATRIX ADDITION BENCHMARK - TRUE MULTIPROCESSING")
    print("="*80)
    print(f"Using Python multiprocessing with {num_cores} separate processes")
    print(f"Matrix Size: {size}x{size}")
    print(f"Sparsity: {sparsity}%")
    print("="*80 + "\n")
    
    results = benchmark_addition(size, sparsity, num_cores)
    
    print("\n✓ Benchmark completed successfully!")
