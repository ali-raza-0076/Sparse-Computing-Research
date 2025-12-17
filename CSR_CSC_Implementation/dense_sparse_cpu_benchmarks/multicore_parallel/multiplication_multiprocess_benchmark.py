"""
Benchmark script for TRUE multiprocessing CSR sparse matrix multiplication vs dense NumPy multiplication.
Uses Python multiprocessing with separate processes (not Numba threading).
Tests at different matrix sizes to show overhead vs computation tradeoff.
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core_implementations'))

from sparse_multiplication_multiprocess_csr import sparse_multiply_csr_multiprocess
from matrix_formats import build_csr_from_coo, COOMatrix


def csr_to_dense(indptr, indices, data, num_rows, num_cols):
    """Convert CSR format to dense NumPy matrix."""
    dense = np.zeros((num_rows, num_cols), dtype=np.float64)
    for row in range(num_rows):
        start = indptr[row]
        end = indptr[row + 1]
        for idx in range(start, end):
            col = indices[idx]
            val = data[idx]
            dense[row, col] = val
    return dense


def generate_sparse_coo(rows, cols, sparsity_percent):
    """Generate random sparse matrix in COO format with specified sparsity."""
    total_elements = rows * cols
    num_nonzeros = int(total_elements * (100 - sparsity_percent) / 100)
    
    row_indices = np.random.randint(0, rows, size=num_nonzeros, dtype=np.int32)
    col_indices = np.random.randint(0, cols, size=num_nonzeros, dtype=np.int32)
    values = np.random.randn(num_nonzeros).astype(np.float64)
    
    return (row_indices, col_indices, values)


def benchmark_multiplication(size, sparsity_percent=99.9, num_cores=16):
    """
    Benchmark sparse CSR multiplication (multiprocess) vs dense NumPy multiplication.
    
    Args:
        size: Matrix dimension (size x size)
        sparsity_percent: Percentage of zero elements
        num_cores: Number of cores for multiprocessing
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking {size}x{size} Matrix Multiplication (Sparsity: {sparsity_percent}%)")
    print(f"TRUE Multiprocessing: {num_cores} cores")
    print(f"{'='*80}\n")
    
    print("Generating random sparse matrices A and B...")
    A_coo = generate_sparse_coo(size, size, sparsity_percent)
    B_coo = generate_sparse_coo(size, size, sparsity_percent)
    
    A_row_coo, A_col_coo, A_val_coo = A_coo
    B_row_coo, B_col_coo, B_val_coo = B_coo
    
    print(f"Matrix A: {len(A_val_coo):,} non-zero entries")
    print(f"Matrix B: {len(B_val_coo):,} non-zero entries")
    
    print("Converting to CSR format...")
    A_data = list(zip(A_row_coo, A_col_coo, A_val_coo))
    B_data = list(zip(B_row_coo, B_col_coo, B_val_coo))
    A_coo = COOMatrix((size, size), data=A_data)
    B_coo = COOMatrix((size, size), data=B_data)
    A_csr = build_csr_from_coo(A_coo)
    B_csr = build_csr_from_coo(B_coo)
    A_indptr, A_indices, A_data = A_csr.row_ptr, A_csr.col_idx, A_csr.values
    B_indptr, B_indices, B_data = B_csr.row_ptr, B_csr.col_idx, B_csr.values
    
    sparse_memory_mb = (A_indptr.nbytes + A_indices.nbytes + A_data.nbytes + 
                        B_indptr.nbytes + B_indices.nbytes + B_data.nbytes) / (1024**2)
    print(f"Sparse memory (A + B): {sparse_memory_mb:.2f} MB")
    
    print(f"\n--- Sparse CSR Multiplication (TRUE Multiprocessing, {num_cores} cores) ---")
    
    _ = sparse_multiply_csr_multiprocess(
        A_indptr, A_indices, A_data,
        B_indptr, B_indices, B_data,
        (size, size), (size, size),
        num_cores=num_cores
    )
    
    start = time.perf_counter()
    C_indptr, C_indices, C_data = sparse_multiply_csr_multiprocess(
        A_indptr, A_indices, A_data,
        B_indptr, B_indices, B_data,
        (size, size), (size, size),
        num_cores=num_cores
    )
    sparse_time = time.perf_counter() - start
    
    print(f"Time: {sparse_time:.6f} seconds")
    print(f"Result C: {len(C_data):,} non-zero entries")
    print(f"Memory (result): {(C_indptr.nbytes + C_indices.nbytes + C_data.nbytes) / (1024**2):.2f} MB")
    
    print(f"\n--- Dense NumPy Multiplication (Single-threaded) ---")
    
    print("Converting to dense matrices...")
    A_dense = csr_to_dense(A_indptr, A_indices, A_data, size, size)
    B_dense = csr_to_dense(B_indptr, B_indices, B_data, size, size)
    
    dense_memory_mb = (A_dense.nbytes + B_dense.nbytes) / (1024**2)
    print(f"Dense memory (A + B): {dense_memory_mb:.2f} MB")
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    _ = A_dense @ B_dense
    
    start = time.perf_counter()
    C_dense = A_dense @ B_dense
    dense_time = time.perf_counter() - start
    
    print(f"Time: {dense_time:.6f} seconds")
    print(f"Memory (result): {C_dense.nbytes / (1024**2):.2f} MB")
    
    print(f"\n{'='*80}")
    print("COMPARISON & ANALYSIS")
    print(f"{'='*80}\n")
    
    speedup = dense_time / sparse_time
    memory_ratio = dense_memory_mb / sparse_memory_mb
    
    print(f"Sparse (TRUE Multiprocess): {sparse_time:.6f}s")
    print(f"Dense (NumPy):              {dense_time:.6f}s")
    print(f"Speedup:                    {speedup:.2f}×")
    print(f"Memory ratio (dense/sparse): {memory_ratio:.2f}×")
    
    if speedup > 1:
        print(f"\n✓ Sparse TRUE multiprocessing is {speedup:.2f}× FASTER")
    else:
        print(f"\n✗ Dense is {1/speedup:.2f}× faster (sparse overhead dominates)")
    
    print(f"\nMemory savings: Sparse uses {memory_ratio:.2f}× LESS memory")
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multiplication_multiprocess_csr_{size}x{size}_{sparsity_percent}pct_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"TRUE Multiprocessing CSR Multiplication Benchmark\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Matrix Size: {size}x{size}\n")
        f.write(f"Sparsity: {sparsity_percent}%\n")
        f.write(f"Number of Cores: {num_cores}\n")
        f.write(f"Non-zeros A: {len(A_data):,}\n")
        f.write(f"Non-zeros B: {len(B_data):,}\n")
        f.write(f"Non-zeros C: {len(C_data):,}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Sparse (TRUE Multiprocess): {sparse_time:.6f}s\n")
        f.write(f"  Dense (NumPy):              {dense_time:.6f}s\n")
        f.write(f"  Speedup:                    {speedup:.2f}×\n")
        f.write(f"  Memory ratio:               {memory_ratio:.2f}×\n")
    
    print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark TRUE multiprocess CSR multiplication')
    parser.add_argument('--size', type=int, default=10000, help='Matrix dimension')
    parser.add_argument('--sparsity', type=float, default=99.9, help='Sparsity percentage')
    parser.add_argument('--cores', type=int, default=16, help='Number of CPU cores')
    
    args = parser.parse_args()
    
    benchmark_multiplication(args.size, args.sparsity, args.cores)
