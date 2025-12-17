"""
TRUE Parallel Sparse Matrix Multiplication (A × B) using COO Format
Uses Python multiprocessing for REAL parallel execution across CPU cores.

Algorithm: Row-Block Multiprocessing COO×COO Multiplication
1. Build index of B entries by row (shared via arguments)
2. Divide A entries into blocks (one per CPU core)
3. Each process independently computes results for its block
4. Merge results from all processes with deduplication

This is TRUE parallelism - each core runs in separate process with own memory.

Time Complexity: O(nnz(A) * avg_row_density(B) / num_cores)
Space Complexity: O(nnz(C))
"""

import numpy as np
from multiprocessing import Pool
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_b_index_python(rows_b, cols_b, vals_b):
    """
    Build index for matrix B: map row -> list of (col, val) tuples.
    Pure Python for multiprocessing compatibility.
    """
    b_index = {}
    
    for idx in range(len(rows_b)):
        row = int(rows_b[idx])
        col = int(cols_b[idx])
        val = float(vals_b[idx])
        
        if row not in b_index:
            b_index[row] = []
        b_index[row].append((col, val))
    
    return b_index


def _process_a_block(args):
    """
    Process a block of A entries in parallel.
    This function runs in a separate process for TRUE parallelism.
    
    Args:
        args: Tuple of (a_block_data, b_index, num_cols)
            a_block_data: List of (row, col, val) tuples from A
            b_index: Dict mapping B row -> list of (col, val)
            num_cols: Number of columns in result matrix
    
    Returns:
        Dict mapping (i,j) -> accumulated value for this block
    """
    a_block_data, b_index, num_cols = args
    
    local_results = {}
    
    for i, k, a_val in a_block_data:
        if k in b_index:
            for j, b_val in b_index[k]:
                key = (i, j)
                if key in local_results:
                    local_results[key] += a_val * b_val
                else:
                    local_results[key] = a_val * b_val
    
    return local_results


def sparse_multiply_coo_multiprocess(rows_a, cols_a, vals_a,
                                     rows_b, cols_b, vals_b,
                                     shape_a: Tuple[int, int],
                                     shape_b: Tuple[int, int],
                                     num_cores=16) -> Tuple:
    """
    Multiply two matrices in pure COO format using TRUE multiprocessing: C = A × B
    
    Each core runs in a SEPARATE PROCESS with independent memory and computation.
    This is REAL parallelism, not just threading.
    
    Args:
        rows_a, cols_a, vals_a: Matrix A in COO format (NumPy arrays)
        rows_b, cols_b, vals_b: Matrix B in COO format (NumPy arrays)
        shape_a: Matrix A dimensions (nrows_a, ncols_a)
        shape_b: Matrix B dimensions (nrows_b, ncols_b)
        num_cores: Number of CPU cores/processes to use
    
    Returns:
        (rows_c, cols_c, vals_c) result in COO format (NumPy arrays)
    """
    if shape_a[1] != shape_b[0]:
        raise ValueError(f"Incompatible shapes: {shape_a} × {shape_b}")
    
    num_rows_result = shape_a[0]
    num_cols_result = shape_b[1]
    
    b_index = _build_b_index_python(rows_b, cols_b, vals_b)
    
    nnz_a = len(rows_a)
    a_data = [(int(rows_a[i]), int(cols_a[i]), float(vals_a[i])) 
              for i in range(nnz_a)]
    
    block_size = max(1, nnz_a // num_cores)
    blocks = []
    
    for core_id in range(num_cores):
        start_idx = core_id * block_size
        if core_id == num_cores - 1:
            end_idx = nnz_a
        else:
            end_idx = start_idx + block_size
        
        if start_idx < nnz_a:
            block_data = a_data[start_idx:end_idx]
            blocks.append((block_data, b_index, num_cols_result))
    
    with Pool(processes=num_cores) as pool:
        block_results = pool.map(_process_a_block, blocks)
    
    final_results = {}
    for block_result in block_results:
        for key, val in block_result.items():
            if key in final_results:
                final_results[key] += val
            else:
                final_results[key] = val
    
    nnz_c = len(final_results)
    rows_c = np.empty(nnz_c, dtype=np.int32)
    cols_c = np.empty(nnz_c, dtype=np.int32)
    vals_c = np.empty(nnz_c, dtype=np.float64)
    
    idx = 0
    for (i, j), val in final_results.items():
        rows_c[idx] = i
        cols_c[idx] = j
        vals_c[idx] = val
        idx += 1
    
    return rows_c, cols_c, vals_c


if __name__ == "__main__":
    print("Testing TRUE Multiprocess COO Multiplication...")
    
    rows_a = np.array([0, 0, 1], dtype=np.int32)
    cols_a = np.array([0, 2, 1], dtype=np.int32)
    vals_a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    shape_a = (2, 3)
    
    rows_b = np.array([0, 1, 2], dtype=np.int32)
    cols_b = np.array([0, 1, 0], dtype=np.int32)
    vals_b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    shape_b = (3, 2)
    
    rows_c, cols_c, vals_c = sparse_multiply_coo_multiprocess(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b,
        shape_a, shape_b,
        num_cores=4
    )
    
    print(f"Result: {len(rows_c)} non-zeros")
    for i in range(len(rows_c)):
        print(f"  ({rows_c[i]}, {cols_c[i]}) = {vals_c[i]}")
