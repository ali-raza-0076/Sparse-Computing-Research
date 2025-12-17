"""
TRUE Parallel Sparse Matrix Multiplication (A × B) using CSR Format
Uses Python multiprocessing for REAL parallel execution across CPU cores.

Algorithm: Row-Block Multiprocessing CSR×CSR Multiplication
1. Build index of B entries by row (for efficient lookup)
2. Divide rows of A into blocks (one per CPU core)
3. Each process independently computes results for its block
4. Merge results from all processes

This is TRUE parallelism - each core runs in separate process with own memory.
DIRECTLY ADAPTED FROM COO_Implementation multiprocessing logic.

Time Complexity: O(nnz(A) * avg_row_density(B) / num_cores)
Space Complexity: O(nnz(C))
"""

import numpy as np
from multiprocessing import Pool
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _build_b_index_from_csr(indptr_b, indices_b, data_b, num_rows_b):
    """
    Build index for matrix B: map row -> list of (col, val) tuples.
    Pure Python for multiprocessing compatibility.
    """
    b_index = {}
    
    for row in range(num_rows_b):
        start = indptr_b[row]
        end = indptr_b[row + 1]
        
        if start < end:
            row_data = []
            for idx in range(start, end):
                col = int(indices_b[idx])
                val = float(data_b[idx])
                row_data.append((col, val))
            b_index[row] = row_data
    
    return b_index


def _process_row_block_multiply(args):
    """
    Process a block of rows for multiplication in parallel.
    This function runs in a separate process for TRUE parallelism.
    
    Args:
        args: Tuple of (start_row, end_row, indptr_a, indices_a, data_a, b_index, num_cols_b)
    
    Returns:
        List of (row_indices, row_values) for each row in block
    """
    start_row, end_row, indptr_a, indices_a, data_a, b_index, num_cols_b = args
    
    block_results = []
    
    for row_a in range(start_row, end_row):
        start_a = indptr_a[row_a]
        end_a = indptr_a[row_a + 1]
        
        row_c_accum = {}
        
        for idx_a in range(start_a, end_a):
            k = int(indices_a[idx_a])
            a_val = float(data_a[idx_a])
            
            if k in b_index:
                for j, b_val in b_index[k]:
                    if j in row_c_accum:
                        row_c_accum[j] += a_val * b_val
                    else:
                        row_c_accum[j] = a_val * b_val
        
        if row_c_accum:
            sorted_items = sorted(row_c_accum.items())
            row_c_indices = [col for col, _ in sorted_items]
            row_c_values = [val for _, val in sorted_items]
        else:
            row_c_indices = []
            row_c_values = []
        
        block_results.append((row_c_indices, row_c_values))
    
    return block_results


def sparse_multiply_csr_multiprocess(indptr_a, indices_a, data_a,
                                     indptr_b, indices_b, data_b,
                                     shape_a: Tuple[int, int],
                                     shape_b: Tuple[int, int],
                                     num_cores=16) -> Tuple:
    """
    Multiply two matrices using TRUE multiprocessing: C = A × B (CSR × CSR)
    
    Each core runs in a SEPARATE PROCESS with independent memory and computation.
    This is REAL parallelism, not just threading.
    
    Args:
        indptr_a, indices_a, data_a: Matrix A in CSR format (NumPy arrays)
        indptr_b, indices_b, data_b: Matrix B in CSR format (NumPy arrays)
        shape_a: Matrix A dimensions (nrows_a, ncols_a)
        shape_b: Matrix B dimensions (nrows_b, ncols_b)
        num_cores: Number of CPU cores/processes to use
    
    Returns:
        (indptr_c, indices_c, data_c) result in CSR format (NumPy arrays)
    """
    num_rows_a, num_cols_a = shape_a
    num_rows_b, num_cols_b = shape_b
    
    if num_cols_a != num_rows_b:
        raise ValueError(f"Incompatible shapes: {shape_a} × {shape_b}")
    
    b_index = _build_b_index_from_csr(indptr_b, indices_b, data_b, num_rows_b)
    
    rows_per_core = max(1, num_rows_a // num_cores)
    blocks = []
    
    for core_id in range(num_cores):
        start_row = core_id * rows_per_core
        if core_id == num_cores - 1:
            end_row = num_rows_a
        else:
            end_row = start_row + rows_per_core
        
        if start_row < num_rows_a:
            blocks.append((
                start_row, end_row,
                indptr_a, indices_a, data_a,
                b_index, num_cols_b
            ))
    
    with Pool(processes=num_cores) as pool:
        block_results = pool.map(_process_row_block_multiply, blocks)
    
    indptr_c = [0]
    indices_c_list = []
    data_c_list = []
    
    for block_result in block_results:
        for row_indices, row_values in block_result:
            indices_c_list.extend(row_indices)
            data_c_list.extend(row_values)
            indptr_c.append(indptr_c[-1] + len(row_indices))
    
    indptr_c = np.array(indptr_c, dtype=np.int64)
    indices_c = np.array(indices_c_list, dtype=np.int32)
    data_c = np.array(data_c_list, dtype=np.float64)
    
    return indptr_c, indices_c, data_c


if __name__ == "__main__":
    print("Testing TRUE Multiprocess CSR Multiplication...")
    
    A_indptr = np.array([0, 2, 3], dtype=np.int64)
    A_indices = np.array([0, 2, 1], dtype=np.int32)
    A_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    B_indptr = np.array([0, 2, 2, 4], dtype=np.int64)
    B_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    B_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    
    C_indptr, C_indices, C_data = sparse_multiply_csr_multiprocess(
        A_indptr, A_indices, A_data,
        B_indptr, B_indices, B_data,
        (2, 3), (3, 2),
        num_cores=2
    )
    
    print(f"Result C (2×2):")
    print(f"  indptr: {C_indptr}")
    print(f"  indices: {C_indices}")
    print(f"  data: {C_data}")
    print(f"  Expected: indptr=[0, 2, 2], indices=[0, 1], data=[7.0, 10.0]")
    print(f"  Non-zeros: {len(C_data)}")
