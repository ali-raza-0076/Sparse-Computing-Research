"""
TRUE Parallel Sparse Matrix Addition (A + B) using CSR Format
Uses Python multiprocessing for REAL parallel execution across CPU cores.

Algorithm: Row-Block Multiprocessing CSR Addition
1. Divide rows of A and B into blocks (one per CPU core)
2. Each process independently adds corresponding row blocks
3. Merge results from all processes into final CSR matrix

This is TRUE parallelism - each core runs in separate process with own memory.

Time Complexity: O((nnz(A) + nnz(B)) / num_cores)
Space Complexity: O(nnz(A) + nnz(B))
"""

import numpy as np
from multiprocessing import Pool
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _add_csr_row_pair(row_a_indices, row_a_values, row_b_indices, row_b_values):
    """
    Add two CSR rows using two-pointer merge technique.
    
    Args:
        row_a_indices, row_a_values: Column indices and values for row from A
        row_b_indices, row_b_values: Column indices and values for row from B
    
    Returns:
        (result_indices, result_values): Merged row data
    """
    result_indices = []
    result_values = []
    
    i, j = 0, 0
    n_a, n_b = len(row_a_indices), len(row_b_indices)
    
    while i < n_a and j < n_b:
        col_a = row_a_indices[i]
        col_b = row_b_indices[j]
        
        if col_a < col_b:
            result_indices.append(col_a)
            result_values.append(row_a_values[i])
            i += 1
        elif col_a > col_b:
            result_indices.append(col_b)
            result_values.append(row_b_values[j])
            j += 1
        else:
            result_indices.append(col_a)
            result_values.append(row_a_values[i] + row_b_values[j])
            i += 1
            j += 1
    
    while i < n_a:
        result_indices.append(row_a_indices[i])
        result_values.append(row_a_values[i])
        i += 1
    
    while j < n_b:
        result_indices.append(row_b_indices[j])
        result_values.append(row_b_values[j])
        j += 1
    
    return result_indices, result_values


def _process_row_block(args):
    """
    Process a block of rows in parallel.
    This function runs in a separate process for TRUE parallelism.
    
    Args:
        args: Tuple of (start_row, end_row, indptr_a, indices_a, data_a, 
                       indptr_b, indices_b, data_b)
    
    Returns:
        List of (row_indices, row_values) for each row in block
    """
    start_row, end_row, indptr_a, indices_a, data_a, indptr_b, indices_b, data_b = args
    
    block_results = []
    
    for row in range(start_row, end_row):
        start_a = indptr_a[row]
        end_a = indptr_a[row + 1]
        row_a_indices = indices_a[start_a:end_a].tolist()
        row_a_values = data_a[start_a:end_a].tolist()
        
        start_b = indptr_b[row]
        end_b = indptr_b[row + 1]
        row_b_indices = indices_b[start_b:end_b].tolist()
        row_b_values = data_b[start_b:end_b].tolist()
        
        result_indices, result_values = _add_csr_row_pair(
            row_a_indices, row_a_values,
            row_b_indices, row_b_values
        )
        
        block_results.append((result_indices, result_values))
    
    return block_results


def sparse_add_csr_multiprocess(indptr_a, indices_a, data_a,
                                indptr_b, indices_b, data_b,
                                num_cores=16) -> Tuple:
    """
    Add two matrices in CSR format using TRUE multiprocessing: C = A + B
    
    Each core runs in a SEPARATE PROCESS with independent memory and computation.
    This is REAL parallelism, not just threading.
    
    Args:
        indptr_a, indices_a, data_a: Matrix A in CSR format (NumPy arrays)
        indptr_b, indices_b, data_b: Matrix B in CSR format (NumPy arrays)
        num_cores: Number of CPU cores/processes to use
    
    Returns:
        (indptr_c, indices_c, data_c) result in CSR format (NumPy arrays)
    """
    num_rows = len(indptr_a) - 1
    
    if num_rows != len(indptr_b) - 1:
        raise ValueError("Matrices must have same number of rows")
    
    rows_per_core = max(1, num_rows // num_cores)
    blocks = []
    
    for core_id in range(num_cores):
        start_row = core_id * rows_per_core
        if core_id == num_cores - 1:
            end_row = num_rows
        else:
            end_row = start_row + rows_per_core
        
        if start_row < num_rows:
            blocks.append((
                start_row, end_row,
                indptr_a, indices_a, data_a,
                indptr_b, indices_b, data_b
            ))
    
    with Pool(processes=num_cores) as pool:
        block_results = pool.map(_process_row_block, blocks)
    
    indptr_c = [0]
    indices_c_list = []
    data_c_list = []
    
    for block_result in block_results:
        for row_indices, row_values in block_result:
            indices_c_list.extend(row_indices)
            data_c_list.extend(row_values)
            indptr_c.append(indptr_c[-1] + len(row_indices))
    
    indptr_c = np.array(indptr_c, dtype=np.int32)
    indices_c = np.array(indices_c_list, dtype=np.int32)
    data_c = np.array(data_c_list, dtype=np.float64)
    
    return indptr_c, indices_c, data_c


if __name__ == "__main__":
    print("Testing TRUE Multiprocess CSR Addition...")
    
    indptr_a = np.array([0, 2, 3, 5], dtype=np.int32)
    indices_a = np.array([0, 2, 1, 2, 3], dtype=np.int32)
    data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    indptr_b = np.array([0, 1, 3, 4], dtype=np.int32)
    indices_b = np.array([1, 1, 3, 0], dtype=np.int32)
    data_b = np.array([6.0, 7.0, 8.0, 9.0], dtype=np.float64)
    
    indptr_c, indices_c, data_c = sparse_add_csr_multiprocess(
        indptr_a, indices_a, data_a,
        indptr_b, indices_b, data_b,
        num_cores=2
    )
    
    print(f"Result: {len(data_c)} non-zeros")
    print(f"indptr: {indptr_c}")
    print(f"indices: {indices_c}")
    print(f"data: {data_c}")
