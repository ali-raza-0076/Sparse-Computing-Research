"""
Parallel Sparse Matrix Addition (A + B) using COO Format
Uses Numba parallel features to speed up sorting and merging operations.

Algorithm: Parallel Sort-Merge Addition  
1. Sort both matrices by (row, col) in parallel
2. Use parallel two-pointer merge for addition
3. Combine results

Time Complexity: O((nnz(A) + nnz(B)) / num_cores)
Space Complexity: O(nnz(A) + nnz(B))
"""

import numpy as np
import numba
from numba import prange
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, parallel=True, cache=True)
def _parallel_sort_coo(rows, cols, vals):
    """
    Sort COO matrix by (row, col) using parallel argsort.
    """
    n = len(rows)
    keys = rows.astype(np.int64) * 1000000 + cols.astype(np.int64)
    indices = np.argsort(keys)
    
    sorted_rows = np.empty(n, dtype=np.int32)
    sorted_cols = np.empty(n, dtype=np.int32)
    sorted_vals = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        idx = indices[i]
        sorted_rows[i] = rows[idx]
        sorted_cols[i] = cols[idx]
        sorted_vals[i] = vals[idx]
    
    return sorted_rows, sorted_cols, sorted_vals


@numba.jit(nopython=True, cache=True)
def _merge_sorted_coo(rows1, cols1, vals1, rows2, cols2, vals2):
    """
    Merge two sorted COO matrices using two-pointer technique.
    """
    n1, n2 = len(rows1), len(rows2)
    max_size = n1 + n2
    result_rows = np.empty(max_size, dtype=np.int32)
    result_cols = np.empty(max_size, dtype=np.int32)
    result_vals = np.empty(max_size, dtype=np.float64)
    
    i, j, k = 0, 0, 0
    
    while i < n1 and j < n2:
        r1, c1, v1 = rows1[i], cols1[i], vals1[i]
        r2, c2, v2 = rows2[j], cols2[j], vals2[j]
        
        if r1 < r2 or (r1 == r2 and c1 < c2):
            result_rows[k] = r1
            result_cols[k] = c1
            result_vals[k] = v1
            i += 1
        elif r1 > r2 or (r1 == r2 and c1 > c2):
            result_rows[k] = r2
            result_cols[k] = c2
            result_vals[k] = v2
            j += 1
        else:
            result_rows[k] = r1
            result_cols[k] = c1
            result_vals[k] = v1 + v2
            i += 1
            j += 1
        k += 1
    
    while i < n1:
        result_rows[k] = rows1[i]
        result_cols[k] = cols1[i]
        result_vals[k] = vals1[i]
        i += 1
        k += 1
    
    while j < n2:
        result_rows[k] = rows2[j]
        result_cols[k] = cols2[j]
        result_vals[k] = vals2[j]
        j += 1
        k += 1
    
    return result_rows[:k], result_cols[:k], result_vals[:k]


def sparse_add_coo_parallel(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, num_cores=None):
    """
    Add two sparse matrices in COO format using parallel operations.
    
    Args:
        rows_A, cols_A, vals_A: Matrix A in COO format
        rows_B, cols_B, vals_B: Matrix B in COO format
        num_cores: Number of cores (for compatibility, Numba uses available threads)
    
    Returns:
        (result_rows, result_cols, result_vals): Result in COO format
    """
    if num_cores is not None:
        numba.set_num_threads(num_cores)
    
    rows_A = np.array(rows_A, dtype=np.int32)
    cols_A = np.array(cols_A, dtype=np.int32)
    vals_A = np.array(vals_A, dtype=np.float64)
    
    rows_B = np.array(rows_B, dtype=np.int32)
    cols_B = np.array(cols_B, dtype=np.int32)
    vals_B = np.array(vals_B, dtype=np.float64)
    
    sorted_rows_A, sorted_cols_A, sorted_vals_A = _parallel_sort_coo(rows_A, cols_A, vals_A)
    sorted_rows_B, sorted_cols_B, sorted_vals_B = _parallel_sort_coo(rows_B, cols_B, vals_B)
    
    result_rows, result_cols, result_vals = _merge_sorted_coo(
        sorted_rows_A, sorted_cols_A, sorted_vals_A,
        sorted_rows_B, sorted_cols_B, sorted_vals_B
    )
    
    return result_rows, result_cols, result_vals
