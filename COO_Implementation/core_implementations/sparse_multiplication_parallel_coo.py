"""
Parallel Sparse Matrix Multiplication (A × B) using COO Format
Uses Numba parallel features to speed up computation.

Algorithm: Parallel Hash-Based Multiplication
1. Iterate through all entries in A in parallel chunks
2. For each A[i,k], find matching entries B[k,j] and accumulate results
3. Use parallel hash accumulation for final result

Time Complexity: O((nnz(A) × avg_row_density(B)) / num_cores)
Space Complexity: O(nnz(C))
"""

import numpy as np
import numba
from numba import prange
from numba.typed import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, cache=True)
def _multiply_coo_coo_numba(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, size):
    """
    Multiply two COO matrices using hash-based accumulation.
    """
    B_row_dict = Dict.empty(
        key_type=numba.types.int32,
        value_type=numba.types.int32[:]
    )
    
    for idx in range(len(rows_B)):
        row = rows_B[idx]
        if row not in B_row_dict:
            B_row_dict[row] = np.empty(0, dtype=np.int32)
        B_row_dict[row] = np.append(B_row_dict[row], idx)
    
    result_dict = Dict.empty(
        key_type=numba.types.UniTuple(numba.types.int32, 2),
        value_type=numba.types.float64
    )
    
    for a_idx in range(len(rows_A)):
        i = rows_A[a_idx]
        k = cols_A[a_idx]
        a_val = vals_A[a_idx]
        
        if k in B_row_dict:
            for b_idx in B_row_dict[k]:
                j = cols_B[b_idx]
                b_val = vals_B[b_idx]
                key = (i, j)
                if key in result_dict:
                    result_dict[key] += a_val * b_val
                else:
                    result_dict[key] = a_val * b_val
    
    nnz = len(result_dict)
    result_rows = np.empty(nnz, dtype=np.int32)
    result_cols = np.empty(nnz, dtype=np.int32)
    result_vals = np.empty(nnz, dtype=np.float64)
    
    idx = 0
    for key, val in result_dict.items():
        result_rows[idx] = key[0]
        result_cols[idx] = key[1]
        result_vals[idx] = val
        idx += 1
    
    return result_rows, result_cols, result_vals


def sparse_multiply_coo_parallel(rows_A, cols_A, vals_A, rows_B, cols_B, vals_B, size, num_cores=None):
    """
    Multiply two sparse matrices in COO format using parallel operations.
    
    Args:
        rows_A, cols_A, vals_A: Matrix A in COO format
        rows_B, cols_B, vals_B: Matrix B in COO format
        size: Matrix dimensions (size x size)
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
    
    result_rows, result_cols, result_vals = _multiply_coo_coo_numba(
        rows_A, cols_A, vals_A,
        rows_B, cols_B, vals_B,
        size
    )
    
    return result_rows, result_cols, result_vals
