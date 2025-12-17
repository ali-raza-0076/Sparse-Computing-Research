"""
Sparse Matrix Multiplication (A × B) using Pure COO Format
Implements COO×COO multiplication using hash-based accumulation.

Algorithm: COO×COO Hash Accumulation
1. For each entry (i,k,a_val) in matrix A
2. For each entry (k',j,b_val) in matrix B where k'==k
3. Accumulate result[i,j] += a_val * b_val
4. Use hash map for efficient accumulation

Time Complexity: O(nnz(A) * avg_row_density(B))
Space Complexity: O(nnz(C))
"""

import numpy as np
import numba
import logging
import time
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, cache=True)
def _multiply_coo_coo_numba(rows_a, cols_a, vals_a, rows_b, cols_b, vals_b, num_rows, num_cols):
    """
    Multiply two COO matrices using Numba acceleration.
    
    Strategy: For each entry in A, find matching entries in B and accumulate results.
    Uses hash-based accumulation for efficiency.
    
    Args:
        rows_a, cols_a, vals_a: Matrix A in COO format
        rows_b, cols_b, vals_b: Matrix B in COO format
        num_rows: Number of rows in result (rows of A)
        num_cols: Number of columns in result (cols of B)
    
    Returns:
        (result_rows, result_cols, result_vals)
    """
    result_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    
    b_by_row = numba.typed.Dict.empty(
        key_type=numba.types.int32,
        value_type=numba.types.int64[:]
    )
    
    for idx in range(len(rows_b)):
        row_b = rows_b[idx]
        if row_b not in b_by_row:
            b_by_row[row_b] = np.empty(0, dtype=np.int64)
        b_by_row[row_b] = np.append(b_by_row[row_b], idx)
    
    for a_idx in range(len(rows_a)):
        i = rows_a[a_idx]
        k = cols_a[a_idx]
        a_val = vals_a[a_idx]
        
        if k in b_by_row:
            for b_idx in b_by_row[k]:
                j = cols_b[b_idx]
                b_val = vals_b[b_idx]
                
                key = i * num_cols + j
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
        i = key // num_cols
        j = key % num_cols
        result_rows[idx] = i
        result_cols[idx] = j
        result_vals[idx] = val
        idx += 1
    
    return result_rows, result_cols, result_vals


def sparse_multiply_coo(rows_a, cols_a, vals_a,
                        rows_b, cols_b, vals_b,
                        shape_a: Tuple[int, int],
                        shape_b: Tuple[int, int]) -> Tuple:
    """
    Multiply two matrices in pure COO format: C = A × B
    
    Args:
        rows_a, cols_a, vals_a: Matrix A in COO format (NumPy arrays)
        rows_b, cols_b, vals_b: Matrix B in COO format (NumPy arrays)
        shape_a: Matrix A dimensions (nrows_a, ncols_a)
        shape_b: Matrix B dimensions (nrows_b, ncols_b)
    
    Returns:
        (rows_c, cols_c, vals_c) result in COO format (NumPy arrays)
    """
    if shape_a[1] != shape_b[0]:
        raise ValueError(f"Incompatible shapes: {shape_a} × {shape_b}")
    
    num_rows_result = shape_a[0]
    num_cols_result = shape_b[1]
    
    rows_c, cols_c, vals_c = _multiply_coo_coo_numba(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b,
        num_rows_result, num_cols_result
    )
    
    return rows_c, cols_c, vals_c
