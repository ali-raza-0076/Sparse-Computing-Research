"""
Sparse Matrix Addition (A + B) using Pure COO Format
Implements efficient two-pointer merge algorithm for sorted COO matrices.

Algorithm: Sort-Merge Addition  
1. Sort both matrices by (row, col)
2. Use two pointers to traverse both matrices simultaneously
3. Merge entries: sum if (i,j) matches, otherwise keep individual entries

Time Complexity: O(nnz(A) + nnz(B))
Space Complexity: O(nnz(A) + nnz(B))
"""

import numpy as np
import numba
import logging
import time
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, cache=True)
def _merge_sorted_coo(rows1, cols1, vals1, rows2, cols2, vals2):
    """
    Merge two sorted COO matrices using two-pointer technique.
    
    Args:
        rows1, cols1, vals1: First matrix (sorted by row, col)
        rows2, cols2, vals2: Second matrix (sorted by row, col)
    
    Returns:
        (result_rows, result_cols, result_vals)
    """
    n1, n2 = len(rows1), len(rows2)
    
    max_size = n1 + n2
    result_rows = np.empty(max_size, dtype=np.int32)
    result_cols = np.empty(max_size, dtype=np.int32)
    result_vals = np.empty(max_size, dtype=np.float64)
    
    i, j, k = 0, 0, 0
    
    while i < n1 and j < n2:
        if rows1[i] < rows2[j]:
            result_rows[k] = rows1[i]
            result_cols[k] = cols1[i]
            result_vals[k] = vals1[i]
            i += 1
            k += 1
        elif rows1[i] > rows2[j]:
            result_rows[k] = rows2[j]
            result_cols[k] = cols2[j]
            result_vals[k] = vals2[j]
            j += 1
            k += 1
        else:
            if cols1[i] < cols2[j]:
                result_rows[k] = rows1[i]
                result_cols[k] = cols1[i]
                result_vals[k] = vals1[i]
                i += 1
                k += 1
            elif cols1[i] > cols2[j]:
                result_rows[k] = rows2[j]
                result_cols[k] = cols2[j]
                result_vals[k] = vals2[j]
                j += 1
                k += 1
            else:
                result_rows[k] = rows1[i]
                result_cols[k] = cols1[i]
                result_vals[k] = vals1[i] + vals2[j]
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


@numba.jit(nopython=True, cache=True)
def _sort_coo(rows, cols, vals):
    """
    Sort COO matrix by (row, col) using lexicographic order.
    Returns sorted arrays.
    """
    n = len(rows)
    keys = rows.astype(np.int64) * 1000000000 + cols.astype(np.int64)
    indices = np.argsort(keys)
    
    return rows[indices], cols[indices], vals[indices]


def sparse_add_coo(rows_a, cols_a, vals_a,
                   rows_b, cols_b, vals_b,
                   shape: Tuple[int, int]) -> Tuple:
    """
    Add two matrices in pure COO format: C = A + B
    
    Args:
        rows_a, cols_a, vals_a: Matrix A in COO format (NumPy arrays)
        rows_b, cols_b, vals_b: Matrix B in COO format (NumPy arrays)
        shape: Matrix dimensions (nrows, ncols)
    
    Returns:
        (rows_c, cols_c, vals_c) result in COO format (NumPy arrays)
    """
    rows_a_sorted, cols_a_sorted, vals_a_sorted = _sort_coo(rows_a, cols_a, vals_a)
    rows_b_sorted, cols_b_sorted, vals_b_sorted = _sort_coo(rows_b, cols_b, vals_b)
    
    rows_c, cols_c, vals_c = _merge_sorted_coo(
        rows_a_sorted, cols_a_sorted, vals_a_sorted,
        rows_b_sorted, cols_b_sorted, vals_b_sorted
    )
    
    return rows_c, cols_c, vals_c


def verify_coo_sorted(rows, cols):
    """Check if COO matrix is sorted by (row, col)."""
    for i in range(len(rows) - 1):
        if rows[i] > rows[i+1]:
            return False
        if rows[i] == rows[i+1] and cols[i] >= cols[i+1]:
            return False
    return True
