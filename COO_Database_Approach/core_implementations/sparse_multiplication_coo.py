"""
Sparse Matrix Multiplication (A × B) - Pure COO Format
Implements COO×COO multiplication using sort-merge algorithm (Database approach).

Algorithm: COO×COO Sort-Merge Multiplication
1. Sort matrix A by column (secondary sort by row for stability)
2. Sort matrix B by row (secondary sort by column for stability)
3. Use two-pointer merge: match A's columns with B's rows
4. For each match (A[i,k] and B[k,j]), accumulate result[i,j] += a_val * b_val

Time Complexity: O(nnz(A)log(nnz(A)) + nnz(B)log(nnz(B)) + nnz(A) + nnz(B))
Space Complexity: O(nnz(C))

Note: This is the database-style external merge algorithm.
Preferred for disk-based operations where sorting is efficient.
"""

import numpy as np
import numba
import logging
import time
from scipy import sparse as sp
from typing import List, Tuple, Dict
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class COOMatrix:
    """Simple COO matrix representation for pure COO operations."""
    
    def __init__(self, shape: Tuple[int, int], data: List[Tuple[int, int, float]]):
        """
        Initialize COO matrix.
        
        Args:
            shape: (num_rows, num_cols)
            data: List of (row, col, value) tuples
        """
        self.shape = shape
        self.data = data
    
    def to_arrays(self):
        """Convert to NumPy arrays."""
        if not self.data:
            return (np.array([], dtype=np.int32), 
                    np.array([], dtype=np.int32), 
                    np.array([], dtype=np.float64))
        
        rows, cols, vals = zip(*self.data)
        return (np.array(rows, dtype=np.int32), 
                np.array(cols, dtype=np.int32), 
                np.array(vals, dtype=np.float64))
    
    def to_scipy_sparse(self):
        """Convert to scipy.sparse.coo_matrix for verification."""
        rows, cols, vals = self.to_arrays()
        return sp.coo_matrix((vals, (rows, cols)), shape=self.shape)
    
    def count_nnz(self):
        """Count non-zero entries."""
        return len(self.data)


@numba.jit(nopython=True, cache=True)
def _multiply_coo_coo_numba(rows_a, cols_a, vals_a, rows_b, cols_b, vals_b, num_rows, num_cols):
    """
    Multiply two COO matrices using sort-merge algorithm (Database approach).
    
    Algorithm:
    1. A is already sorted by column (then row)
    2. B is already sorted by row (then column)
    3. Use two-pointer merge: match A's columns with B's rows
    4. Accumulate products in hash map
    
    Args:
        rows_a, cols_a, vals_a: Matrix A sorted by (col, row)
        rows_b, cols_b, vals_b: Matrix B sorted by (row, col)
        num_rows: Number of rows in result
        num_cols: Number of columns in result
    
    Returns:
        (result_rows, result_cols, result_vals)
    """
    result_dict = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    
    n_a = len(rows_a)
    n_b = len(rows_b)
    
    a_idx = 0
    
    while a_idx < n_a:
        k = cols_a[a_idx]
        
        a_start = a_idx
        while a_idx < n_a and cols_a[a_idx] == k:
            a_idx += 1
        a_end = a_idx
        
        b_idx = 0
        while b_idx < n_b and rows_b[b_idx] < k:
            b_idx += 1
        
        if b_idx >= n_b or rows_b[b_idx] != k:
            continue
        
        b_start = b_idx
        while b_idx < n_b and rows_b[b_idx] == k:
            b_idx += 1
        b_end = b_idx
        
        for a_pos in range(a_start, a_end):
            i = rows_a[a_pos]
            a_val = vals_a[a_pos]
            
            for b_pos in range(b_start, b_end):
                j = cols_b[b_pos]
                b_val = vals_b[b_pos]
                
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


def sparse_multiply_coo(coo_a: COOMatrix, coo_b: COOMatrix) -> COOMatrix:
    """
    Multiply sparse matrices: C = A × B
    Uses sort-merge algorithm (database approach).
    
    Algorithm:
    1. Sort A by column (then row)
    2. Sort B by row (then column)
    3. Merge where A's columns match B's rows
    
    Args:
        coo_a: Matrix A in COO format
        coo_b: Matrix B in COO format
    
    Returns:
        COOMatrix result
    """
    if coo_a.shape[1] != coo_b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A is {coo_a.shape}, B is {coo_b.shape}. "
            f"A's columns ({coo_a.shape[1]}) must equal B's rows ({coo_b.shape[0]})"
        )
    
    result_shape = (coo_a.shape[0], coo_b.shape[1])
    
    rows_a, cols_a, vals_a = coo_a.to_arrays()
    rows_b, cols_b, vals_b = coo_b.to_arrays()
    
    logger.info("Sorting A by column...")
    a_order = np.lexsort((rows_a, cols_a))
    rows_a = rows_a[a_order]
    cols_a = cols_a[a_order]
    vals_a = vals_a[a_order]
    
    logger.info("Sorting B by row...")
    b_order = np.lexsort((cols_b, rows_b))
    rows_b = rows_b[b_order]
    cols_b = cols_b[b_order]
    vals_b = vals_b[b_order]
    
    logger.info("Performing sort-merge multiplication...")
    result_rows, result_cols, result_vals = _multiply_coo_coo_numba(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b,
        result_shape[0], result_shape[1]
    )
    
    result_data = [(int(result_rows[i]), int(result_cols[i]), float(result_vals[i])) 
                   for i in range(len(result_rows))]
    
    return COOMatrix(shape=result_shape, data=result_data)


def verify_multiplication_scipy(coo_a: COOMatrix, coo_b: COOMatrix, result: COOMatrix) -> bool:
    """
    Verify multiplication result against scipy.sparse.
    
    Args:
        coo_a, coo_b: Input matrices
        result: Our result
    
    Returns:
        True if correct
    """
    logger.info("Verifying result against scipy.sparse...")
    
    try:
        scipy_a = coo_a.to_scipy_sparse().tocsr()
        scipy_b = coo_b.to_scipy_sparse().tocsc()
        scipy_result = result.to_scipy_sparse()
        
        expected = scipy_a @ scipy_b
        
        diff = scipy_result - expected
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        
        if max_diff > 1e-9:
            logger.error(f"✗ Verification failed: max difference = {max_diff}")
            return False
        
        logger.info("✓ Verification passed! Result matches scipy.sparse")
        return True
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False
