"""
Sparse Matrix Addition (A + B) - Pure COO Format
Implements efficient two-pointer merge algorithm for sorted COO matrices.

Algorithm: Sort-Merge Addition  
1. Both matrices must be sorted by (row, col)
2. Use two pointers to traverse both matrices simultaneously
3. Merge entries: sum if (i,j) matches, otherwise keep individual entries

Time Complexity: O(nnz(A) + nnz(B))
Space Complexity: O(nnz(A) + nnz(B))

Note: This implementation works purely with COO format (no CSR/CSC conversions)
"""

import numpy as np
import numba
import logging
import time
from scipy import sparse as sp
from typing import List, Tuple

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


def sparse_add_coo(coo_a: COOMatrix, coo_b: COOMatrix) -> COOMatrix:
    """
    Add two sparse matrices in COO format: C = A + B
    Uses Numba-accelerated two-pointer merge.
    
    Args:
        coo_a: First matrix (must be sorted by row, col)
        coo_b: Second matrix (must be sorted by row, col)
    
    Returns:
        COOMatrix result
    """
    if coo_a.shape != coo_b.shape:
        raise ValueError(f"Matrix dimensions don't match: {coo_a.shape} vs {coo_b.shape}")
    
    rows_a, cols_a, vals_a = coo_a.to_arrays()
    rows_b, cols_b, vals_b = coo_b.to_arrays()
    
    result_rows, result_cols, result_vals = _merge_sorted_coo(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b
    )
    
    result_data = [(int(result_rows[i]), int(result_cols[i]), float(result_vals[i])) 
                   for i in range(len(result_rows))]
    
    return COOMatrix(shape=coo_a.shape, data=result_data)


def verify_addition_scipy(coo_a: COOMatrix, coo_b: COOMatrix, result: COOMatrix) -> bool:
    """
    Verify addition result against scipy.sparse.
    
    Args:
        coo_a, coo_b: Input matrices
        result: Our result
    
    Returns:
        True if correct
    """
    logger.info("Verifying result against scipy.sparse...")
    
    try:
        scipy_a = coo_a.to_scipy_sparse()
        scipy_b = coo_b.to_scipy_sparse()
        scipy_result = result.to_scipy_sparse()
        
        expected = scipy_a + scipy_b
        
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
