"""
Sparse Matrix Multiplication (A ├ù B)
Implements efficient CSR × CSC sort-merge algorithm (Database approach).

Algorithm: CSR × CSC Sort-Merge Two-Pointer
1. Convert A to CSR (sorts A by row, then column)
2. Convert B to CSC (sorts B by column, then row)
3. For each (i,j): compute dot product using two-pointer merge
   - A[i,:] has columns sorted (from CSR)
   - B[:,j] has rows sorted (from CSC)
   - Merge where A's columns match B's rows
4. Accumulate products where indices align

Time Complexity: O(nnz(A)log(nnz(A)) + nnz(B)log(nnz(B)) + nnz(A) + nnz(B) + nnz(C))
Best case: O(nnz(A) + nnz(B)) when result is sparse
Space Complexity: O(nnz(A) + nnz(B) + nnz(C))

Note: This is the database-style external merge algorithm.
CSR/CSC formats naturally provide the required sorting.
"""

import numpy as np
import numba
import logging
from pathlib import Path
from typing import Tuple
import time
from scipy import sparse as sp

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, cache=True)
def _sparse_dot_product(indices1, vals1, indices2, vals2):
    """
    Compute dot product of two sparse vectors using two-pointer merge.
    This is the core of the sort-merge algorithm.
    
    indices1: A's column indices (sorted from CSR)
    vals1: A's values
    indices2: B's row indices (sorted from CSC)  
    vals2: B's values
    
    Merge where indices match: A's columns == B's rows
    
    Args:
        indices1, vals1: First sparse vector (sorted indices)
        indices2, vals2: Second sparse vector (sorted indices)
    
    Returns:
        Dot product value
    """
    result = 0
    i, j = 0, 0
    n1, n2 = len(indices1), len(indices2)
    
    while i < n1 and j < n2:
        if indices1[i] < indices2[j]:
            i += 1
        elif indices1[i] > indices2[j]:
            j += 1
        else:
            result += vals1[i] * vals2[j]
            i += 1
            j += 1
    
    return result


@numba.jit(nopython=True, cache=True)
def _multiply_csr_csc_numba(
    a_row_ptr, a_col_idx, a_values,
    b_col_ptr, b_row_idx, b_values,
    num_rows_a, num_cols_b
):
    """
    Multiply CSR matrix A by CSC matrix B using Numba.
    
    Returns arrays for result in COO format.
    """
    max_nnz = min(num_rows_a * num_cols_b, 10000000)
    result_rows = np.zeros(max_nnz, dtype=np.int32)
    result_cols = np.zeros(max_nnz, dtype=np.int32)
    result_vals = np.zeros(max_nnz, dtype=np.float64)
    
    count = 0
    
    for i in range(num_rows_a):
        a_start = a_row_ptr[i]
        a_end = a_row_ptr[i + 1]
        
        if a_start == a_end:  # Empty row
            continue
        
        a_cols = a_col_idx[a_start:a_end]
        a_vals = a_values[a_start:a_end]
        
        for k in range(num_cols_b):
            b_start = b_col_ptr[k]
            b_end = b_col_ptr[k + 1]
            
            if b_start == b_end:  # Empty column
                continue
            
            b_rows = b_row_idx[b_start:b_end]
            b_vals = b_values[b_start:b_end]
            
            dot = _sparse_dot_product(a_cols, a_vals, b_rows, b_vals)
            
            if dot != 0:
                if count >= max_nnz:
                    break
                result_rows[count] = i
                result_cols[count] = k
                result_vals[count] = dot
                count += 1
    
    return result_rows[:count], result_cols[:count], result_vals[:count]


def sparse_multiply(csr_a: CSRMatrix, csc_b: CSCMatrix, output_file: str = None) -> COOMatrix:
    """
    Multiply sparse matrices: C = A x B
    Uses CSR(A) x CSC(B) sort-merge algorithm (database approach).
    
    Algorithm:
    - CSR(A) provides rows with sorted columns
    - CSC(B) provides columns with sorted rows
    - Two-pointer merge matches A's columns with B's rows
    
    Args:
        csr_a: Matrix A in CSR format (sorted by row, then column)
        csc_b: Matrix B in CSC format (sorted by column, then row)
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info(f"Sparse multiplication (sort-merge): A({csr_a.shape}) x B({csc_b.shape})")
    
    if csr_a.shape[1] != csc_b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A is {csr_a.shape}, B is {csc_b.shape}. "
            f"A's columns ({csr_a.shape[1]}) must equal B's rows ({csc_b.shape[0]})"
        )
    
    result_shape = (csr_a.shape[0], csc_b.shape[1])
    
    logger.info(f"A: {csr_a.nnz():,} nonzeros (CSR: sorted by row)")
    logger.info(f"B: {csc_b.nnz():,} nonzeros (CSC: sorted by column)")
    logger.info(f"Result will be {result_shape[0]} x {result_shape[1]}")
    
    logger.info("Running CSR x CSC sort-merge multiplication (Numba-accelerated)...")
    start = time.time()
    
    result_rows, result_cols, result_vals = _multiply_csr_csc_numba(
        csr_a.row_ptr, csr_a.col_idx, csr_a.values,
        csc_b.col_ptr, csc_b.row_idx, csc_b.values,
        csr_a.shape[0], csc_b.shape[1]
    )
    
    elapsed = time.time() - start
    
    logger.info(f"[OK] Multiplication complete in {elapsed:.4f}s")
    logger.info(f"Result has {len(result_rows):,} nonzeros")
    
    result_data = [(int(result_rows[i]), int(result_cols[i]), int(result_vals[i])) 
                   for i in range(len(result_rows))]
    
    result_coo = COOMatrix(shape=result_shape, data=result_data)
    
    if output_file:
        logger.info(f"Writing result to {output_file}...")
        result_coo.to_csv(output_file)
    
    return result_coo


def sparse_multiply_from_coo(coo_a: COOMatrix, coo_b: COOMatrix, output_file: str = None) -> COOMatrix:
    """
    Multiply sparse matrices given as COO format.
    Automatically converts to CSR/CSC and performs multiplication.
    
    Args:
        coo_a: Matrix A in COO format
        coo_b: Matrix B in COO format
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info("Converting matrices to CSR/CSC...")
    
    logger.info("DEBUG: Starting A ΓåÆ CSR conversion...")
    start = time.time()
    csr_a = coo_a.to_csr()
    logger.info("DEBUG: Returned from to_csr()!")
    elapsed = time.time() - start
    logger.info(f"A ΓåÆ CSR: {elapsed:.4f}s")
    
    from matrix_formats import build_csr_from_coo
    csr_a = build_csr_from_coo(coo_a)
    
    logger.info("DEBUG: CSR built successfully")

    logger.info("DEBUG: Starting B ΓåÆ CSC conversion...")  # ΓåÉ ADD THIS
    start = time.time()
    csc_b = coo_b.to_csc()
    logger.info(f"B ΓåÆ CSC: {time.time() - start:.4f}s")
    
    logger.info("DEBUG: Starting multiplication...")  # ΓåÉ ADD THIS
    return sparse_multiply(csr_a, csc_b, output_file=output_file)


def sparse_multiply_blocked(
    csr_a: CSRMatrix, 
    csc_b: CSCMatrix, 
    block_size: int = 1000,
    output_file: str = None
) -> COOMatrix:
    """
    Blocked sparse multiplication for very large matrices.
    Processes result in blocks to limit memory usage.
    
    Args:
        csr_a: Matrix A in CSR format
        csc_b: Matrix B in CSC format
        block_size: Number of rows to process at once
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info(f"Blocked multiplication: block_size={block_size}")
    
    if csr_a.shape[1] != csc_b.shape[0]:
        raise ValueError(f"Incompatible dimensions")
    
    result_shape = (csr_a.shape[0], csc_b.shape[1])
    num_blocks = (csr_a.shape[0] + block_size - 1) // block_size
    
    all_rows = []
    all_cols = []
    all_vals = []
    
    for block_idx in range(num_blocks):
        row_start = block_idx * block_size
        row_end = min(row_start + block_size, csr_a.shape[0])
        
        logger.info(f"Processing block {block_idx + 1}/{num_blocks}: rows {row_start}-{row_end}")
        
        a_block = csr_a.get_row_block(row_start, row_end)
        
        block_rows, block_cols, block_vals = _multiply_csr_csc_numba(
            a_block.row_ptr, a_block.col_idx, a_block.values,
            csc_b.col_ptr, csc_b.row_idx, csc_b.values,
            a_block.shape[0], csc_b.shape[1]
        )
        
        for r in block_rows:
            all_rows.append(r + row_start)
        all_cols.extend(block_cols)
        all_vals.extend(block_vals)
    
    logger.info(f"Γ£ô Blocked multiplication complete: {len(all_rows):,} nonzeros")
    
    result_data = [(all_rows[i], all_cols[i], all_vals[i]) for i in range(len(all_rows))]
    result_coo = COOMatrix(shape=result_shape, data=result_data)
    
    if output_file:
        result_coo.to_csv(output_file)
    
    return result_coo


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
            logger.error(f"Γ£ù Verification failed: max difference = {max_diff}")
            return False
        
        logger.info("Γ£ô Verification passed! Result matches scipy.sparse")
        return True
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False
