"""
Sparse Matrix Addition (A + B) using CSR Format
Implements efficient row-by-row addition for CSR matrices.

Algorithm: CSR Row-by-Row Addition
1. For each row i, merge the entries from A[i,:] and B[i,:]
2. Use two-pointer technique within each row
3. Build result CSR directly

Time Complexity: O(nnz(A) + nnz(B))
Space Complexity: O(nnz(A) + nnz(B))
"""

import numpy as np
import numba
import logging
import csv
from pathlib import Path
from typing import Tuple
import time

from matrix_formats import CSRMatrix, COOMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, cache=True)
def _add_csr_rows(row_ptr_a, col_idx_a, vals_a,
                  row_ptr_b, col_idx_b, vals_b,
                  nrows):
    """
    Add two CSR matrices row by row.
    
    Args:
        row_ptr_a, col_idx_a, vals_a: Matrix A in CSR format
        row_ptr_b, col_idx_b, vals_b: Matrix B in CSR format
        nrows: Number of rows
    
    Returns:
        (row_ptr_c, col_idx_c, vals_c) for result matrix C
    """
    max_nnz = len(col_idx_a) + len(col_idx_b)
    col_idx_c = np.empty(max_nnz, dtype=np.int32)
    vals_c = np.empty(max_nnz, dtype=vals_a.dtype)
    row_ptr_c = np.zeros(nrows + 1, dtype=np.int64)
    
    nnz_c = 0
    
    for row in range(nrows):
        start_a = row_ptr_a[row]
        end_a = row_ptr_a[row + 1]
        start_b = row_ptr_b[row]
        end_b = row_ptr_b[row + 1]
        
        i, j = start_a, start_b
        
        while i < end_a and j < end_b:
            col_a = col_idx_a[i]
            col_b = col_idx_b[j]
            
            if col_a < col_b:
                col_idx_c[nnz_c] = col_a
                vals_c[nnz_c] = vals_a[i]
                nnz_c += 1
                i += 1
            elif col_a > col_b:
                col_idx_c[nnz_c] = col_b
                vals_c[nnz_c] = vals_b[j]
                nnz_c += 1
                j += 1
            else:
                col_idx_c[nnz_c] = col_a
                vals_c[nnz_c] = vals_a[i] + vals_b[j]
                nnz_c += 1
                i += 1
                j += 1
        
        while i < end_a:
            col_idx_c[nnz_c] = col_idx_a[i]
            vals_c[nnz_c] = vals_a[i]
            nnz_c += 1
            i += 1
        
        while j < end_b:
            col_idx_c[nnz_c] = col_idx_b[j]
            vals_c[nnz_c] = vals_b[j]
            nnz_c += 1
            j += 1
        
        row_ptr_c[row + 1] = nnz_c
    
    return row_ptr_c, col_idx_c[:nnz_c], vals_c[:nnz_c]


def sparse_add_csr(csr_a: CSRMatrix, csr_b: CSRMatrix) -> CSRMatrix:
    """
    Add two sparse matrices in CSR format: C = A + B
    Uses Numba-accelerated row-by-row addition.
    
    Args:
        csr_a: First matrix in CSR format
        csr_b: Second matrix in CSR format
    
    Returns:
        CSRMatrix result
    """
    if csr_a.shape != csr_b.shape:
        raise ValueError(f"Matrix dimensions don't match: {csr_a.shape} vs {csr_b.shape}")
    
    start = time.time()
    
    row_ptr_c, col_idx_c, vals_c = _add_csr_rows(
        csr_a.row_ptr, csr_a.col_idx, csr_a.values,
        csr_b.row_ptr, csr_b.col_idx, csr_b.values,
        csr_a.shape[0]
    )
    
    elapsed = time.time() - start
    
    result = CSRMatrix(csr_a.shape, row_ptr_c, col_idx_c, vals_c)
    
    return result


def sparse_add_from_coo(rows_a, cols_a, vals_a,
                        rows_b, cols_b, vals_b,
                        shape: Tuple[int, int]) -> Tuple:
    """
    Add two matrices given as COO arrays by converting to CSR.
    
    Args:
        rows_a, cols_a, vals_a: Matrix A in COO format (NumPy arrays)
        rows_b, cols_b, vals_b: Matrix B in COO format (NumPy arrays)
        shape: Matrix dimensions (nrows, ncols)
    
    Returns:
        (rows_c, cols_c, vals_c) result in COO format (NumPy arrays)
    """
    data_a = [(int(r), int(c), float(v)) for r, c, v in zip(rows_a, cols_a, vals_a)]
    coo_a = COOMatrix(shape=shape, data=data_a)
    csr_a = coo_a.to_csr()
    
    data_b = [(int(r), int(c), float(v)) for r, c, v in zip(rows_b, cols_b, vals_b)]
    coo_b = COOMatrix(shape=shape, data=data_b)
    csr_b = coo_b.to_csr()
    
    csr_c = sparse_add_csr(csr_a, csr_b)
    
    rows_c, cols_c, vals_c = [], [], []
    for row in range(csr_c.shape[0]):
        start = csr_c.row_ptr[row]
        end = csr_c.row_ptr[row + 1]
        for idx in range(start, end):
            rows_c.append(row)
            cols_c.append(csr_c.col_idx[idx])
            vals_c.append(csr_c.values[idx])
    
    return np.array(rows_c, dtype=np.int32), np.array(cols_c, dtype=np.int32), np.array(vals_c, dtype=np.float64)


def sparse_add_coo(coo_a: COOMatrix, coo_b: COOMatrix) -> COOMatrix:
    """
    Add two COOMatrix objects by converting to CSR internally.
    Convenience wrapper for benchmark compatibility.
    
    Args:
        coo_a: First matrix as COOMatrix
        coo_b: Second matrix as COOMatrix
    
    Returns:
        COOMatrix result
    """
    rows_a = np.array([entry[0] for entry in coo_a.data], dtype=np.int32)
    cols_a = np.array([entry[1] for entry in coo_a.data], dtype=np.int32)
    vals_a = np.array([entry[2] for entry in coo_a.data], dtype=np.float64)
    
    rows_b = np.array([entry[0] for entry in coo_b.data], dtype=np.int32)
    cols_b = np.array([entry[1] for entry in coo_b.data], dtype=np.int32)
    vals_b = np.array([entry[2] for entry in coo_b.data], dtype=np.float64)
    
    rows_c, cols_c, vals_c = sparse_add_from_coo(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b,
        coo_a.shape
    )
    
    data_c = [(int(r), int(c), float(v)) for r, c, v in zip(rows_c, cols_c, vals_c)]
    return COOMatrix(shape=coo_a.shape, data=data_c)


def sparse_add_from_files(file_a: str, file_b: str, output_file: str, shape: Tuple[int, int]):
    """
    Add two sparse matrices from CSV files.
    
    Args:
        file_a: Path to first matrix CSV file (COO format)
        file_b: Path to second matrix CSV file (COO format)
        output_file: Path to output CSV file
        shape: Matrix dimensions (nrows, ncols)
    """
    logger.info(f"Loading matrix A from {file_a}...")
    rows_a, cols_a, vals_a = [], [], []
    with open(file_a, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 3:
                rows_a.append(int(line[0]))
                cols_a.append(int(line[1]))
                vals_a.append(int(line[2]))
    
    logger.info(f"Loading matrix B from {file_b}...")
    rows_b, cols_b, vals_b = [], [], []
    with open(file_b, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 3:
                rows_b.append(int(line[0]))
                cols_b.append(int(line[1]))
                vals_b.append(int(line[2]))
    
    rows_c, cols_c, vals_c = sparse_add_from_coo(
        np.array(rows_a, dtype=np.int32),
        np.array(cols_a, dtype=np.int32),
        np.array(vals_a, dtype=np.int32),
        np.array(rows_b, dtype=np.int32),
        np.array(cols_b, dtype=np.int32),
        np.array(vals_b, dtype=np.int32),
        shape
    )
    
    logger.info(f"Writing result to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in zip(rows_c, cols_c, vals_c):
            writer.writerow([i, j, v])
    
    logger.info(f"✓ Sparse addition complete: {output_file}")
