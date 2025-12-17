"""
Parallel Sparse Matrix Addition using CSR Format
Multi-core CPU parallelization with row-block decomposition

Key Features:
- Parallel row-block processing using multiprocessing
- CSR format for efficient row operations
- Minimal overhead with process pools

Usage:
    from sparse_addition_parallel_csr import add_matrices_parallel_csr
    
    add_matrices_parallel_csr(
        file_a='data/matrix_a.csv',
        file_b='data/matrix_b.csv',
        output_file='data/sum.csv',
        shape=(1000, 1000),
        num_workers=16
    )
"""

import csv
import numpy as np
import logging
import multiprocessing as mp
import time
import os
import tempfile
from typing import Tuple

from sparse_addition_csr import _add_csr_rows
from matrix_formats import CSRMatrix, COOMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _process_row_block(start_row, end_row,
                      row_ptr_a, col_idx_a, vals_a,
                      row_ptr_b, col_idx_b, vals_b):
    """
    Process a block of rows in parallel.
    
    Args:
        start_row, end_row: Row range to process
        row_ptr_a, col_idx_a, vals_a: Matrix A in CSR
        row_ptr_b, col_idx_b, vals_b: Matrix B in CSR
    
    Returns:
        (rows, cols, vals) for this block in COO format
    """
    rows, cols, vals = [], [], []
    
    for row in range(start_row, end_row):
        start_a = row_ptr_a[row]
        end_a = row_ptr_a[row + 1]
        start_b = row_ptr_b[row]
        end_b = row_ptr_b[row + 1]
        
        i, j = start_a, start_b
        
        while i < end_a and j < end_b:
            col_a = col_idx_a[i]
            col_b = col_idx_b[j]
            
            if col_a < col_b:
                rows.append(row)
                cols.append(col_a)
                vals.append(vals_a[i])
                i += 1
            elif col_a > col_b:
                rows.append(row)
                cols.append(col_b)
                vals.append(vals_b[j])
                j += 1
            else:
                rows.append(row)
                cols.append(col_a)
                vals.append(vals_a[i] + vals_b[j])
                i += 1
                j += 1
        
        while i < end_a:
            rows.append(row)
            cols.append(col_idx_a[i])
            vals.append(vals_a[i])
            i += 1
        
        while j < end_b:
            rows.append(row)
            cols.append(col_idx_b[j])
            vals.append(vals_b[j])
            j += 1
    
    return rows, cols, vals


def add_matrices_parallel_csr(file_a: str, file_b: str, output_file: str,
                              shape: Tuple[int, int], num_workers: int = 16):
    """
    Add two sparse matrices in parallel using CSR format.
    
    Args:
        file_a: Path to first matrix CSV (COO format)
        file_b: Path to second matrix CSV (COO format)
        output_file: Path to output CSV
        shape: Matrix dimensions (nrows, ncols)
        num_workers: Number of parallel workers
    
    Returns:
        Total execution time
    """
    logger.info("="*70)
    logger.info("PARALLEL Sparse Matrix Addition - BENCHMARK MODE")
    logger.info(f"Workers: {num_workers} CPU cores")
    logger.info("="*70)
    
    total_start = time.time()
    
    logger.info("Loading matrices...")
    load_start = time.time()
    
    rows_a, cols_a, vals_a = [], [], []
    with open(file_a, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 3:
                rows_a.append(int(line[0]))
                cols_a.append(int(line[1]))
                vals_a.append(float(line[2]))
    
    rows_b, cols_b, vals_b = [], [], []
    with open(file_b, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) >= 3:
                rows_b.append(int(line[0]))
                cols_b.append(int(line[1]))
                vals_b.append(float(line[2]))
    
    load_time = time.time() - load_start
    logger.info(f"Matrix 1: {len(rows_a):,} nonzeros")
    logger.info(f"Matrix 2: {len(rows_b):,} nonzeros")
    logger.info(f"Total input: {len(rows_a) + len(rows_b):,} entries")
    logger.info(f"Loading took: {load_time:.3f}s")
    
    logger.info("Converting to CSR format...")
    conv_start = time.time()
    
    data_a = [(int(r), int(c), float(v)) for r, c, v in zip(rows_a, cols_a, vals_a)]
    coo_a = COOMatrix(shape=shape, data=data_a)
    csr_a = coo_a.to_csr()
    
    data_b = [(int(r), int(c), float(v)) for r, c, v in zip(rows_b, cols_b, vals_b)]
    coo_b = COOMatrix(shape=shape, data=data_b)
    csr_b = coo_b.to_csr()
    
    conv_time = time.time() - conv_start
    logger.info(f"Conversion took: {conv_time:.3f}s")
    
    logger.info("")
    logger.info("="*70)
    logger.info("PARALLEL PROCESSING PHASE")
    logger.info("="*70)
    
    compute_start = time.time()
    
    nrows = shape[0]
    rows_per_worker = (nrows + num_workers - 1) // num_workers
    
    tasks = []
    for i in range(num_workers):
        start_row = i * rows_per_worker
        end_row = min((i + 1) * rows_per_worker, nrows)
        if start_row < nrows:
            tasks.append((start_row, end_row,
                         csr_a.row_ptr, csr_a.col_idx, csr_a.values,
                         csr_b.row_ptr, csr_b.col_idx, csr_b.values))
    
    logger.info(f"Processing {nrows} rows across {len(tasks)} workers...")
    logger.info("")
    
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(_process_row_block, tasks)
    
    compute_time = time.time() - compute_start
    logger.info(f"Parallel processing completed in: {compute_time:.3f}s")
    
    logger.info("")
    logger.info("="*70)
    logger.info("MERGING PHASE")
    logger.info("="*70)
    
    merge_start = time.time()
    
    all_rows, all_cols, all_vals = [], [], []
    for rows, cols, vals in results:
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_vals.extend(vals)
    
    merge_time = time.time() - merge_start
    logger.info(f"Merging took: {merge_time:.3f}s")
    
    logger.info("Writing result...")
    write_start = time.time()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, j, v in zip(all_rows, all_cols, all_vals):
            writer.writerow([i, j, v])
    
    write_time = time.time() - write_start
    logger.info(f"Writing took: {write_time:.3f}s")
    
    total_time = time.time() - total_start
    
    logger.info("")
    logger.info("="*70)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Total execution time:        {total_time:.3f}s")
    logger.info(f"  - Loading matrices:        {load_time:.3f}s ({100*load_time/total_time:.1f}%)")
    logger.info(f"  - CSR conversion:          {conv_time:.3f}s ({100*conv_time/total_time:.1f}%)")
    logger.info(f"  - Parallel processing:     {compute_time:.3f}s ({100*compute_time/total_time:.1f}%)")
    logger.info(f"  - Merging results:         {merge_time:.3f}s ({100*merge_time/total_time:.1f}%)")
    logger.info(f"  - Writing output:          {write_time:.3f}s ({100*write_time/total_time:.1f}%)")
    logger.info("")
    logger.info(f"Parallelization configuration:")
    logger.info(f"  - Number of CPU cores:     {num_workers}")
    logger.info("="*70)
    logger.info(f"✓ Parallel addition complete: {output_file}")
    logger.info("="*70)
    
    return total_time
