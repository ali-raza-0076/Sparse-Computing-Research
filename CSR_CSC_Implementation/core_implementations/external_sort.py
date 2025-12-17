"""
External Sorting Module for Sparse Matrix Operations
Handles sorting of large CSV files (COO format) that don't fit in RAM.

This module sorts sparse matrix data stored as (row, col, value) triples
by (row, col) to enable efficient merge-based operations.

Uses Numba JIT for acceleration of sorting operations.
"""

import heapq
import tempfile
import os
from pathlib import Path
import logging
import numba
import numpy as np


import numba
import numpy as np


@numba.jit(nopython=True, cache=True)
def _compare_coo_entries(row1, col1, row2, col2):
    """
    Compare two COO entries by (row, col) for sorting.
    Returns: -1 if entry1 < entry2, 0 if equal, 1 if entry1 > entry2
    """
    if row1 < row2:
        return -1
    elif row1 > row2:
        return 1
    else:
        if col1 < col2:
            return -1
        elif col1 > col2:
            return 1
        else:
            return 0


@numba.jit(nopython=True, cache=True)
def _quicksort_coo(rows, cols, values, low, high):
    """
    Numba-accelerated quicksort for COO data by (row, col).
    
    Args:
        rows: Array of row indices
        cols: Array of column indices
        values: Array of values
        low: Starting index
        high: Ending index
    """
    if low < high:
        pivot_row = rows[high]
        pivot_col = cols[high]
        i = low - 1
        
        for j in range(low, high):
            if _compare_coo_entries(rows[j], cols[j], pivot_row, pivot_col) <= 0:
                i += 1
                rows[i], rows[j] = rows[j], rows[i]
                cols[i], cols[j] = cols[j], cols[i]
                values[i], values[j] = values[j], values[i]
        
        rows[i + 1], rows[high] = rows[high], rows[i + 1]
        cols[i + 1], cols[high] = cols[high], cols[i + 1]
        values[i + 1], values[high] = values[high], values[i + 1]
        
        pi = i + 1
        
        _quicksort_coo(rows, cols, values, low, pi - 1)
        _quicksort_coo(rows, cols, values, pi + 1, high)


def sort_coo_arrays(rows, cols, values):
    """
    Sort COO arrays by (row, col) using Numba-accelerated quicksort.
    
    Args:
        rows: NumPy array of row indices
        cols: NumPy array of column indices
        values: NumPy array of values
    
    Returns:
        (rows, cols, values) sorted by (row, col)
    """
    n = len(rows)
    if n <= 1:
        return rows, cols, values
    
    rows = np.ascontiguousarray(rows, dtype=np.int32)
    cols = np.ascontiguousarray(cols, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.int32)
    
    _quicksort_coo(rows, cols, values, 0, n - 1)
    
    return rows, cols, values


class ExternalSorter:
    """
    External merge sort for large CSV files containing sparse matrix data.
    
    Sorts data in COO format: (row, col, value) triples
    Sorting key: (row, col) lexicographically
    
    Process:
    1. Read input file in chunks that fit in RAM
    2. Sort each chunk in memory by (row, col)
    3. Write sorted chunks to temporary files
    4. Merge all sorted chunks using k-way merge (heap-based)
    """
    
    def __init__(self, chunk_size_mb=100, encoding="utf-8", logger=None):
        """
        Args:
            chunk_size_mb: Size of each chunk in megabytes (default 100 MB)
            encoding: File encoding (default utf-8)
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self.encoding = encoding
        self.tmp_files = []
        self.logger = logger or logging.getLogger(__name__)
    
    def _parse_coo_line(self, line):
        """
        Parse a COO format line: row,col,value
        Returns: (row, col, value) tuple for sorting, and original line
        """
        parts = line.strip().split(',')
        if len(parts) != 3:
            return None, line
        
        try:
            row = int(parts[0])
            col = int(parts[1])
            value = int(float(parts[2]))
            return (row, col, value), line
        except (ValueError, IndexError):
            return None, line
    
    def _sort_chunk(self, lines):
        """
        Sort a chunk of lines by (row, col) using Numba-accelerated sorting.
        
        Args:
            lines: List of CSV lines
            
        Returns:
            List of lines sorted by (row, col)
        """
        rows_list = []
        cols_list = []
        values_list = []
        valid_lines = []
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    row = int(parts[0])
                    col = int(parts[1])
                    value = int(float(parts[2]))
                    
                    rows_list.append(row)
                    cols_list.append(col)
                    values_list.append(value)
                    valid_lines.append(line)
                except ValueError:
                    continue
        
        if not rows_list:
            return []
        
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)
        values = np.array(values_list, dtype=np.int32)
        
        rows, cols, values = sort_coo_arrays(rows, cols, values)
        
        sorted_lines = []
        for i in range(len(rows)):
            line = f"{rows[i]},{cols[i]},{values[i]}\n"
            sorted_lines.append(line)
        
        return sorted_lines
    
    def _write_sorted_chunk(self, lines, chunk_number):
        """
        Sort lines and write to a temporary file.
        
        Args:
            lines: List of CSV lines
            chunk_number: Sequential chunk identifier
            
        Returns:
            Path to temporary file
        """
        sorted_lines = self._sort_chunk(lines)
        
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"sparse_sort_chunk_{chunk_number}_",
            suffix=".csv"
        )
        os.close(fd)
        
        tmp = Path(tmp_path)
        
        with open(tmp, "w", encoding=self.encoding) as f:
            for line in sorted_lines:
                if not line.endswith('\n'):
                    f.write(line + '\n')
                else:
                    f.write(line)
        
        self.tmp_files.append(tmp)
        self.logger.info(f"Wrote sorted chunk {chunk_number}: {tmp.name} ({len(sorted_lines)} lines)")
        
        return tmp
    
    def create_sorted_chunks(self, input_file):
        """
        Read input file and create sorted chunks.
        
        Args:
            input_file: Path to input CSV file (COO format)
        """
        self.logger.info(f"Creating sorted chunks from {input_file}")
        
        with open(input_file, "r", encoding=self.encoding) as f:
            buffer = []
            bytes_read = 0
            chunk_number = 0
            
            for line in f:
                buffer.append(line)
                bytes_read += len(line.encode(self.encoding))
                
                if bytes_read >= self.chunk_size:
                    self._write_sorted_chunk(buffer, chunk_number)
                    buffer = []
                    bytes_read = 0
                    chunk_number += 1
            
            if buffer:
                self._write_sorted_chunk(buffer, chunk_number)
        
        self.logger.info(f"Created {len(self.tmp_files)} sorted chunks")
    
    def merge_sorted_chunks(self, output_file):
        """
        Merge all sorted chunks using k-way merge (heap-based).
        
        The heap maintains the smallest (row, col) from each chunk,
        ensuring the output is sorted by (row, col).
        
        Args:
            output_file: Path to output sorted CSV file
        """
        if not self.tmp_files:
            self.logger.warning("No chunks to merge")
            return
        
        self.logger.info(f"Merging {len(self.tmp_files)} sorted chunks into {output_file}")
        
        open_files = [open(f, "r", encoding=self.encoding) for f in self.tmp_files]
        
        heap = []
        for idx, f in enumerate(open_files):
            line = f.readline()
            if line:
                key, _ = self._parse_coo_line(line)
                if key is not None:
                    heapq.heappush(heap, (key[0], key[1], idx, line))
        
        with open(output_file, "w", encoding=self.encoding) as out:
            while heap:
                row, col, file_idx, line = heapq.heappop(heap)
                
                if not line.endswith('\n'):
                    out.write(line + '\n')
                else:
                    out.write(line)
                
                next_line = open_files[file_idx].readline()
                if next_line:
                    key, _ = self._parse_coo_line(next_line)
                    if key is not None:
                        heapq.heappush(heap, (key[0], key[1], file_idx, next_line))
        
        for f in open_files:
            f.close()
        
        self.logger.info(f"Merge complete: {output_file}")
    
    def clean_tmp_files(self):
        """Remove all temporary chunk files."""
        self.logger.info(f"Cleaning up {len(self.tmp_files)} temporary files")
        
        for tmp_file in self.tmp_files:
            try:
                if tmp_file.exists():
                    os.remove(tmp_file)
            except OSError as e:
                self.logger.warning(f"Failed to remove {tmp_file}: {e}")
        
        self.tmp_files.clear()
    
    def sort_file(self, input_file, output_file):
        """
        Complete external sort: create chunks, merge, and clean up.
        
        Args:
            input_file: Path to input CSV file (unsorted COO format)
            output_file: Path to output CSV file (sorted by row, col)
        """
        try:
            self.logger.info(f"Starting external sort: {input_file} -> {output_file}")
            
            self.create_sorted_chunks(input_file)
            
            self.merge_sorted_chunks(output_file)
            
            self.logger.info("External sort complete")
            
        finally:
            self.clean_tmp_files()


def sort_sparse_matrix(input_csv, output_csv, chunk_size_mb=100, logger=None):
    """
    Convenience function to sort a sparse matrix CSV file.
    
    Args:
        input_csv: Path to input file (COO format: row,col,value)
        output_csv: Path to output file (sorted by row, col)
        chunk_size_mb: Chunk size in MB (default 100)
        logger: Optional logger
    
    Returns:
        Path to sorted output file
    """
    sorter = ExternalSorter(chunk_size_mb=chunk_size_mb, logger=logger)
    sorter.sort_file(input_csv, output_csv)
    return output_csv
