"""
Parallel Sparse Matrix Addition Module
Multi-core CPU parallelization for faster sparse matrix addition

Key Features:
- Parallel chunk processing using multiprocessing
- Reuses optimized merge algorithm from sparse_addition.py
- Minimal overhead with process pools
- Compatible with existing sparse_addition.py

Usage:
    from sparse_addition_parallel import add_matrices_parallel
    
    add_matrices_parallel(
        file_a='data/output/matrix_a.csv',
        file_b='data/output/matrix_b.csv',
        output_file='data/output/sum.csv',
        num_workers=8  # Use 8 CPU cores
    )
"""

import csv
import heapq
from typing import Iterator, Tuple, List, Optional
import os
import tempfile
import logging
import multiprocessing as mp
from functools import partial
import itertools
import numpy as np
import time
import json
from datetime import datetime

from sparse_addition import _merge_sorted_coo


def _process_chunk_pair(chunk_id: int,
                       file1: str, start1: int, size1: int,
                       file2: str, start2: int, size2: int,
                       temp_dir: str) -> str:
    """
    Worker function to process a pair of chunks in parallel.
    Uses the existing merge function from sparse_addition.py
    
    Args:
        chunk_id: Chunk identifier
        file1: Path to first input file
        start1, size1: Start line and size for first chunk
        file2: Path to second input file
        start2, size2: Start line and size for second chunk
        temp_dir: Directory for temporary files
    
    Returns:
        Path to temporary file containing merged chunk
    """
    rows1, cols1, vals1 = _read_chunk_as_arrays(file1, start1, size1)
    rows2, cols2, vals2 = _read_chunk_as_arrays(file2, start2, size2)
    
    if len(rows1) == 0 and len(rows2) == 0:
        return None
    
    result_rows, result_cols, result_vals = _merge_sorted_coo(
        rows1, cols1, vals1,
        rows2, cols2, vals2
    )
    
    if len(result_rows) == 0:
        return None
    
    temp_file = os.path.join(temp_dir, f'chunk_{chunk_id:04d}.csv')
    with open(temp_file, 'w', newline='') as f:
        for idx in range(len(result_rows)):
            f.write(f"{result_rows[idx]},{result_cols[idx]},{result_vals[idx]}\n")
    
    return temp_file


def _read_chunk_as_iterator(filepath: str, start_line: int, num_lines: int) -> Iterator[Tuple[int, int, float]]:
    """
    Read a chunk of entries from a CSV file as an iterator.
    
    Args:
        filepath: Path to input file
        start_line: Starting line number (0-indexed, skips header)
        num_lines: Number of lines to read
    
    Yields:
        (row, col, value) tuples
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        for _ in range(start_line):
            try:
                next(reader)
            except StopIteration:
                return
        
        for _ in range(num_lines):
            try:
                row_data = next(reader)
                yield (int(row_data['row']), int(row_data['col']), float(row_data['value']))
            except StopIteration:
                break


def _read_chunk_as_arrays(filepath: str, start_line: int, num_lines: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a chunk of entries from a CSV file into numpy arrays.
    
    Args:
        filepath: Path to input file
        start_line: Starting line number (0-indexed)
        num_lines: Number of lines to read
    
    Returns:
        (rows, cols, vals) as numpy arrays
    """
    rows, cols, vals = [], [], []
    
    with open(filepath, 'r') as f:
        for _ in range(start_line):
            try:
                next(f)
            except StopIteration:
                break
        
        for _ in range(num_lines):
            try:
                line = next(f)
                parts = line.strip().split(',')
                if len(parts) == 3:
                    rows.append(int(parts[0]) - 1)
                    cols.append(int(parts[1]) - 1)
                    vals.append(float(parts[2]))
            except StopIteration:
                break
    
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.int32)
    )


class ParallelSparseAddition:
    """
    Parallel sparse matrix addition using multi-core CPU.
    
    Processes chunks in parallel then merges results.
    """
    
    def __init__(self,
                 chunk_size: int = 100000,
                 num_workers: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            chunk_size: Number of entries per chunk
            num_workers: Number of parallel workers (default: CPU count)
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.num_workers = num_workers or mp.cpu_count()
        self.logger = logger or logging.getLogger(__name__)
        self.benchmarks = {
            'total_time': 0,
            'count_time': 0,
            'parallel_processing_time': 0,
            'merge_time': 0,
            'num_workers': self.num_workers,
            'chunk_size': chunk_size
        }
    
    def add_matrices(self, file1: str, file2: str, output_file: str) -> str:
        """
        Parallel addition algorithm.
        
        Args:
            file1: Path to first matrix (COO CSV, sorted)
            file2: Path to second matrix (COO CSV, sorted)
            output_file: Path for result matrix
        
        Returns:
            Path to output file
        """
        start_total = time.time()
        
        self.logger.info(f"="*70)
        self.logger.info(f"PARALLEL Sparse Matrix Addition - BENCHMARK MODE")
        self.logger.info(f"Workers: {self.num_workers} CPU cores")
        self.logger.info(f"Chunk size: {self.chunk_size} entries")
        self.logger.info(f"="*70)
        
        self.logger.info("Counting matrix entries...")
        start_count = time.time()
        size1 = self._count_lines(file1)
        size2 = self._count_lines(file2)
        self.benchmarks['count_time'] = time.time() - start_count
        
        self.logger.info(f"Matrix 1: {size1:,} nonzeros")
        self.logger.info(f"Matrix 2: {size2:,} nonzeros")
        self.logger.info(f"Total input: {size1 + size2:,} entries")
        self.logger.info(f"Counting took: {self.benchmarks['count_time']:.3f}s")
        
        if size1 + size2 <= self.chunk_size:
            self.logger.info("Small matrices - using direct merge...")
            start_merge = time.time()
            self._add_direct(file1, file2, output_file)
            self.benchmarks['parallel_processing_time'] = time.time() - start_merge
            self.benchmarks['total_time'] = time.time() - start_total
            self._save_benchmark_report(output_file, size1, size2)
            return output_file
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"PARALLEL PROCESSING PHASE")
        self.logger.info(f"{'='*70}")
        start_parallel = time.time()
        temp_files = self._process_chunks_parallel(file1, size1, file2, size2)
        self.benchmarks['parallel_processing_time'] = time.time() - start_parallel
        
        self.logger.info(f"\nParallel processing completed in: {self.benchmarks['parallel_processing_time']:.3f}s")
        self.logger.info(f"Speedup potential: {self.num_workers}x (with {self.num_workers} workers)")
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"MERGING PHASE")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Merging {len(temp_files)} chunks...")
        start_merge = time.time()
        self._merge_chunks(temp_files, output_file)
        self.benchmarks['merge_time'] = time.time() - start_merge
        
        self.benchmarks['total_time'] = time.time() - start_total
        
        self._print_summary()
        
        self._save_benchmark_report(output_file, size1, size2)
        
        self.logger.info(f"\nΓ£ô Parallel addition complete: {output_file}")
        self.logger.info(f"="*70)
        
        return output_file
    
    def _count_lines(self, filepath: str) -> int:
        """Count number of entries in file (no header)."""
        count = 0
        with open(filepath, 'r') as f:
            for _ in f:
                count += 1
        return count
    
    def _add_direct(self, file1: str, file2: str, output_file: str):
        """Direct merge for small matrices using existing algorithm."""
        rows1, cols1, vals1 = _read_chunk_as_arrays(file1, 0, 999999999)
        rows2, cols2, vals2 = _read_chunk_as_arrays(file2, 0, 999999999)
        
        result_rows, result_cols, result_vals = _merge_sorted_coo(
            rows1, cols1, vals1,
            rows2, cols2, vals2
        )
        
        with open(output_file, 'w', newline='') as f:
            for idx in range(len(result_rows)):
                f.write(f"{result_rows[idx]+1},{result_cols[idx]+1},{result_vals[idx]}\n")
    
    def _process_chunks_parallel(self, file1: str, size1: int, 
                                 file2: str, size2: int) -> List[str]:
        """
        Process chunks in parallel.
        """
        max_size = max(size1, size2)
        num_chunks = (max_size + self.chunk_size - 1) // self.chunk_size
        
        work_items = []
        temp_dir = tempfile.mkdtemp(prefix='sparse_add_parallel_')
        
        for chunk_id in range(num_chunks):
            start1 = chunk_id * self.chunk_size
            chunk_size1 = min(self.chunk_size, size1 - start1) if start1 < size1 else 0
            
            start2 = chunk_id * self.chunk_size
            chunk_size2 = min(self.chunk_size, size2 - start2) if start2 < size2 else 0
            
            if chunk_size1 > 0 or chunk_size2 > 0:
                work_items.append((
                    chunk_id,
                    file1, start1, chunk_size1,
                    file2, start2, chunk_size2,
                    temp_dir
                ))
        
        self.logger.info(f"Processing {len(work_items)} chunk pairs...")
        
        with mp.Pool(processes=self.num_workers) as pool:
            result_files = pool.starmap(_process_chunk_pair, work_items)
        
        result_files = [f for f in result_files if f is not None]
        
        return result_files
    
    def _merge_chunks(self, temp_files: List[str], output_file: str):
        """
        Merge all chunk files using existing merge algorithm.
        """
        if not temp_files:
            with open(output_file, 'w', newline='') as f:
                pass
            return
        
        if len(temp_files) == 1:
            with open(output_file, 'w', newline='') as out:
                with open(temp_files[0], 'r') as inf:
                    for line in inf:
                        parts = line.strip().split(',')
                        if len(parts) == 3:
                            row = int(parts[0]) + 1
                            col = int(parts[1]) + 1
                            val = parts[2]
                            out.write(f"{row},{col},{val}\n")
            os.remove(temp_files[0])
            return
        
        current_files = temp_files[:]
        merge_round = 0
        
        while len(current_files) > 1:
            merge_round += 1
            next_files = []
            
            for i in range(0, len(current_files), 2):
                if i + 1 < len(current_files):
                    file1 = current_files[i]
                    file2 = current_files[i + 1]
                    merged_file = tempfile.mktemp(
                        prefix=f'merge_r{merge_round}_',
                        suffix='.csv',
                        dir=os.path.dirname(temp_files[0])
                    )
                    
                    rows1, cols1, vals1 = self._read_temp_file_as_arrays(file1)
                    rows2, cols2, vals2 = self._read_temp_file_as_arrays(file2)
                    
                    result_rows, result_cols, result_vals = _merge_sorted_coo(
                        rows1, cols1, vals1,
                        rows2, cols2, vals2
                    )
                    
                    with open(merged_file, 'w', newline='') as out:
                        for idx in range(len(result_rows)):
                            out.write(f"{result_rows[idx]},{result_cols[idx]},{result_vals[idx]}\n")
                    
                    next_files.append(merged_file)
                    
                    os.remove(file1)
                    os.remove(file2)
                else:
                    next_files.append(current_files[i])
            
            current_files = next_files
        
        with open(output_file, 'w', newline='') as out:
            with open(current_files[0], 'r') as inf:
                for line in inf:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        row = int(parts[0]) + 1
                        col = int(parts[1]) + 1
                        val = parts[2]
                        out.write(f"{row},{col},{val}\n")
        
        os.remove(current_files[0])
    
    def _read_temp_file_iterator(self, filepath: str) -> Iterator[Tuple[int, int, float]]:
        """Read temporary file (no header) as iterator."""
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    yield (int(parts[0]), int(parts[1]), float(parts[2]))
    
    def _read_temp_file_as_arrays(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read temporary file (no header) as numpy arrays."""
        rows, cols, vals = [], [], []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    rows.append(int(parts[0]))
                    cols.append(int(parts[1]))
                    vals.append(float(parts[2]))
        
        return (
            np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(vals, dtype=np.int32)
        )
    
    def _print_summary(self):
        """Print benchmark summary."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"PERFORMANCE SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total execution time:        {self.benchmarks['total_time']:.3f}s")
        self.logger.info(f"  - Counting entries:        {self.benchmarks['count_time']:.3f}s ({self.benchmarks['count_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"  - Parallel processing:     {self.benchmarks['parallel_processing_time']:.3f}s ({self.benchmarks['parallel_processing_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"  - Merging results:         {self.benchmarks['merge_time']:.3f}s ({self.benchmarks['merge_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"\nParallelization configuration:")
        self.logger.info(f"  - Number of CPU cores:     {self.num_workers}")
        self.logger.info(f"  - Chunk size:              {self.chunk_size:,} entries")
        self.logger.info(f"{'='*70}")
    
    def _save_benchmark_report(self, output_file: str, size1: int, size2: int):
        """Save detailed benchmark report to file."""
        result_size = 0
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_size = sum(1 for _ in f)
        
        report = {
            'operation': 'Sparse Matrix Addition (Parallel)',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_workers': self.num_workers,
                'chunk_size': self.chunk_size,
                'available_cores': mp.cpu_count()
            },
            'input': {
                'matrix_1_nonzeros': size1,
                'matrix_2_nonzeros': size2,
                'total_input_entries': size1 + size2
            },
            'output': {
                'result_nonzeros': result_size
            },
            'performance': {
                'total_time_seconds': round(self.benchmarks['total_time'], 3),
                'count_time_seconds': round(self.benchmarks['count_time'], 3),
                'parallel_processing_seconds': round(self.benchmarks['parallel_processing_time'], 3),
                'merge_time_seconds': round(self.benchmarks['merge_time'], 3)
            },
            'metrics': {
                'entries_per_second': round((size1 + size2) / self.benchmarks['total_time'], 2) if self.benchmarks['total_time'] > 0 else 0,
                'parallel_efficiency': round(self.benchmarks['parallel_processing_time'] / self.benchmarks['total_time'] * 100, 1),
                'theoretical_speedup': self.num_workers
            }
        }
        
        report_file = output_file.replace('.csv', '_benchmark.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        report_txt = output_file.replace('.csv', '_benchmark.txt')
        with open(report_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SPARSE MATRIX ADDITION - PARALLEL BENCHMARK REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  CPU Cores Used:        {report['configuration']['num_workers']}\n")
            f.write(f"  Available Cores:       {report['configuration']['available_cores']}\n")
            f.write(f"  Chunk Size:            {report['configuration']['chunk_size']:,} entries\n\n")
            
            f.write("INPUT DATA:\n")
            f.write(f"  Matrix A non-zeros:    {report['input']['matrix_1_nonzeros']:,}\n")
            f.write(f"  Matrix B non-zeros:    {report['input']['matrix_2_nonzeros']:,}\n")
            f.write(f"  Total input entries:   {report['input']['total_input_entries']:,}\n\n")
            
            f.write("OUTPUT DATA:\n")
            f.write(f"  Result non-zeros:      {report['output']['result_nonzeros']:,}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Total Time:            {report['performance']['total_time_seconds']:.3f}s\n")
            f.write(f"    - Counting:          {report['performance']['count_time_seconds']:.3f}s\n")
            f.write(f"    - Parallel Process:  {report['performance']['parallel_processing_seconds']:.3f}s\n")
            f.write(f"    - Merging:           {report['performance']['merge_time_seconds']:.3f}s\n\n")
            
            f.write("EFFICIENCY:\n")
            f.write(f"  Throughput:            {report['metrics']['entries_per_second']:,.0f} entries/sec\n")
            f.write(f"  Parallel Efficiency:   {report['metrics']['parallel_efficiency']:.1f}%\n")
            f.write(f"  Theoretical Speedup:   {report['metrics']['theoretical_speedup']}x\n\n")
            
            f.write("="*70 + "\n")
        
        self.logger.info(f"\n≡ƒôè Benchmark reports saved:")
        self.logger.info(f"   JSON: {report_file}")
        self.logger.info(f"   TXT:  {report_txt}")


def add_matrices_parallel(file1: str, file2: str, output_file: str,
                          chunk_size: int = 100000,
                          num_workers: Optional[int] = None) -> str:
    """
    Parallel sparse matrix addition using multi-core CPU.
    
    Args:
        file1: Path to first matrix (COO CSV, sorted)
        file2: Path to second matrix (COO CSV, sorted)
        output_file: Path for result matrix
        chunk_size: Number of entries per chunk (default 100000)
        num_workers: Number of CPU cores to use (default: all available)
    
    Returns:
        Path to output file
    """
    logger = logging.getLogger(__name__)
    
    adder = ParallelSparseAddition(
        chunk_size=chunk_size,
        num_workers=num_workers,
        logger=logger
    )
    
    return adder.add_matrices(file1, file2, output_file)
