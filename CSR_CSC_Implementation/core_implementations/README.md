================================================================================
    CORE IMPLEMENTATIONS
================================================================================

Custom sparse matrix algorithms written from scratch for this project.


CONTENTS
-----------------

Matrix Operations:
  * sparse_multiplication.py
    CSR * CSC single-threaded multiplication algorithm

  * sparse_multiplication_parallel.py
    CSR * CSC multicore parallel multiplication (16 cores)

  * sparse_addition_csr.py
    CSR row-by-row single-threaded addition (Added December 2025)

  * sparse_addition_parallel_csr.py
    CSR row-block parallel addition (16 cores) (Added December 2025)

  * sparse_addition.py
    COO merge addition (legacy, for comparison)

  * sparse_addition_parallel.py
    COO parallel addition (legacy, for comparison)


Matrix Formats:
  * matrix_formats.py
    COOMatrix, CSRMatrix, CSCMatrix classes with conversions


External Sorting:
  * external_sort.py
    Out-of-core sorting for matrices exceeding RAM


USAGE
-----------------

All benchmark folders, GNN tests, and GPU comparisons import from these files:

  sys.path.insert(0, '../core_implementations')
  from sparse_multiplication import sparse_multiply_from_coo
  from sparse_addition import sparse_add_coo
  from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix


