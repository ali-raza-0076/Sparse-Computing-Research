================================================================================
    CORE COO IMPLEMENTATIONS
================================================================================

Sparse matrix operations using pure COO (Coordinate) format.


FILES
-----------------

  * sparse_addition_coo.py
    Addition using two-pointer merge, O(nnz(A) + nnz(B))

  * sparse_multiplication_coo.py
    Multiplication using hash accumulation, O(nnz(A) * density(B))


KEY DIFFERENCE FROM CSR/CSC
---------------------------

COO uses simple (row, col, value) triplets. No pointer arrays like CSR/CSC.

  * Simpler to understand
  * 2-10x slower for multiplication (hash lookup vs direct indexing)
  * Similar speed for addition


================================================================================
CONCLUSION
================================================================================

The COO format presents a fundamental trade-off between implementation
simplicity and computational efficiency. While the triplet-based representation
facilitates straightforward algorithm design and maintains comparable
performance for sparse addition operations, the absence of indexed access
structures necessitates hash-based accumulation during multiplication,
resulting in significant performance degradation relative to compressed formats.
This computational overhead, quantified at 2-10x slower execution times,
suggests that COO is best suited for applications prioritizing development
velocity and code maintainability over raw performance, or in scenarios where
matrix operations are dominated by addition rather than multiplication.

