================================================================================
    VERIFICATION SCRIPTS
================================================================================


OVERVIEW
-----------------

Verification utilities for CSR/CSC database approach implementations.


FILES
-----------------

  * verify_operations.py - Correctness verification for sparse matrix operations


USAGE
-----------------

Run verification from project root:

  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\verification\verify_operations.py


TECHNICAL NOTES
-----------------

  * Verifies CSR/CSC addition and multiplication against scipy.sparse reference
    implementations
  * Checks correctness of database I/O workflow (read - compute - write)
  * Validates format conversions (COO - CSR/CSC - COO)

