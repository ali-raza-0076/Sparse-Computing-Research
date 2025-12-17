"""Quick test of TRUE multiprocessing sparse operations"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'core_implementations'))

from sparse_addition_multiprocess_csr import sparse_add_csr_multiprocess
from sparse_multiplication_multiprocess_csr import sparse_multiply_csr_multiprocess

print("="*60)
print("Testing TRUE Multiprocess Multiplication")
print("="*60)

A_indptr = np.array([0, 2, 3], dtype=np.int64)
A_indices = np.array([0, 2, 1], dtype=np.int32)
A_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

B_indptr = np.array([0, 2, 2, 4], dtype=np.int64)
B_indices = np.array([0, 1, 0, 1], dtype=np.int32)
B_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

print(f"\nMatrix A (2×3):")
print(f"  indptr: {A_indptr}")
print(f"  indices: {A_indices}")
print(f"  data: {A_data}")

print(f"\nMatrix B (3×2):")
print(f"  indptr: {B_indptr}")
print(f"  indices: {B_indices}")
print(f"  data: {B_data}")

print(f"\nExpected result C = A × B (2×2):")
print(f"  [[1*1 + 2*3, 1*2 + 2*4], [3*0, 3*0]]")
print(f"  [[7, 10], [0, 0]]")

C_indptr, C_indices, C_data = sparse_multiply_csr_multiprocess(
    A_indptr, A_indices, A_data,
    B_indptr, B_indices, B_data,
    (2, 3), (3, 2),
    num_cores=2
)

print(f"\nResult C (2×2):")
print(f"  indptr: {C_indptr}")
print(f"  indices: {C_indices}")
print(f"  data: {C_data}")
print(f"  Non-zeros: {len(C_data)}")

C_dense = np.zeros((2, 2))
for row in range(2):
    start = C_indptr[row]
    end = C_indptr[row + 1]
    for idx in range(start, end):
        col = C_indices[idx]
        val = C_data[idx]
        C_dense[row, col] = val

print(f"\nDense representation:")
print(C_dense)

print("\n" + "="*60)
print("Test multiplication complete!")
print("="*60)
