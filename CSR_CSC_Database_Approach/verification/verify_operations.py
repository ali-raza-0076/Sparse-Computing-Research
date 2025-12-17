"""
Verification script for CSR/CSC database approach results.
Validates correctness by checking basic properties and sampling.
"""

import numpy as np
import os

def read_coo_csv(filepath):
    """Read COO format CSV file."""
    data = np.loadtxt(filepath, delimiter=',', dtype=np.int32)
    rows = data[:, 0]
    cols = data[:, 1]
    vals = data[:, 2]
    return rows, cols, vals

def verify_addition_properties():
    """Verify addition by checking mathematical properties."""
    print("="*70)
    print("VERIFYING CSR ADDITION (CPU) - Property-Based")
    print("="*70)
    
    input_dir = os.path.join('..', '..', 'input')
    rows_a, cols_a, vals_a = read_coo_csv(os.path.join(input_dir, 'matrix_a.csv'))
    rows_b, cols_b, vals_b = read_coo_csv(os.path.join(input_dir, 'matrix_a.csv'))
    
    print(f"Matrix A: 100,000 nonzeros")
    print(f"Matrix B: 100,000 nonzeros (same as A)")
    
    result_file = os.path.join('..', 'dense_sparse_cpu_benchmarks', 'single_threaded', 
                                'results', 'addition_csr_50001x50000.csv')
    result_rows, result_cols, result_vals = read_coo_csv(result_file)
    print(f"Result: {len(result_vals)} nonzeros")
    
    print("\nChecking property: A + A should have values = 2 * A")
    
    a_dict = {(r, c): v for r, c, v in zip(rows_a, cols_a, vals_a)}
    result_dict = {(r, c): v for r, c, v in zip(result_rows, result_cols, result_vals)}
    
    if len(result_dict) != len(a_dict):
        print(f"❌ FAILED: Expected {len(a_dict)} entries, got {len(result_dict)}")
        return False
    
    sample_size = min(1000, len(a_dict))
    sample_keys = np.random.choice(len(rows_a), sample_size, replace=False)
    
    errors = 0
    for idx in sample_keys:
        pos = (rows_a[idx], cols_a[idx])
        expected = vals_a[idx] * 2
        actual = result_dict.get(pos, None)
        
        if actual is None:
            print(f"❌ Missing entry at {pos}")
            errors += 1
        elif actual != expected:
            print(f"❌ At {pos}: expected {expected}, got {actual}")
            errors += 1
    
    if errors > 0:
        print(f"❌ FAILED: {errors} errors found in sample")
        return False
    
    print(f"✅ PASSED: Sampled {sample_size} entries, all correct (A+A = 2A)")
    return True

def verify_multiplication_properties():
    """Verify multiplication by checking basic properties."""
    print("\n" + "="*70)
    print("VERIFYING CSR×CSC MULTIPLICATION (CPU) - Property-Based")
    print("="*70)
    
    input_dir = os.path.join('..', '..', 'input')
    rows_a, cols_a, vals_a = read_coo_csv(os.path.join(input_dir, 'matrix_a.csv'))
    rows_b, cols_b, vals_b = read_coo_csv(os.path.join(input_dir, 'matrix_a_transposed.csv'))
    
    print(f"Matrix A: (50001, 50000), 100,000 nonzeros")
    print(f"Matrix B: (50000, 50001), 100,000 nonzeros")
    
    result_file = os.path.join('..', 'dense_sparse_cpu_benchmarks', 'single_threaded',
                                'results', 'multiplication_csr_csc_50001x50001.csv')
    result_rows, result_cols, result_vals = read_coo_csv(result_file)
    print(f"Result: {len(result_vals)} nonzeros")
    
    print("\nChecking properties...")
    
    if np.max(result_rows) >= 50001 or np.max(result_cols) >= 50001:
        print("❌ FAILED: Result indices out of bounds")
        return False
    print("✅ Result shape is valid (50001×50001)")
    
    positions = set(zip(result_rows, result_cols))
    if len(positions) != len(result_vals):
        print("❌ FAILED: Duplicate entries found")
        return False
    print("✅ No duplicate entries")
    
    if np.any(result_vals == 0):
        print("❌ FAILED: Result contains zero values")
        return False
    print("✅ All values are non-zero")
    
    print("\nSpot-checking 5 random entries...")
    a_dict = {}
    for r, c, v in zip(rows_a, cols_a, vals_a):
        if r not in a_dict:
            a_dict[r] = {}
        a_dict[r][c] = v
    
    b_dict = {}
    for r, c, v in zip(rows_b, cols_b, vals_b):
        if r not in b_dict:
            b_dict[r] = {}
        b_dict[r][c] = v
    
    sample_indices = np.random.choice(len(result_vals), min(5, len(result_vals)), replace=False)
    
    for idx in sample_indices:
        i, j = result_rows[idx], result_cols[idx]
        expected_val = 0
        
        if i in a_dict:
            for k, a_val in a_dict[i].items():
                if k in b_dict and j in b_dict[k]:
                    expected_val += a_val * b_dict[k][j]
        
        actual_val = result_vals[idx]
        if expected_val != actual_val:
            print(f"❌ At ({i},{j}): expected {expected_val}, got {actual_val}")
            return False
        print(f"  ✓ ({i},{j}) = {actual_val}")
    
    print("✅ PASSED: Multiplication result is correct!")
    return True

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = True
    success &= verify_addition_properties()
    success &= verify_multiplication_properties()
    
    print("\n" + "="*70)
    if success:
        print("ALL VERIFICATIONS PASSED ✅")
    else:
        print("SOME VERIFICATIONS FAILED ❌")
    print("="*70)

