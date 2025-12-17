"""Verify correctness of COO benchmark results"""
import csv
import numpy as np
from collections import defaultdict

def read_coo(filepath):
    """Read COO format CSV file and merge duplicates"""
    from collections import defaultdict
    data_dict = defaultdict(float)
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            i, j, val = int(row[0]), int(row[1]), float(row[2])
            data_dict[(i, j)] += val
    data = [(i, j, v) for (i, j), v in data_dict.items() if abs(v) > 1e-10]
    return data

def coo_to_dict(coo_data):
    """Convert COO to dictionary for easy comparison"""
    d = {}
    for i, j, val in coo_data:
        d[(i, j)] = val
    return d

def verify_addition(A, B, result):
    """Verify A + B = result"""
    dict_A = coo_to_dict(A)
    dict_B = coo_to_dict(B)
    dict_result = coo_to_dict(result)
    
    all_positions = set(dict_A.keys()) | set(dict_B.keys()) | set(dict_result.keys())
    
    errors = 0
    for pos in all_positions:
        val_A = dict_A.get(pos, 0.0)
        val_B = dict_B.get(pos, 0.0)
        expected = val_A + val_B
        actual = dict_result.get(pos, 0.0)
        
        if abs(expected - actual) > 1e-6:
            errors += 1
            if errors <= 5:
                print(f"  Error at {pos}: A={val_A}, B={val_B}, expected={expected}, got={actual}")
    
    return errors

def verify_multiplication(A, B, result):
    """Verify B × A = result (sparse matrix multiplication)"""
    dict_A = coo_to_dict(A)
    dict_B = coo_to_dict(B)
    dict_result = coo_to_dict(result)
    
    A_by_col = defaultdict(list)
    for i, j, val in A:
        A_by_col[i].append((j, val))
    
    B_by_row = defaultdict(list)
    for i, j, val in B:
        B_by_row[i].append((j, val))
    
    import random
    sample_size = min(100, len(result))
    sample = random.sample(result, sample_size)
    
    errors = 0
    for i, j, result_val in sample:
        expected = 0.0
        for k, b_val in B_by_row.get(i, []):
            a_val = dict_A.get((k, j), 0.0)
            expected += b_val * a_val
        
        if abs(expected - result_val) > 1e-5:
            errors += 1
            if errors <= 5:
                print(f"  Error at ({i},{j}): expected={expected}, got={result_val}")
    
    return errors, sample_size

print("="*70)
print("VERIFICATION OF COO BENCHMARK RESULTS")
print("="*70)

print("\n[1] Reading input matrices...")
A = read_coo('input/matrix_a.csv')
B = read_coo('input/matrix_b.csv')
print(f"  Matrix A: {len(A)} non-zeros")
print(f"  Matrix B: {len(B)} non-zeros")

print("\n[2] Verifying ADDITION results...")

print("\n  a) Single-threaded addition:")
result_st = read_coo('COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/results/addition_coo_50001x50001_100000A_100000B.csv')
print(f"     Result has {len(result_st)} non-zeros")
errors = verify_addition(A, B, result_st)
if errors == 0:
    print(f"     ✓ CORRECT - All {len(result_st)} entries verified")
else:
    print(f"     ✗ ERRORS - {errors} mismatches found")

print("\n  b) Parallel addition:")
result_par = read_coo('COO_Database_Approach/dense_sparse_cpu_benchmarks/multicore_parallel/results/addition_parallel_coo_50001x50001_32cores_result.csv')
print(f"     Result has {len(result_par)} non-zeros")
errors = verify_addition(A, B, result_par)
if errors == 0:
    print(f"     ✓ CORRECT - All {len(result_par)} entries verified")
else:
    print(f"     ✗ ERRORS - {errors} mismatches found")

print("\n  c) GPU addition:")
result_gpu = read_coo('COO_Database_Approach/dense_sparse_gpu_benchmarks/results/addition_gpu_coo_50001x50001_100000A_100000B.csv')
print(f"     Result has {len(result_gpu)} non-zeros")
errors = verify_addition(A, B, result_gpu)
if errors == 0:
    print(f"     ✓ CORRECT - All {len(result_gpu)} entries verified")
else:
    print(f"     ✗ ERRORS - {errors} mismatches found")

print("\n[3] Verifying MULTIPLICATION results (B×A, sampled)...")

print("\n  a) Single-threaded multiplication:")
result_mult_st = read_coo('COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/results/multiplication_coo_BxA_50001x50000_result.csv')
print(f"     Result has {len(result_mult_st)} non-zeros")
errors, sample = verify_multiplication(A, B, result_mult_st)
if errors == 0:
    print(f"     ✓ CORRECT - Sampled {sample} entries, all verified")
else:
    print(f"     ✗ ERRORS - {errors}/{sample} sampled entries incorrect")

print("\n  b) GPU multiplication:")
result_mult_gpu = read_coo('COO_Database_Approach/dense_sparse_gpu_benchmarks/results/multiplication_gpu_coo_BxA_50001x50000.csv')
print(f"     Result has {len(result_mult_gpu)} non-zeros")
errors, sample = verify_multiplication(A, B, result_mult_gpu)
if errors == 0:
    print(f"     ✓ CORRECT - Sampled {sample} entries, all verified")
else:
    print(f"     ✗ ERRORS - {errors}/{sample} sampled entries incorrect")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
