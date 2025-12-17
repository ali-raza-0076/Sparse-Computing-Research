"""
GPU Matrix Multiplication Benchmark: CSR × CSC Sparse vs Dense
Mirrors CPU benchmark format with command-line arguments for matrix size and sparsity.
Compares sparse GPU (PyTorch) vs dense GPU multiplication performance.
"""

import torch
import time
import json
import os
import argparse
import numpy as np
from scipy import sparse as sp
from tabulate import tabulate


def generate_sparse_matrices(size, sparsity_percent, seed_a=42, seed_b=123):
    """
    Generate two sparse matrices (CSR and CSC) for multiplication.
    
    Args:
        size: Matrix dimension (NxN)
        sparsity_percent: Sparsity level (90, 99, 99.9, etc.)
        seed_a: Random seed for matrix A
        seed_b: Random seed for matrix B
    
    Returns:
        A_csr: First matrix in CSR format
        B_csc: Second matrix in CSC format
        actual_nnz: Actual number of non-zero elements
        actual_sparsity: Actual sparsity percentage
    """
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(size * size * density)
    
    np.random.seed(seed_a)
    rows_a = np.random.randint(0, size, size=num_entries)
    cols_a = np.random.randint(0, size, size=num_entries)
    vals_a = np.random.randint(1, 11, size=num_entries).astype(np.float32)
    A_csr = sp.csr_matrix((vals_a, (rows_a, cols_a)), shape=(size, size))
    
    np.random.seed(seed_b)
    rows_b = np.random.randint(0, size, size=num_entries)
    cols_b = np.random.randint(0, size, size=num_entries)
    vals_b = np.random.randint(1, 11, size=num_entries).astype(np.float32)
    B_csc = sp.csc_matrix((vals_b, (rows_b, cols_b)), shape=(size, size))
    
    actual_nnz = A_csr.nnz
    actual_sparsity = 100 * (1 - actual_nnz / (size * size))
    
    return A_csr, B_csc, actual_nnz, actual_sparsity


def csr_to_pytorch_sparse(csr_matrix, device):
    """Convert scipy CSR matrix to PyTorch COO sparse tensor."""
    coo = csr_matrix.tocoo()
    indices = torch.LongTensor([coo.row, coo.col]).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, runs=3):
    """
    Benchmark GPU sparse matrix multiplication (CSR × CSC).
    
    Args:
        A_sparse_gpu: First sparse tensor on GPU
        B_sparse_gpu: Second sparse tensor on GPU
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    _ = torch.sparse.mm(A_sparse_gpu, B_sparse_gpu.to_dense())
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.sparse.mm(A_sparse_gpu, B_sparse_gpu.to_dense())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def benchmark_gpu_dense(A_gpu, B_gpu, runs=3):
    """
    Benchmark GPU dense matrix multiplication.
    
    Args:
        A_gpu: First dense tensor on GPU
        B_gpu: Second dense tensor on GPU
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    _ = A_gpu @ B_gpu
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = A_gpu @ B_gpu
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def main():
    parser = argparse.ArgumentParser(description='GPU Matrix Multiplication Benchmark: CSR × CSC')
    parser.add_argument('--size', type=int, required=True, 
                        help='Matrix size (NxN)')
    parser.add_argument('--sparsity', type=float, required=True,
                        help='Sparsity percentage (e.g., 90, 99, 99.9)')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs for averaging (default: 3)')
    
    args = parser.parse_args()
    
    if args.size <= 0:
        print("ERROR: Matrix size must be positive")
        return
    if not (0 <= args.sparsity < 100):
        print("ERROR: Sparsity must be between 0 and 100")
        return
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! This benchmark requires a GPU.")
        return
    
    device = torch.device('cuda')
    
    print("\n" + "=" * 80)
    print("GPU MATRIX MULTIPLICATION BENCHMARK: CSR × CSC")
    print("Sparse vs Dense Comparison")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"\nConfiguration:")
    print(f"  Matrix Size: {args.size}×{args.size}")
    print(f"  Target Sparsity: {args.sparsity}%")
    print(f"  Runs per test: {args.num_runs}")
    
    print(f"\nGenerating sparse matrices...")
    A_csr, B_csc, actual_nnz, actual_sparsity = generate_sparse_matrices(
        args.size, args.sparsity
    )
    
    print(f"  Actual non-zeros: {actual_nnz:,}")
    print(f"  Actual sparsity: {actual_sparsity:.4f}%")
    
    print(f"\nConverting to dense format...")
    A_dense = A_csr.toarray()
    B_dense = B_csc.toarray()
    
    print(f"Transferring to GPU...")
    A_gpu_dense = torch.from_numpy(A_dense).float().to(device)
    B_gpu_dense = torch.from_numpy(B_dense).float().to(device)
    
    A_gpu_sparse = csr_to_pytorch_sparse(A_csr, device)
    B_gpu_sparse = csr_to_pytorch_sparse(B_csc, device)
    
    print(f"\nBenchmarking GPU Sparse (CSR × CSC)...")
    gpu_sparse_time = benchmark_gpu_sparse(A_gpu_sparse, B_gpu_sparse, args.num_runs)
    print(f"  GPU Sparse Time: {gpu_sparse_time:.6f}s")
    
    print(f"\nBenchmarking GPU Dense...")
    gpu_dense_time = benchmark_gpu_dense(A_gpu_dense, B_gpu_dense, args.num_runs)
    print(f"  GPU Dense Time: {gpu_dense_time:.6f}s")
    
    speedup = gpu_dense_time / gpu_sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"GPU Sparse Time: {gpu_sparse_time:.6f}s")
    print(f"GPU Dense Time:  {gpu_dense_time:.6f}s")
    print(f"Speedup:         {speedup:.2f}×")
    print(f"Winner:          {winner}")
    
    sparse_memory_mb = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes) / (1024**2)
    dense_memory_mb = A_dense.nbytes / (1024**2)
    memory_ratio = dense_memory_mb / sparse_memory_mb
    
    print(f"\nMemory Usage:")
    print(f"  Sparse: {sparse_memory_mb:.2f} MB")
    print(f"  Dense:  {dense_memory_mb:.2f} MB")
    print(f"  Ratio:  {memory_ratio:.2f}× (Dense uses {memory_ratio:.2f}× more memory)")
    
    result = {
        "matrix_size": args.size,
        "target_sparsity_pct": args.sparsity,
        "actual_sparsity_pct": actual_sparsity,
        "actual_nnz": actual_nnz,
        "num_runs": args.num_runs,
        "gpu_sparse_time_s": gpu_sparse_time,
        "gpu_dense_time_s": gpu_dense_time,
        "speedup": speedup,
        "winner": winner,
        "sparse_memory_mb": sparse_memory_mb,
        "dense_memory_mb": dense_memory_mb,
        "memory_ratio": memory_ratio,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__
    }
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    sparsity_str = f"{args.sparsity}".replace(".", "_")
    base_filename = f"multiplication_gpu_{args.size}x{args.size}_{sparsity_str}pct"
    
    json_file = os.path.join(output_dir, f"{base_filename}_results.json")
    with open(json_file, "w") as f:
        json.dump(result, f, indent=2)
    
    txt_file = os.path.join(output_dir, f"{base_filename}_results.txt")
    with open(txt_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GPU MATRIX MULTIPLICATION BENCHMARK: CSR × CSC\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Matrix Size: {args.size}×{args.size}\n")
        f.write(f"  Target Sparsity: {args.sparsity}%\n")
        f.write(f"  Actual Sparsity: {actual_sparsity:.4f}%\n")
        f.write(f"  Non-zeros: {actual_nnz:,}\n")
        f.write(f"  Runs per test: {args.num_runs}\n\n")
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"GPU Sparse Time: {gpu_sparse_time:.6f}s\n")
        f.write(f"GPU Dense Time:  {gpu_dense_time:.6f}s\n")
        f.write(f"Speedup:         {speedup:.2f}×\n")
        f.write(f"Winner:          {winner}\n\n")
        f.write(f"Memory Usage:\n")
        f.write(f"  Sparse: {sparse_memory_mb:.2f} MB\n")
        f.write(f"  Dense:  {dense_memory_mb:.2f} MB\n")
        f.write(f"  Ratio:  {memory_ratio:.2f}×\n")
    
    csv_file = os.path.join(output_dir, f"{base_filename}_results.csv")
    with open(csv_file, "w") as f:
        f.write("metric,value\n")
        f.write(f"matrix_size,{args.size}\n")
        f.write(f"target_sparsity_pct,{args.sparsity}\n")
        f.write(f"actual_sparsity_pct,{actual_sparsity:.4f}\n")
        f.write(f"actual_nnz,{actual_nnz}\n")
        f.write(f"num_runs,{args.num_runs}\n")
        f.write(f"gpu_sparse_time_s,{gpu_sparse_time:.6f}\n")
        f.write(f"gpu_dense_time_s,{gpu_dense_time:.6f}\n")
        f.write(f"speedup,{speedup:.2f}\n")
        f.write(f"winner,{winner}\n")
        f.write(f"sparse_memory_mb,{sparse_memory_mb:.2f}\n")
        f.write(f"dense_memory_mb,{dense_memory_mb:.2f}\n")
        f.write(f"memory_ratio,{memory_ratio:.2f}\n")
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  - {json_file}")
    print(f"  - {txt_file}")
    print(f"  - {csv_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
