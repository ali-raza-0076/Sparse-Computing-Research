"""
GPU Matrix Multiplication Benchmark: COO × COO Sparse vs Dense
Compares sparse GPU (PyTorch COO) vs dense GPU multiplication performance using true COO format.
"""

import torch
import time
import json
import os
import argparse
import numpy as np
from scipy import sparse as sp
from tabulate import tabulate


def generate_sparse_coo_matrices(size, sparsity_percent, seed_a=42, seed_b=123):
    """
    Generate two sparse matrices in COO format for multiplication.
    
    Args:
        size: Matrix dimension (NxN)
        sparsity_percent: Sparsity level (90, 99, 99.9, etc.)
        seed_a: Random seed for matrix A
        seed_b: Random seed for matrix B
    
    Returns:
        A_coo: First matrix in COO format
        B_coo: Second matrix in COO format
        actual_nnz: Actual number of non-zero elements
        actual_sparsity: Actual sparsity percentage
    """
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(size * size * density)
    
    np.random.seed(seed_a)
    rows_a = np.random.randint(0, size, size=num_entries)
    cols_a = np.random.randint(0, size, size=num_entries)
    vals_a = np.random.randint(1, 11, size=num_entries).astype(np.float32)
    A_coo = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(size, size))
    
    np.random.seed(seed_b)
    rows_b = np.random.randint(0, size, size=num_entries)
    cols_b = np.random.randint(0, size, size=num_entries)
    vals_b = np.random.randint(1, 11, size=num_entries).astype(np.float32)
    B_coo = sp.coo_matrix((vals_b, (rows_b, cols_b)), shape=(size, size))
    
    actual_nnz = A_coo.nnz
    actual_sparsity = 100 * (1 - actual_nnz / (size * size))
    
    return A_coo, B_coo, actual_nnz, actual_sparsity


def coo_to_pytorch_sparse(coo_matrix, device):
    """Convert scipy COO matrix to PyTorch COO sparse tensor."""
    indices = torch.LongTensor([coo_matrix.row, coo_matrix.col]).to(device)
    values = torch.FloatTensor(coo_matrix.data).to(device)
    shape = coo_matrix.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, runs=3):
    """
    Benchmark GPU sparse matrix multiplication (COO × COO).
    
    Args:
        A_sparse_gpu: First sparse COO tensor on GPU
        B_sparse_gpu: Second sparse COO tensor on GPU
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    _ = torch.sparse.mm(A_sparse_gpu, B_sparse_gpu)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.sparse.mm(A_sparse_gpu, B_sparse_gpu)
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
    _ = torch.mm(A_gpu, B_gpu)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.mm(A_gpu, B_gpu)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def save_results(results, output_dir, size, sparsity):
    """Save benchmark results to JSON, TXT, and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_name = f"multiplication_gpu_coo_{size}x{size}_{sparsity_str}pct_results"
    
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GPU MULTIPLICATION BENCHMARK RESULTS (COO Format)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Matrix Size: {size}×{size}\n")
        f.write(f"Sparsity: {sparsity}%\n")
        f.write(f"Non-zeros: {results['non_zeros']:,}\n")
        f.write(f"GPU Device: {results['gpu_device']}\n\n")
        f.write(f"Sparse GPU Time: {results['sparse_time']:.6f}s\n")
        f.write(f"Dense GPU Time:  {results['dense_time']:.6f}s\n")
        f.write(f"Speedup: {results['speedup']:.2f}×\n")
        f.write(f"Winner: {results['winner']}\n\n")
        f.write(f"Memory - Sparse: {results['sparse_memory_mb']:.2f} MB\n")
        f.write(f"Memory - Dense:  {results['dense_memory_mb']:.2f} MB\n")
        f.write(f"Memory Ratio: {results['memory_ratio']:.2f}×\n")
    
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    with open(csv_path, 'w') as f:
        f.write("metric,value\n")
        f.write(f"matrix_size,{size}\n")
        f.write(f"sparsity_percent,{sparsity}\n")
        f.write(f"non_zeros,{results['non_zeros']}\n")
        f.write(f"sparse_time_sec,{results['sparse_time']}\n")
        f.write(f"dense_time_sec,{results['dense_time']}\n")
        f.write(f"speedup,{results['speedup']}\n")
        f.write(f"winner,{results['winner']}\n")
        f.write(f"sparse_memory_mb,{results['sparse_memory_mb']}\n")
        f.write(f"dense_memory_mb,{results['dense_memory_mb']}\n")
        f.write(f"memory_ratio,{results['memory_ratio']}\n")
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - {base_name}.json")
    print(f"  - {base_name}.txt")
    print(f"  - {base_name}.csv")


def main():
    parser = argparse.ArgumentParser(description='GPU Sparse Multiplication Benchmark (COO Format)')
    parser.add_argument('--size', type=int, default=10000, help='Matrix size (NxN)')
    parser.add_argument('--sparsity', type=float, default=99.0, help='Sparsity percentage (90, 99, 99.9)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of benchmark runs')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available!")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    
    print("=" * 70)
    print("GPU MULTIPLICATION BENCHMARK: COO Sparse vs Dense")
    print("=" * 70)
    print(f"\nGPU Device: {gpu_name}")
    print(f"Matrix Size: {args.size}×{args.size}")
    print(f"Sparsity: {args.sparsity}%")
    print(f"Runs: {args.num_runs}")
    print()
    
    print("Generating sparse COO matrices...")
    A_coo, B_coo, actual_nnz, actual_sparsity = generate_sparse_coo_matrices(
        args.size, args.sparsity
    )
    
    print(f"Actual non-zeros: {actual_nnz:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    print()
    
    print("Converting to PyTorch COO sparse tensors (GPU)...")
    A_sparse_gpu = coo_to_pytorch_sparse(A_coo, device)
    B_sparse_gpu = coo_to_pytorch_sparse(B_coo, device)
    
    print("Converting to dense tensors (GPU)...")
    A_dense_gpu = A_sparse_gpu.to_dense()
    B_dense_gpu = B_sparse_gpu.to_dense()
    
    sparse_mem_mb = (actual_nnz * (8 + 4)) / (1024 * 1024)
    dense_mem_mb = (args.size * args.size * 4) / (1024 * 1024)  # float32
    memory_ratio = dense_mem_mb / sparse_mem_mb if sparse_mem_mb > 0 else 0
    
    print(f"\nMemory Usage:")
    print(f"  Sparse (COO): {sparse_mem_mb:.2f} MB")
    print(f"  Dense:        {dense_mem_mb:.2f} MB")
    print(f"  Ratio:        {memory_ratio:.2f}×")
    print()
    
    print(f"Benchmarking SPARSE GPU (COO × COO, {args.num_runs} runs)...")
    sparse_time = benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, args.num_runs)
    print(f"  Average: {sparse_time:.6f}s")
    
    print(f"Benchmarking DENSE GPU ({args.num_runs} runs)...")
    dense_time = benchmark_gpu_dense(A_dense_gpu, B_dense_gpu, args.num_runs)
    print(f"  Average: {dense_time:.6f}s")
    print()
    
    speedup = dense_time / sparse_time if sparse_time > 0 else 0
    winner = "Sparse" if speedup > 1.0 else "Dense"
    
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    table_data = [[
        f"{args.sparsity}%",
        f"{actual_nnz:,}",
        f"{sparse_time:.6f}s",
        f"{dense_time:.6f}s",
        f"{speedup:.2f}×",
        winner,
        f"{sparse_mem_mb:.2f} MB",
        f"{dense_mem_mb:.2f} MB"
    ]]
    headers = ["Sparsity", "Non-Zeros", "Sparse Time", "Dense Time", "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()
    
    results = {
        "matrix_size": args.size,
        "sparsity_percent": args.sparsity,
        "non_zeros": int(actual_nnz),
        "actual_sparsity": float(actual_sparsity),
        "sparse_time": float(sparse_time),
        "dense_time": float(dense_time),
        "speedup": float(speedup),
        "winner": winner,
        "sparse_memory_mb": float(sparse_mem_mb),
        "dense_memory_mb": float(dense_mem_mb),
        "memory_ratio": float(memory_ratio),
        "gpu_device": gpu_name,
        "format": "COO",
        "num_runs": args.num_runs
    }
    
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    save_results(results, output_dir, args.size, args.sparsity)
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION: {winner} wins with {speedup:.2f}× speedup")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
