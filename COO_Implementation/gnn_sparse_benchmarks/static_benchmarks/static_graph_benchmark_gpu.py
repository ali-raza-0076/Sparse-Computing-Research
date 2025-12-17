"""
GPU Graph Benchmark: Static Graph Operations (COO Sparse vs Dense)
Tests graph operations across different sizes and sparsity levels on GPU.
Compares sparse PyTorch COO matrix multiplication vs dense tensor operations.
"""
import torch
import numpy as np
import time
import json
import os
import argparse
from scipy import sparse as sp
from tabulate import tabulate


def check_gpu():
    """Check GPU availability and display info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_version = torch.version.cuda
    
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    print(f"CUDA Version: {cuda_version}")
    print(f"PyTorch Version: {torch.__version__}")
    return True


def generate_graph(num_vertices, sparsity_percent, seed=42):
    """
    Generate random graph adjacency matrix in COO format.
    
    Args:
        num_vertices: Number of nodes
        sparsity_percent: Sparsity level
        seed: Random seed
    
    Returns:
        coo_matrix, actual_edges, actual_sparsity
    """
    np.random.seed(seed)
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(num_vertices * num_vertices * density)
    
    rows = np.random.randint(0, num_vertices, size=num_edges)
    cols = np.random.randint(0, num_vertices, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(num_vertices, num_vertices))
    return matrix, matrix.nnz, 100 * (1 - matrix.nnz / (num_vertices * num_vertices))


def coo_to_pytorch_sparse(coo_matrix, device):
    """Convert scipy COO matrix to PyTorch COO sparse tensor."""
    indices = torch.LongTensor(np.stack([coo_matrix.row, coo_matrix.col])).to(device)
    values = torch.FloatTensor(coo_matrix.data).to(device)
    shape = coo_matrix.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, runs=3):
    """
    Benchmark GPU sparse matrix multiplication (COO format).
    
    Args:
        A_sparse_gpu: First sparse tensor on GPU
        B_sparse_gpu: Second sparse tensor on GPU
        runs: Number of runs
    
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
        runs: Number of runs
    
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


def save_results(results, size, sparsity):
    """Save benchmark results in multiple formats."""
    os.makedirs('results', exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_filename = f"static_graph_gpu_coo_{size}vertices_{sparsity_str}pct_results"
    
    with open(f'results/{base_filename}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GPU STATIC GRAPH BENCHMARK (COO): Sparse vs Dense\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"GPU: {results['gpu_name']}\n")
        f.write(f"Graph Size: {results['num_vertices']} nodes\n")
        f.write(f"Sparsity: {results['sparsity']}%\n")
        f.write(f"Edges: {results['num_edges']:,}\n")
        f.write(f"Runs: {results['num_runs']}\n\n")
        
        f.write("Performance:\n")
        f.write(f"  Sparse (COO): {results['sparse_time']:.6f}s\n")
        f.write(f"  Dense:        {results['dense_time']:.6f}s\n")
        f.write(f"  Speedup:      {results['speedup']:.2f}×\n")
        f.write(f"  Winner:       {results['winner']}\n\n")
        
        f.write("Memory:\n")
        f.write(f"  Sparse: {results['sparse_memory_mb']:.2f} MB\n")
        f.write(f"  Dense:  {results['dense_memory_mb']:.2f} MB\n")
        f.write(f"  Ratio:  {results['memory_ratio']:.2f}×\n")
    
    with open(f'results/{base_filename}.csv', 'w') as f:
        f.write("metric,value\n")
        f.write(f"gpu_name,{results['gpu_name']}\n")
        f.write(f"num_vertices,{results['num_vertices']}\n")
        f.write(f"sparsity,{results['sparsity']}\n")
        f.write(f"num_edges,{results['num_edges']}\n")
        f.write(f"sparse_time,{results['sparse_time']}\n")
        f.write(f"dense_time,{results['dense_time']}\n")
        f.write(f"speedup,{results['speedup']}\n")
        f.write(f"winner,{results['winner']}\n")
        f.write(f"sparse_memory_mb,{results['sparse_memory_mb']}\n")
        f.write(f"dense_memory_mb,{results['dense_memory_mb']}\n")


def main():
    parser = argparse.ArgumentParser(description='GPU Static Graph Benchmark (COO)')
    parser.add_argument('--vertices', type=int, default=10000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99.0, help='Sparsity percentage (90, 99, 99.9)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU STATIC GRAPH BENCHMARK (COO): Sparse vs Dense")
    print("=" * 70)
    print()
    
    if not check_gpu():
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    
    print(f"\nGraph Size: {args.vertices} nodes")
    print(f"Sparsity: {args.sparsity}%")
    print(f"Runs: {args.num_runs}")
    
    print("\nGenerating graph adjacency matrix (COO format)...")
    A, edges, actual_sparsity = generate_graph(args.vertices, args.sparsity, seed=42)
    B, _, _ = generate_graph(args.vertices, args.sparsity, seed=123)
    
    print(f"Actual edges: {edges:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    print("Converting to PyTorch COO sparse tensors (GPU)...")
    A_sparse_gpu = coo_to_pytorch_sparse(A, device)
    B_sparse_gpu = coo_to_pytorch_sparse(B, device)
    
    print("Converting to dense tensors (GPU)...")
    A_dense_gpu = torch.FloatTensor(A.toarray()).to(device)
    B_dense_gpu = torch.FloatTensor(B.toarray()).to(device)
    
    sparse_mem = (A.nnz * (8 + 4 + 4)) / (1024 * 1024)
    dense_mem = (args.vertices * args.vertices * 4) / (1024 * 1024)
    mem_ratio = dense_mem / sparse_mem if sparse_mem > 0 else 0
    
    print(f"\nMemory Usage:")
    print(f"  Sparse (COO): {sparse_mem:.2f} MB")
    print(f"  Dense:        {dense_mem:.2f} MB")
    print(f"  Ratio:        {mem_ratio:.2f}×")
    
    print(f"\nBenchmarking SPARSE GPU (COO, {args.num_runs} runs)...")
    sparse_time = benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, args.num_runs)
    print(f"  Average: {sparse_time:.6f}s")
    
    print(f"Benchmarking DENSE GPU ({args.num_runs} runs)...")
    dense_time = benchmark_gpu_dense(A_dense_gpu, B_dense_gpu, args.num_runs)
    print(f"  Average: {dense_time:.6f}s")
    
    speedup = dense_time / sparse_time if sparse_time > 0 else 0
    winner = "Sparse" if speedup > 1.0 else "Dense"
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    table_data = [[
        f"{args.sparsity}%",
        f"{edges:,}",
        f"{sparse_time:.6f}s",
        f"{dense_time:.6f}s",
        f"{speedup:.2f}×",
        winner,
        f"{sparse_mem:.2f} MB",
        f"{dense_mem:.2f} MB"
    ]]
    
    headers = ["Sparsity", "Edges", "Sparse Time", "Dense Time", "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    results = {
        "gpu_name": gpu_name,
        "num_vertices": args.vertices,
        "sparsity": args.sparsity,
        "num_edges": edges,
        "actual_sparsity": actual_sparsity,
        "num_runs": args.num_runs,
        "sparse_time": sparse_time,
        "dense_time": dense_time,
        "speedup": speedup,
        "winner": winner,
        "sparse_memory_mb": sparse_mem,
        "dense_memory_mb": dense_mem,
        "memory_ratio": mem_ratio
    }
    
    save_results(results, args.vertices, args.sparsity)
    
    result_files = f"static_graph_gpu_coo_{args.vertices}vertices_{str(args.sparsity).replace('.', '_')}pct_results.*"
    print(f"\nResults saved to: results/{result_files}")
    
    print("\n" + "=" * 70)
    print(f"CONCLUSION: {winner} wins with {speedup:.2f}× speedup")
    print("=" * 70)


if __name__ == "__main__":
    main()

