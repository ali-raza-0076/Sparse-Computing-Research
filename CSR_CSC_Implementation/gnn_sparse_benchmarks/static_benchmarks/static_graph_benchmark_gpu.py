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
    Generate random graph adjacency matrix in scipy CSR format.
    
    Args:
        num_vertices: Number of nodes
        sparsity_percent: Sparsity level
        seed: Random seed
    
    Returns:
        csr_matrix, actual_edges, actual_sparsity
    """
    np.random.seed(seed)
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(num_vertices * num_vertices * density)
    
    rows = np.random.randint(0, num_vertices, size=num_edges)
    cols = np.random.randint(0, num_vertices, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_vertices, num_vertices))
    return matrix, matrix.nnz, 100 * (1 - matrix.nnz / (num_vertices * num_vertices))


def csr_to_pytorch_sparse(csr_matrix, device):
    """Convert scipy CSR matrix to PyTorch COO sparse tensor."""
    coo = csr_matrix.tocoo()
    indices = torch.LongTensor([coo.row, coo.col]).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = coo.shape
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
        runs: Number of runs
    
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


def save_results(results, size, sparsity):
    """Save benchmark results in multiple formats."""
    os.makedirs('results', exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_filename = f'static_graph_gpu_{size}vertices_{sparsity_str}pct'
    
    with open(f'results/{base_filename}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU STATIC GRAPH BENCHMARK: Sparse COO vs Dense\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"GPU: {results['gpu_name']}\n")
        f.write(f"CUDA Version: {results['cuda_version']}\n")
        f.write(f"PyTorch Version: {results['pytorch_version']}\n\n")
        
        f.write(f"Graph Configuration:\n")
        f.write(f"  Vertices: {results['num_vertices']:,}\n")
        f.write(f"  Target Sparsity: {results['target_sparsity']}%\n")
        f.write(f"  Actual Edges: {results['actual_edges']:,}\n")
        f.write(f"  Actual Sparsity: {results['actual_sparsity']:.4f}%\n")
        f.write(f"  Runs per test: {results['num_runs']}\n\n")
        
        f.write("Results:\n")
        f.write(f"  GPU Sparse (COO) Time: {results['sparse_time']:.6f}s\n")
        f.write(f"  GPU Dense Time: {results['dense_time']:.6f}s\n")
        f.write(f"  Speedup: {results['speedup']:.2f}×\n")
        f.write(f"  Winner: {results['winner']}\n\n")
        
        f.write("Memory Usage:\n")
        f.write(f"  Sparse: {results['sparse_memory_mb']:.2f} MB\n")
        f.write(f"  Dense: {results['dense_memory_mb']:.2f} MB\n")
        f.write(f"  Memory Ratio: {results['memory_ratio']:.2f}× savings\n")
    
    with open(f'results/{base_filename}_results.csv', 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"GPU,{results['gpu_name']}\n")
        f.write(f"Vertices,{results['num_vertices']}\n")
        f.write(f"Target_Sparsity,{results['target_sparsity']}\n")
        f.write(f"Actual_Edges,{results['actual_edges']}\n")
        f.write(f"Actual_Sparsity,{results['actual_sparsity']}\n")
        f.write(f"Sparse_Time_s,{results['sparse_time']}\n")
        f.write(f"Dense_Time_s,{results['dense_time']}\n")
        f.write(f"Speedup,{results['speedup']}\n")
        f.write(f"Winner,{results['winner']}\n")
        f.write(f"Sparse_Memory_MB,{results['sparse_memory_mb']}\n")
        f.write(f"Dense_Memory_MB,{results['dense_memory_mb']}\n")
        f.write(f"Memory_Ratio,{results['memory_ratio']}\n")


def main():
    parser = argparse.ArgumentParser(description='GPU Static Graph Benchmark')
    parser.add_argument('--vertices', type=int, default=1000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99, help='Sparsity percentage')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPU STATIC GRAPH BENCHMARK: Sparse COO vs Dense")
    print("=" * 80)
    print()
    
    if not check_gpu():
        return
    print()
    
    device = torch.device('cuda')
    
    print(f"Configuration:")
    print(f"  Graph Size: {args.vertices}×{args.vertices} adjacency matrix")
    print(f"  Target Sparsity: {args.sparsity}%")
    print(f"  Runs per test: {args.num_runs}")
    print()
    
    print("Generating graph adjacency matrices...")
    A_csr, actual_edges, actual_sparsity = generate_graph(args.vertices, args.sparsity, seed=42)
    B_csr, _, _ = generate_graph(args.vertices, args.sparsity, seed=123)
    print(f"  Actual edges: {actual_edges:,}")
    print(f"  Actual sparsity: {actual_sparsity:.4f}%")
    print()
    
    print("Converting to dense format...")
    A_dense = A_csr.toarray()
    B_dense = B_csr.toarray()
    print()
    
    print("Transferring to GPU...")
    A_sparse_gpu = csr_to_pytorch_sparse(A_csr, device)
    B_sparse_gpu = csr_to_pytorch_sparse(B_csr, device)
    A_dense_gpu = torch.from_numpy(A_dense).float().to(device)
    B_dense_gpu = torch.from_numpy(B_dense).float().to(device)
    print()
    
    print("Benchmarking GPU Sparse (COO)...")
    sparse_time = benchmark_gpu_sparse(A_sparse_gpu, B_sparse_gpu, args.num_runs)
    print(f"  GPU Sparse Time: {sparse_time:.6f}s")
    print()
    
    print("Benchmarking GPU Dense...")
    dense_time = benchmark_gpu_dense(A_dense_gpu, B_dense_gpu, args.num_runs)
    print(f"  GPU Dense Time: {dense_time:.6f}s")
    print()
    
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    sparse_memory_mb = (A_sparse_gpu._nnz() * 12) / (1024**2)
    dense_memory_mb = (args.vertices * args.vertices * 4) / (1024**2)
    memory_ratio = dense_memory_mb / sparse_memory_mb
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"GPU Sparse Time: {sparse_time:.6f}s")
    print(f"GPU Dense Time:  {dense_time:.6f}s")
    print(f"Speedup:         {speedup:.2f}×")
    print(f"Winner:          {winner}")
    print()
    print("Memory Usage:")
    print(f"  Sparse: {sparse_memory_mb:.2f} MB")
    print(f"  Dense:  {dense_memory_mb:.2f} MB")
    print(f"  Ratio:  {memory_ratio:.2f}× (Dense uses {memory_ratio:.2f}× more memory)")
    print()
    
    results = {
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'num_vertices': args.vertices,
        'target_sparsity': args.sparsity,
        'actual_edges': int(actual_edges),
        'actual_sparsity': float(actual_sparsity),
        'num_runs': args.num_runs,
        'sparse_time': float(sparse_time),
        'dense_time': float(dense_time),
        'speedup': float(speedup),
        'winner': winner,
        'sparse_memory_mb': float(sparse_memory_mb),
        'dense_memory_mb': float(dense_memory_mb),
        'memory_ratio': float(memory_ratio)
    }
    
    save_results(results, args.vertices, args.sparsity)
    
    sparsity_str = str(args.sparsity).replace('.', '_')
    print("=" * 80)
    print(f"Results saved to:")
    print(f"  - results/static_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_results.json")
    print(f"  - results/static_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_results.txt")
    print(f"  - results/static_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()

