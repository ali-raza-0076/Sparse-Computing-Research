"""
CPU Graph Benchmark: Static Graph Operations (CSR Sparse vs Dense)
Tests graph operations across different sizes and sparsity levels.
Compares sparse CSR matrix multiplication vs dense NumPy operations.
"""
import numpy as np
import time
import json
import os
import argparse
from scipy import sparse as sp
from tabulate import tabulate


def generate_graph(num_nodes, sparsity_percent, seed=42):
    """
    Generate random graph adjacency matrix.
    
    Args:
        num_nodes: Number of nodes in graph
        sparsity_percent: Sparsity level (90, 99, 99.9)
        seed: Random seed for reproducibility
    
    Returns:
        csr_matrix: Sparse CSR adjacency matrix
        actual_edges: Number of edges
        actual_sparsity: Actual sparsity percentage
    """
    np.random.seed(seed)
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(num_nodes * num_nodes * density)
    
    rows = np.random.randint(0, num_nodes, size=num_edges)
    cols = np.random.randint(0, num_nodes, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    actual_edges = matrix.nnz
    actual_sparsity = 100 * (1 - actual_edges / (num_nodes * num_nodes))
    
    return matrix, actual_edges, actual_sparsity


def benchmark_cpu_sparse(A_csr, B_csr, runs=3):
    """
    Benchmark CPU sparse CSR matrix multiplication.
    
    Args:
        A_csr: First sparse matrix
        B_csr: Second sparse matrix
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    _ = A_csr @ B_csr
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = A_csr @ B_csr
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def benchmark_cpu_dense(A_dense, B_dense, runs=3):
    """
    Benchmark CPU dense NumPy matrix multiplication.
    
    Args:
        A_dense: First dense array
        B_dense: Second dense array
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    _ = A_dense @ B_dense
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = A_dense @ B_dense
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def save_results(results, size, sparsity):
    """Save benchmark results in multiple formats."""
    os.makedirs('results', exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_filename = f'static_graph_cpu_{size}nodes_{sparsity_str}pct'
    
    with open(f'results/{base_filename}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CPU STATIC GRAPH BENCHMARK: Sparse CSR vs Dense\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Graph Configuration:\n")
        f.write(f"  Vertices: {results['num_vertices']:,}\n")
        f.write(f"  Target Sparsity: {results['target_sparsity']}%\n")
        f.write(f"  Actual Edges: {results['actual_edges']:,}\n")
        f.write(f"  Actual Sparsity: {results['actual_sparsity']:.4f}%\n")
        f.write(f"  Runs per test: {results['num_runs']}\n\n")
        
        f.write("Results:\n")
        f.write(f"  CPU Sparse (CSR) Time: {results['sparse_time']:.6f}s\n")
        f.write(f"  CPU Dense (NumPy) Time: {results['dense_time']:.6f}s\n")
        f.write(f"  Speedup: {results['speedup']:.2f}×\n")
        f.write(f"  Winner: {results['winner']}\n\n")
        
        f.write("Memory Usage:\n")
        f.write(f"  Sparse: {results['sparse_memory_mb']:.2f} MB\n")
        f.write(f"  Dense: {results['dense_memory_mb']:.2f} MB\n")
        f.write(f"  Memory Ratio: {results['memory_ratio']:.2f}× savings\n")
    
    with open(f'results/{base_filename}_results.csv', 'w') as f:
        f.write("Metric,Value\n")
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
    parser = argparse.ArgumentParser(description='CPU Static Graph Benchmark')
    parser.add_argument('--vertices', type=int, default=1000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99, help='Sparsity percentage')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CPU STATIC GRAPH BENCHMARK: Sparse CSR vs Dense")
    print("=" * 80)
    print()
    
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
    
    print("Benchmarking CPU Sparse (CSR)...")
    sparse_time = benchmark_cpu_sparse(A_csr, B_csr, args.num_runs)
    print(f"  CPU Sparse Time: {sparse_time:.6f}s")
    print()
    
    print("Benchmarking CPU Dense (NumPy)...")
    dense_time = benchmark_cpu_dense(A_dense, B_dense, args.num_runs)
    print(f"  CPU Dense Time: {dense_time:.6f}s")
    print()
    
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    sparse_memory_mb = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes) / (1024**2)
    dense_memory_mb = A_dense.nbytes / (1024**2)
    memory_ratio = dense_memory_mb / sparse_memory_mb
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"CPU Sparse Time: {sparse_time:.6f}s")
    print(f"CPU Dense Time:  {dense_time:.6f}s")
    print(f"Speedup:         {speedup:.2f}×")
    print(f"Winner:          {winner}")
    print()
    print("Memory Usage:")
    print(f"  Sparse: {sparse_memory_mb:.2f} MB")
    print(f"  Dense:  {dense_memory_mb:.2f} MB")
    print(f"  Ratio:  {memory_ratio:.2f}× (Dense uses {memory_ratio:.2f}× more memory)")
    print()
    
    results = {
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
    
    print("=" * 80)
    sparsity_str = str(args.sparsity).replace('.', '_')
    print(f"Results saved to:")
    print(f"  - results/static_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_results.json")
    print(f"  - results/static_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_results.txt")
    print(f"  - results/static_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
