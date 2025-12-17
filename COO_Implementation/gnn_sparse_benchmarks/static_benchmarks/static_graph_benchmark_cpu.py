"""
CPU Graph Benchmark: Static Graph Operations (COO Sparse vs Dense)
Tests graph operations across different sizes and sparsity levels.
Compares sparse COO matrix multiplication vs dense NumPy operations.
"""
import numpy as np
import time
import json
import os
import argparse
from scipy import sparse as sp
from tabulate import tabulate


def generate_graph(num_vertices, sparsity_percent, seed=42):
    """
    Generate random graph adjacency matrix in COO format.
    
    Args:
        num_vertices: Number of nodes in graph
        sparsity_percent: Sparsity level (90, 99, 99.9)
        seed: Random seed for reproducibility
    
    Returns:
        coo_matrix: Sparse COO adjacency matrix
        actual_edges: Number of edges
        actual_sparsity: Actual sparsity percentage
    """
    np.random.seed(seed)
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(num_vertices * num_vertices * density)
    
    rows = np.random.randint(0, num_vertices, size=num_edges)
    cols = np.random.randint(0, num_vertices, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(num_vertices, num_vertices))
    actual_edges = matrix.nnz
    actual_sparsity = 100 * (1 - actual_edges / (num_vertices * num_vertices))
    
    return matrix, actual_edges, actual_sparsity


def benchmark_cpu_sparse(A_coo, B_coo, runs=3):
    """
    Benchmark CPU sparse COO matrix multiplication.
    
    Args:
        A_coo: First sparse COO matrix
        B_coo: Second sparse COO matrix
        runs: Number of runs for averaging
    
    Returns:
        Average time in seconds
    """
    A_csr = A_coo.tocsr()
    B_csr = B_coo.tocsr()
    
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
    base_filename = f"static_graph_cpu_coo_{size}vertices_{sparsity_str}pct_results"
    
    with open(f'results/{base_filename}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CPU STATIC GRAPH BENCHMARK (COO): Sparse vs Dense\n")
        f.write("=" * 70 + "\n\n")
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
    parser = argparse.ArgumentParser(description='CPU Static Graph Benchmark (COO)')
    parser.add_argument('--vertices', type=int, default=10000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99.0, help='Sparsity percentage (90, 99, 99.9)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CPU STATIC GRAPH BENCHMARK (COO): Sparse vs Dense")
    print("=" * 70)
    print(f"\nGraph Size: {args.vertices} nodes")
    print(f"Sparsity: {args.sparsity}%")
    print(f"Runs: {args.num_runs}")
    
    print("\nGenerating graph adjacency matrix (COO format)...")
    A, edges, actual_sparsity = generate_graph(args.vertices, args.sparsity, seed=42)
    B, _, _ = generate_graph(args.vertices, args.sparsity, seed=123)
    
    print(f"Actual edges: {edges:,}")
    print(f"Actual sparsity: {actual_sparsity:.2f}%")
    
    print("Converting to dense arrays...")
    A_dense = A.toarray()
    B_dense = B.toarray()
    
    sparse_mem = (A.nnz * (8 + 4 + 4)) / (1024 * 1024)
    dense_mem = (args.vertices * args.vertices * 4) / (1024 * 1024)  # float32
    mem_ratio = dense_mem / sparse_mem if sparse_mem > 0 else 0
    
    print(f"\nMemory Usage:")
    print(f"  Sparse (COO): {sparse_mem:.2f} MB")
    print(f"  Dense:        {dense_mem:.2f} MB")
    print(f"  Ratio:        {mem_ratio:.2f}×")
    
    print(f"\nBenchmarking SPARSE CPU (COO, {args.num_runs} runs)...")
    sparse_time = benchmark_cpu_sparse(A, B, args.num_runs)
    print(f"  Average: {sparse_time:.6f}s")
    
    print(f"Benchmarking DENSE CPU ({args.num_runs} runs)...")
    dense_time = benchmark_cpu_dense(A_dense, B_dense, args.num_runs)
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
    
    result_files = f"static_graph_cpu_coo_{args.vertices}vertices_{str(args.sparsity).replace('.', '_')}pct_results.*"
    print(f"\nResults saved to: results/{result_files}")
    
    print("\n" + "=" * 70)
    print(f"CONCLUSION: {winner} wins with {speedup:.2f}× speedup")
    print("=" * 70)


if __name__ == "__main__":
    main()

