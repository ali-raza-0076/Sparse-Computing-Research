"""
CPU Dynamic Graph Benchmark: Incremental Edge Addition
Tests full recomputation vs incremental updates when adding 1, 2, or 3 edges.
Simulates real-world scenarios like adding friendships to a social network.
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
    Generate random graph adjacency matrix.
    
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


def add_edges_full_recomputation(base_matrix, new_edges, runs=3):
    """
    Add edges by rebuilding entire matrix from scratch.
    
    Args:
        base_matrix: Original CSR matrix
        new_edges: List of (row, col, val) tuples
        runs: Number of runs
    
    Returns:
        Average time in seconds
    """
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        
        coo = base_matrix.tocoo()
        rows = list(coo.row)
        cols = list(coo.col)
        vals = list(coo.data)
        
        for r, c, v in new_edges:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        
        new_matrix = sp.csr_matrix((vals, (rows, cols)), shape=base_matrix.shape)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def add_edges_incremental(base_matrix, new_edges, runs=3):
    """
    Add edges using incremental update (LIL→CSR conversion).
    
    Args:
        base_matrix: Original CSR matrix
        new_edges: List of (row, col, val) tuples
        runs: Number of runs
    
    Returns:
        Average time in seconds
    """
    times = []
    
    for _ in range(runs):
        start = time.perf_counter()
        
        lil = base_matrix.tolil()
        for r, c, v in new_edges:
            lil[r, c] += v
        
        new_matrix = lil.tocsr()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def save_results(results, nodes, sparsity, num_edges):
    """Save benchmark results in multiple formats."""
    os.makedirs('results', exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_filename = f'dynamic_graph_cpu_{nodes}vertices_{sparsity_str}pct_{num_edges}edges'
    
    with open(f'results/{base_filename}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CPU DYNAMIC GRAPH BENCHMARK: Incremental Edge Addition\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Graph Configuration:\n")
        f.write(f"  Vertices: {results['num_vertices']:,}\n")
        f.write(f"  Base Edges: {results['base_edges']:,}\n")
        f.write(f"  Sparsity: {results['actual_sparsity']:.4f}%\n")
        f.write(f"  New Edges Added: {results['num_new_edges']}\n")
        f.write(f"  Runs per test: {results['num_runs']}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Full Recomputation Time: {results['full_recomp_time']:.6f}s\n")
        f.write(f"  Incremental Update Time: {results['incremental_time']:.6f}s\n")
        f.write(f"  Speedup: {results['speedup']:.2f}×\n")
        f.write(f"  Winner: {results['winner']}\n")
    
    with open(f'results/{base_filename}_results.csv', 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Nodes,{results['num_vertices']}\n")
        f.write(f"Base_Edges,{results['base_edges']}\n")
        f.write(f"Sparsity,{results['actual_sparsity']}\n")
        f.write(f"New_Edges,{results['num_new_edges']}\n")
        f.write(f"Full_Recomp_Time_s,{results['full_recomp_time']}\n")
        f.write(f"Incremental_Time_s,{results['incremental_time']}\n")
        f.write(f"Speedup,{results['speedup']}\n")
        f.write(f"Winner,{results['winner']}\n")


def main():
    parser = argparse.ArgumentParser(description='CPU Dynamic Graph Benchmark')
    parser.add_argument('--vertices', type=int, default=1000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99, help='Sparsity percentage')
    parser.add_argument('--num-edges', type=int, default=1, help='Number of new edges to add (1, 2, or 3)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CPU DYNAMIC GRAPH BENCHMARK: Incremental Edge Addition")
    print("Full Recomputation vs Incremental Update")
    print("=" * 80)
    print()
    
    print(f"Configuration:")
    print(f"  Graph Size: {args.vertices} nodes")
    print(f"  Target Sparsity: {args.sparsity}%")
    print(f"  New Edges to Add: {args.num_edges}")
    print(f"  Runs per test: {args.num_runs}")
    print()
    
    print("Generating base graph...")
    base_graph, base_edges, actual_sparsity = generate_graph(args.vertices, args.sparsity)
    print(f"  Base edges: {base_edges:,}")
    print(f"  Actual sparsity: {actual_sparsity:.4f}%")
    print()
    
    np.random.seed(999)
    new_edges = [
        (np.random.randint(0, args.vertices), 
         np.random.randint(0, args.vertices), 
         1.0)
        for _ in range(args.num_edges)
    ]
    print(f"New edges to add: {new_edges}")
    print()
    
    print("Benchmarking Full Recomputation...")
    full_recomp_time = add_edges_full_recomputation(base_graph, new_edges, args.num_runs)
    print(f"  Full Recomputation Time: {full_recomp_time:.6f}s")
    print()
    
    print("Benchmarking Incremental Update...")
    incremental_time = add_edges_incremental(base_graph, new_edges, args.num_runs)
    print(f"  Incremental Update Time: {incremental_time:.6f}s")
    print()
    
    speedup = full_recomp_time / incremental_time
    winner = "Incremental" if speedup > 1 else "Full Recomputation"
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Full Recomputation: {full_recomp_time:.6f}s")
    print(f"Incremental Update: {incremental_time:.6f}s")
    print(f"Speedup:            {speedup:.2f}×")
    print(f"Winner:             {winner}")
    print()
    
    results = {
        'num_vertices': args.vertices,
        'base_edges': int(base_edges),
        'actual_sparsity': float(actual_sparsity),
        'num_new_edges': args.num_edges,
        'num_runs': args.num_runs,
        'full_recomp_time': float(full_recomp_time),
        'incremental_time': float(incremental_time),
        'speedup': float(speedup),
        'winner': winner
    }
    
    save_results(results, args.vertices, args.sparsity, args.num_edges)
    
    sparsity_str = str(args.sparsity).replace('.', '_')
    print("=" * 80)
    print(f"Results saved to:")
    print(f"  - results/dynamic_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.json")
    print(f"  - results/dynamic_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.txt")
    print(f"  - results/dynamic_graph_cpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()

