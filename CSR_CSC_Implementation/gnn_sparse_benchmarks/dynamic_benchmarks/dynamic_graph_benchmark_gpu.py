"""
GPU Dynamic Graph Benchmark: Incremental Edge Addition
Tests full recomputation vs incremental updates when adding 1, 2, or 3 edges on GPU.
Uses PyTorch sparse tensors with COO format.
"""
import torch
import numpy as np
import time
import json
import os
import argparse
from scipy import sparse as sp


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
    """Generate random graph adjacency matrix."""
    np.random.seed(seed)
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(num_vertices * num_vertices * density)
    
    rows = np.random.randint(0, num_vertices, size=num_edges)
    cols = np.random.randint(0, num_vertices, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_vertices, num_vertices))
    return matrix, matrix.nnz, 100 * (1 - matrix.nnz / (num_vertices * num_vertices))


def csr_to_pytorch_sparse(csr_matrix, device):
    """Convert scipy CSR to PyTorch COO sparse tensor."""
    coo = csr_matrix.tocoo()
    indices = torch.LongTensor([coo.row, coo.col]).to(device)
    values = torch.FloatTensor(coo.data).to(device)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def add_edges_full_recomputation_gpu(sparse_tensor, new_edges, device, runs=3):
    """
    Add edges by rebuilding entire sparse tensor from scratch.
    
    Args:
        sparse_tensor: Original sparse tensor
        new_edges: List of (row, col, val) tuples
        device: GPU device
        runs: Number of runs
    
    Returns:
        Average time in seconds
    """
    times = []
    
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        indices = sparse_tensor.coalesce().indices()
        values = sparse_tensor.coalesce().values()
        
        new_indices = torch.LongTensor([[e[0] for e in new_edges], 
                                        [e[1] for e in new_edges]]).to(device)
        new_values = torch.FloatTensor([e[2] for e in new_edges]).to(device)
        
        all_indices = torch.cat([indices, new_indices], dim=1)
        all_values = torch.cat([values, new_values])
        
        new_tensor = torch.sparse_coo_tensor(all_indices, all_values, 
                                             sparse_tensor.shape, device=device)
        new_tensor = new_tensor.coalesce()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def add_edges_incremental_gpu(sparse_tensor, new_edges, device, runs=3):
    """
    Add edges using incremental update with index_add_.
    
    Args:
        sparse_tensor: Original sparse tensor
        new_edges: List of (row, col, val) tuples
        device: GPU device
        runs: Number of runs
    
    Returns:
        Average time in seconds
    """
    times = []
    shape = sparse_tensor.shape
    
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        dense = sparse_tensor.to_dense()
        
        for r, c, v in new_edges:
            dense[r, c] += v
        
        new_tensor = dense.to_sparse()
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times)


def save_results(results, nodes, sparsity, num_edges):
    """Save benchmark results in multiple formats."""
    os.makedirs('results', exist_ok=True)
    
    sparsity_str = str(sparsity).replace('.', '_')
    base_filename = f'dynamic_graph_gpu_{nodes}vertices_{sparsity_str}pct_{num_edges}edges'
    
    with open(f'results/{base_filename}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f'results/{base_filename}_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU DYNAMIC GRAPH BENCHMARK: Incremental Edge Addition\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"GPU: {results['gpu_name']}\n")
        f.write(f"CUDA Version: {results['cuda_version']}\n\n")
        
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
        f.write(f"GPU,{results['gpu_name']}\n")
        f.write(f"Nodes,{results['num_vertices']}\n")
        f.write(f"Base_Edges,{results['base_edges']}\n")
        f.write(f"Sparsity,{results['actual_sparsity']}\n")
        f.write(f"New_Edges,{results['num_new_edges']}\n")
        f.write(f"Full_Recomp_Time_s,{results['full_recomp_time']}\n")
        f.write(f"Incremental_Time_s,{results['incremental_time']}\n")
        f.write(f"Speedup,{results['speedup']}\n")
        f.write(f"Winner,{results['winner']}\n")


def main():
    parser = argparse.ArgumentParser(description='GPU Dynamic Graph Benchmark')
    parser.add_argument('--vertices', type=int, default=1000, help='Number of vertices')
    parser.add_argument('--sparsity', type=float, default=99, help='Sparsity percentage')
    parser.add_argument('--num-edges', type=int, default=1, help='Number of new edges to add (1, 2, or 3)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPU DYNAMIC GRAPH BENCHMARK: Incremental Edge Addition")
    print("Full Recomputation vs Incremental Update")
    print("=" * 80)
    print()
    
    if not check_gpu():
        return
    print()
    
    device = torch.device('cuda')
    
    print(f"Configuration:")
    print(f"  Graph Size: {args.vertices} nodes")
    print(f"  Target Sparsity: {args.sparsity}%")
    print(f"  New Edges to Add: {args.num_edges}")
    print(f"  Runs per test: {args.num_runs}")
    print()
    
    print("Generating base graph...")
    base_graph_csr, base_edges, actual_sparsity = generate_graph(args.vertices, args.sparsity)
    print(f"  Base edges: {base_edges:,}")
    print(f"  Actual sparsity: {actual_sparsity:.4f}%")
    print()
    
    print("Transferring to GPU...")
    base_graph_gpu = csr_to_pytorch_sparse(base_graph_csr, device)
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
    full_recomp_time = add_edges_full_recomputation_gpu(base_graph_gpu, new_edges, device, args.num_runs)
    print(f"  Full Recomputation Time: {full_recomp_time:.6f}s")
    print()
    
    print("Benchmarking Incremental Update...")
    incremental_time = add_edges_incremental_gpu(base_graph_gpu, new_edges, device, args.num_runs)
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
        'gpu_name': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
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
    print(f"  - results/dynamic_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.json")
    print(f"  - results/dynamic_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.txt")
    print(f"  - results/dynamic_graph_gpu_{args.vertices}vertices_{sparsity_str}pct_{args.num_edges}edges_results.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()

