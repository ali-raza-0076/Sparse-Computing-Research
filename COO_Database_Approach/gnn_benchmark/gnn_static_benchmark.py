"""
COO GNN Static Benchmark - Database I/O Workflow

Benchmarks 2-layer GCN on citation networks using COO sparse format with database I/O:
- Phase 1: Read graph structure and features from disk (CSV)
- Phase 2: Run 2-layer GCN forward pass in memory (GPU if available)
- Phase 3: Write final node embeddings to disk (CSV)

Tests on: Cora, CiteSeer, PubMed datasets
"""
import numpy as np
import torch
import time
import csv
import os
import json
import argparse
from coo_gnn_layers import TwoLayerGCN, normalize_adjacency_coo, load_graph_from_csv


def run_gnn_benchmark(graph_name, edge_file, feature_file, output_dir='results'):
    """
    Run GNN benchmark with database I/O workflow.
    
    Args:
        graph_name: Name of graph (e.g., 'cora')
        edge_file: Path to edge list CSV
        feature_file: Path to node features CSV
        output_dir: Directory to save results
    """
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"DATABASE I/O WORKFLOW: COO GNN Benchmark - {graph_name.upper()}")
    print(f"{'='*70}")
    print(f"Device: {device} (GPU sm_120 not supported by PyTorch)")
    
    print(f"\n[PHASE 1] Reading from secondary storage...")
    io_read_start = time.perf_counter()
    
    print(f"  Loading graph: {edge_file}")
    adj_coo, num_nodes = load_graph_from_csv(edge_file)
    num_edges = adj_coo.nnz
    print(f"    Nodes: {num_nodes:,}, Edges: {num_edges:,}")
    
    print(f"  Normalizing adjacency matrix...")
    adj_norm = normalize_adjacency_coo(adj_coo)
    
    print(f"  Loading features: {feature_file}")
    features = []
    with open(feature_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features.append([float(x) for x in row])
    features = np.array(features, dtype=np.float32)
    num_features = features.shape[1]
    print(f"    Feature dimension: {num_features}")
    
    io_read_end = time.perf_counter()
    io_read_time = io_read_end - io_read_start
    print(f"  I/O Read Time: {io_read_time:.6f}s")
    
    print(f"\n[PHASE 2] Processing with 2-layer GCN...")
    compute_start = time.perf_counter()
    
    features_tensor = torch.from_numpy(features).to(device)
    
    hidden_dim = 64
    output_dim = 32
    model = TwoLayerGCN(num_features, hidden_dim, output_dim, device=device)
    model.eval()
    
    layer_times = []
    
    with torch.no_grad():
        _ = model(adj_norm, features_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        num_runs = 3
        for run in range(num_runs):
            run_start = time.perf_counter()
            embeddings = model(adj_norm, features_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            run_end = time.perf_counter()
            layer_times.append(run_end - run_start)
    
    compute_end = time.perf_counter()
    compute_time = compute_end - compute_start
    avg_forward_time = np.mean(layer_times)
    std_forward_time = np.std(layer_times)
    
    print(f"  Forward pass time: {avg_forward_time:.6f}s (avg over {num_runs} runs ± {std_forward_time:.6f}s)")
    print(f"  Total compute time: {compute_time:.6f}s")
    print(f"  Output embeddings: ({num_nodes}, {output_dim})")
    
    print(f"\n[PHASE 3] Writing to secondary storage...")
    io_write_start = time.perf_counter()
    
    embeddings_np = embeddings.cpu().numpy()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{graph_name}_embeddings.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in embeddings_np:
            writer.writerow(row)
    
    io_write_end = time.perf_counter()
    io_write_time = io_write_end - io_write_start
    print(f"  I/O Write Time: {io_write_time:.6f}s")
    print(f"  Embeddings saved to: {output_file}")
    
    total_time = io_read_time + compute_time + io_write_time
    io_overhead = io_read_time + io_write_time
    io_percent = (io_overhead / total_time) * 100
    compute_percent = (compute_time / total_time) * 100
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY - {graph_name.upper()}")
    print(f"{'='*70}")
    print(f"Total Time:       {total_time:.6f}s")
    print(f"  Phase 1 (I/O Read):  {io_read_time:.6f}s ({io_read_time/total_time*100:.2f}%)")
    print(f"  Phase 2 (GNN Compute): {compute_time:.6f}s ({compute_percent:.2f}%)")
    print(f"    - Avg forward pass: {avg_forward_time:.6f}s")
    print(f"  Phase 3 (I/O Write): {io_write_time:.6f}s ({io_write_time/total_time*100:.2f}%)")
    print(f"I/O Overhead:     {io_overhead:.6f}s ({io_percent:.2f}%)")
    print(f"{'='*70}\n")
    
    metrics = {
        'graph': graph_name,
        'device': str(device),
        'num_nodes': int(num_nodes),
        'num_edges': int(num_edges),
        'num_features': int(num_features),
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'total_time': float(total_time),
        'io_read_time': float(io_read_time),
        'compute_time': float(compute_time),
        'avg_forward_time': float(avg_forward_time),
        'std_forward_time': float(std_forward_time),
        'io_write_time': float(io_write_time),
        'io_overhead': float(io_overhead),
        'io_percent': float(io_percent),
        'compute_percent': float(compute_percent)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='COO GNN Static Benchmark')
    parser.add_argument('--graph', type=str, default='all', 
                       choices=['cora', 'citeseer', 'pubmed', 'all'],
                       help='Graph to benchmark (default: all)')
    args = parser.parse_args()
    
    datasets = {
        'cora': {
            'edges': 'data/cora_edges.csv',
            'features': 'data/cora_features.csv'
        },
        'citeseer': {
            'edges': 'data/citeseer_edges.csv',
            'features': 'data/citeseer_features.csv'
        },
        'pubmed': {
            'edges': 'data/pubmed_edges.csv',
            'features': 'data/pubmed_features.csv'
        }
    }
    
    if args.graph == 'all':
        graphs_to_run = ['cora', 'citeseer', 'pubmed']
    else:
        graphs_to_run = [args.graph]
    
    all_metrics = {}
    for graph_name in graphs_to_run:
        dataset = datasets[graph_name]
        metrics = run_gnn_benchmark(
            graph_name, 
            dataset['edges'], 
            dataset['features'],
            output_dir='results'
        )
        all_metrics[graph_name] = metrics
    
    metrics_file = 'results/metrics_gnn_static.json'
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"All metrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
