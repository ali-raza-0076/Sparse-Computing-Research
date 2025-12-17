"""
Generate synthetic graph datasets for COO GNN benchmarks.

Creates three citation-like graphs with realistic sizes:
- Small: 2708 nodes (Cora-like)
- Medium: 3327 nodes (CiteSeer-like)
- Large: 19717 nodes (PubMed-like)

Saves as CSV files for database I/O workflow.
"""
import numpy as np
import csv
import os


def generate_citation_graph(num_nodes, avg_degree=4, feature_dim=128, seed=42):
    """
    Generate a synthetic citation network graph.
    
    Args:
        num_nodes: Number of nodes in the graph
        avg_degree: Average degree per node
        feature_dim: Dimension of node features
        seed: Random seed
        
    Returns:
        edges: List of (source, target) tuples
        features: Node feature matrix (num_nodes × feature_dim)
    """
    np.random.seed(seed)
    
    edges = set()
    num_edges_target = num_nodes * avg_degree // 2  # Undirected
    
    degrees = np.zeros(num_nodes)
    
    for i in range(min(10, num_nodes)):
        for j in range(i + 1, min(10, num_nodes)):
            edges.add((min(i, j), max(i, j)))
            degrees[i] += 1
            degrees[j] += 1
    
    while len(edges) < num_edges_target:
        src = np.random.randint(0, num_nodes)
        
        probs = (degrees + 1) / (degrees.sum() + num_nodes)
        tgt = np.random.choice(num_nodes, p=probs)
        
        if src != tgt:
            edge = (min(src, tgt), max(src, tgt))
            if edge not in edges:
                edges.add(edge)
                degrees[src] += 1
                degrees[tgt] += 1
    
    edge_list = []
    for src, tgt in edges:
        edge_list.append((src, tgt))
        edge_list.append((tgt, src))
    
    features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    for i in range(num_nodes):
        num_features = np.random.randint(5, 21)
        feature_indices = np.random.choice(feature_dim, num_features, replace=False)
        features[i, feature_indices] = 1.0
    
    return edge_list, features


def save_graph_to_csv(name, edges, features, output_dir='data'):
    """Save graph to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    edge_file = os.path.join(output_dir, f'{name}_edges.csv')
    with open(edge_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for src, tgt in edges:
            writer.writerow([src, tgt])
    
    feature_file = os.path.join(output_dir, f'{name}_features.csv')
    with open(feature_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in features:
            writer.writerow(row)
    
    print(f"Generated {name}:")
    print(f"  Nodes: {features.shape[0]}, Edges: {len(edges)}, Features: {features.shape[1]}")
    print(f"  Saved to: {edge_file}, {feature_file}")


def main():
    print("Generating synthetic citation graphs for COO GNN benchmarks...\n")
    
    print("[1/3] Small graph (Cora-like)...")
    edges, features = generate_citation_graph(
        num_nodes=2708,
        avg_degree=4,
        feature_dim=128,
        seed=42
    )
    save_graph_to_csv('cora', edges, features)
    
    print("\n[2/3] Medium graph (CiteSeer-like)...")
    edges, features = generate_citation_graph(
        num_nodes=3327,
        avg_degree=4,
        feature_dim=128,
        seed=43
    )
    save_graph_to_csv('citeseer', edges, features)
    
    print("\n[3/3] Large graph (PubMed-like)...")
    edges, features = generate_citation_graph(
        num_nodes=19717,
        avg_degree=5,
        feature_dim=128,
        seed=44
    )
    save_graph_to_csv('pubmed', edges, features)
    
    print("\nAll graphs generated successfully!")


if __name__ == '__main__':
    main()
