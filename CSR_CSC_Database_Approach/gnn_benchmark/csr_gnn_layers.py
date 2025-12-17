"""
CSR-based GNN Layer Implementations

Implements Graph Convolutional Network (GCN) layers using CSR sparse format.
Uses PyTorch CSR sparse tensors for efficient operations.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix


class CSRGCNLayer(nn.Module):
    """
    Graph Convolutional Network layer using CSR sparse format.
    
    Forward pass: H' = σ(A × H × W)
    where:
    - A: Normalized adjacency matrix (CSR format)
    - H: Node feature matrix (dense)
    - W: Learnable weight matrix
    - σ: Activation function (ReLU)
    """
    def __init__(self, in_features, out_features, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device=device) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        
    def forward(self, adj_csr, features):
        """
        Forward pass through GCN layer.
        
        Args:
            adj_csr: scipy.sparse.csr_matrix - normalized adjacency matrix
            features: torch.Tensor (N, in_features) - node features
            
        Returns:
            torch.Tensor (N, out_features) - updated node features
        """
        features = features.float()
        
        support = torch.matmul(features, self.weight)
        
        if self.device == 'cuda':
            crow_indices = torch.from_numpy(adj_csr.indptr).long().to(self.device)
            col_indices = torch.from_numpy(adj_csr.indices).long().to(self.device)
            values = torch.from_numpy(adj_csr.data).float().to(self.device)
            adj_sparse = torch.sparse_csr_tensor(crow_indices, col_indices, values, 
                                                   adj_csr.shape, device=self.device)
            output = torch.sparse.mm(adj_sparse, support)
        else:
            support_np = support.cpu().numpy()
            output_np = adj_csr.dot(support_np)
            output = torch.from_numpy(output_np).to(self.device)
        
        output = output + self.bias
        
        return output


class TwoLayerGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network using CSR format.
    
    Architecture:
        Input (in_features) → GCN(hidden) → ReLU → GCN(out_features)
    """
    def __init__(self, in_features, hidden_features, out_features, device='cpu'):
        super().__init__()
        self.device = device
        self.layer1 = CSRGCNLayer(in_features, hidden_features, device)
        self.layer2 = CSRGCNLayer(hidden_features, out_features, device)
        self.relu = nn.ReLU()
        
    def forward(self, adj_csr, features):
        """
        Forward pass through 2-layer GCN.
        
        Args:
            adj_csr: scipy.sparse.csr_matrix - normalized adjacency matrix
            features: torch.Tensor (N, in_features) - initial node features
            
        Returns:
            torch.Tensor (N, out_features) - final node embeddings
        """
        h = self.layer1(adj_csr, features)
        h = self.relu(h)
        
        out = self.layer2(adj_csr, h)
        
        return out


def normalize_adjacency_csr(adj_csr):
    """
    Normalize adjacency matrix: D^(-1/2) × A × D^(-1/2)
    where D is the degree matrix.
    
    Args:
        adj_csr: scipy.sparse.csr_matrix - adjacency matrix
        
    Returns:
        scipy.sparse.csr_matrix - normalized adjacency matrix
    """
    num_nodes = adj_csr.shape[0]
    adj_with_selfloops = adj_csr + csr_matrix(np.eye(num_nodes))
    
    degrees = np.array(adj_with_selfloops.sum(axis=1)).flatten()
    
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    d_inv_sqrt_mat = csr_matrix((d_inv_sqrt, (range(num_nodes), range(num_nodes))), 
                                  shape=(num_nodes, num_nodes))
    
    adj_normalized = d_inv_sqrt_mat @ adj_with_selfloops @ d_inv_sqrt_mat
    
    return adj_normalized


def load_graph_from_csv(edge_file):
    """
    Load graph from edge list CSV and convert to CSR format.
    
    Args:
        edge_file: Path to edge list CSV (source,target per line)
        
    Returns:
        adj_csr: scipy.sparse.csr_matrix - adjacency matrix
        num_nodes: int - number of nodes
    """
    edges = []
    max_node = 0
    with open(edge_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                src, dst = int(parts[0]), int(parts[1])
                edges.append((src, dst))
                max_node = max(max_node, src, dst)
    
    num_nodes = max_node + 1
    
    row_indices = [e[0] for e in edges]
    col_indices = [e[1] for e in edges]
    data = [1.0] * len(edges)
    
    adj_coo = coo_matrix((data, (row_indices, col_indices)), shape=(num_nodes, num_nodes))
    
    adj_csr = adj_coo.tocsr()
    
    return adj_csr, num_nodes
