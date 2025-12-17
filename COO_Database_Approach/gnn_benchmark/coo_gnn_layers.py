"""
COO-based GNN Layer Implementations

Implements Graph Convolutional Network (GCN) layers using COO sparse format.
Uses scipy.sparse for efficient COO operations.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix


class COOGCNLayer(nn.Module):
    """
    Graph Convolutional Network layer using COO sparse format.
    
    Forward pass: H' = σ(A × H × W)
    where:
    - A: Normalized adjacency matrix (COO format)
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
        
    def forward(self, adj_coo, features):
        """
        Forward pass through GCN layer.
        
        Args:
            adj_coo: scipy.sparse.coo_matrix - normalized adjacency matrix
            features: torch.Tensor (N, in_features) - node features
            
        Returns:
            torch.Tensor (N, out_features) - updated node features
        """
        features = features.float()
        
        support = torch.matmul(features, self.weight)
        
        if self.device == 'cuda':
            indices = torch.from_numpy(np.vstack([adj_coo.row, adj_coo.col])).long().to(self.device)
            values = torch.from_numpy(adj_coo.data).float().to(self.device)
            adj_sparse = torch.sparse_coo_tensor(indices, values, adj_coo.shape, device=self.device)
            output = torch.sparse.mm(adj_sparse, support)
        else:
            support_np = support.cpu().numpy()
            output_np = adj_coo.dot(support_np)
            output = torch.from_numpy(output_np).to(self.device)
        
        output = output + self.bias
        
        return output


class TwoLayerGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.
    
    Architecture:
        Input (in_features) → GCN(hidden) → ReLU → GCN(out_features)
    """
    def __init__(self, in_features, hidden_features, out_features, device='cpu'):
        super().__init__()
        self.device = device
        self.layer1 = COOGCNLayer(in_features, hidden_features, device)
        self.layer2 = COOGCNLayer(hidden_features, out_features, device)
        self.relu = nn.ReLU()
        
    def forward(self, adj_coo, features):
        """
        Forward pass through 2-layer GCN.
        
        Args:
            adj_coo: scipy.sparse.coo_matrix - normalized adjacency matrix
            features: torch.Tensor (N, in_features) - initial node features
            
        Returns:
            torch.Tensor (N, out_features) - final node embeddings
        """
        h = self.layer1(adj_coo, features)
        h = self.relu(h)
        
        out = self.layer2(adj_coo, h)
        
        return out


def normalize_adjacency_coo(adj_coo):
    """
    Normalize adjacency matrix: A_norm = D^(-1/2) × (A + I) × D^(-1/2)
    
    Args:
        adj_coo: scipy.sparse.coo_matrix - adjacency matrix
        
    Returns:
        scipy.sparse.coo_matrix - normalized adjacency with self-loops
    """
    n = adj_coo.shape[0]
    adj_with_self_loops = adj_coo + coo_matrix(np.eye(n))
    
    adj_csr = adj_with_self_loops.tocsr()
    
    degrees = np.array(adj_csr.sum(axis=1)).flatten()
    
    degrees_inv_sqrt = np.power(degrees, -0.5)
    degrees_inv_sqrt[np.isinf(degrees_inv_sqrt)] = 0.0
    
    from scipy.sparse import diags
    D_inv_sqrt = diags(degrees_inv_sqrt)
    
    adj_normalized = D_inv_sqrt @ adj_csr @ D_inv_sqrt
    
    return adj_normalized.tocoo()


def load_graph_from_csv(edge_file):
    """
    Load graph adjacency matrix from CSV edge list.
    
    Expected CSV format: source,target (no weights, undirected)
    
    Args:
        edge_file: Path to CSV file with edges
        
    Returns:
        scipy.sparse.coo_matrix - adjacency matrix
        int - number of nodes
    """
    edges = []
    max_node = 0
    
    with open(edge_file, 'r') as f:
        for line in f:
            src, tgt = map(int, line.strip().split(','))
            edges.append((src, tgt))
            max_node = max(max_node, src, tgt)
    
    num_nodes = max_node + 1
    
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    data = [1.0] * len(edges)
    
    adj_coo = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    return adj_coo, num_nodes
