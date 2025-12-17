================================================================================
    CSR GNN STATIC BENCHMARK - DATABASE I/O WORKFLOW
================================================================================


OVERVIEW
-----------------

2-layer Graph Convolutional Network (GCN) on citation networks using CSR sparse
format with database I/O workflow.

Workflow: Load graph from CSV - Run GCN in memory - Save embeddings to CSV


COMMANDS
-----------------

Run from project root (DB_Project_MatMul/):

Run benchmarks:
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\gnn_benchmark\gnn_static_benchmark.py --graph all
  .\venv313\Scripts\python.exe CSR_CSC_Database_Approach\gnn_benchmark\gnn_static_benchmark.py --graph cora


================================================================================
RESULTS
================================================================================

Device: CPU
Architecture: 2-layer GCN (128 - 64 - 32 dimensions)

Graph         Vertices    Edges      Total Time    I/O Read    GNN Compute    I/O Write    I/O %
--------------------------------------------------------------------------------
Cora          2,708       10,832     0.284s        0.123s      0.032s         0.128s       88.7%
CiteSeer      3,327       13,308     0.391s        0.172s      0.010s         0.209s       97.5%
PubMed        19,717      98,584     3.088s        2.525s      0.078s         0.486s       97.5%


KEY FINDINGS
-----------------

  * I/O dominates: 88.7-97.5% of total time spent on disk operations
  * Compute is fast: GNN layers only 2.5-11.3% of total time (0.010-0.078s per forward pass)
  * Database workflow trade-off: Persistent storage adds significant overhead for small compute tasks
  * Scalability: PubMed (7x more nodes) shows ~11x slowdown, mostly from I/O


PERFORMANCE BREAKDOWN
---------------------

Cora (smallest):
  * Read graph + features: 0.123s (43.4%)
  * 2-layer GCN forward: 0.032s (11.3%)
  * Write embeddings: 0.128s (45.1%)

PubMed (largest):
  * Read graph + features: 2.525s (81.8%)
  * 2-layer GCN forward: 0.078s (2.5%)
  * Write embeddings: 0.486s (15.7%)


TECHNICAL NOTES
-----------------

  * Format: CSR (scipy.sparse) for adjacency, dense PyTorch for features
  * Normalization: Symmetric normalization D^(-1/2) * (A + I) * D^(-1/2)
  * Graphs: Synthetic citation networks with power-law degree distribution
  * Features: Sparse binary features (128-dim, 5-20 non-zeros per node)


OUTPUT FILES
-----------------

  * data/ - Generated graph CSV files (edges + features)
  * results/cora_embeddings.csv - Node embeddings (2708 x 32)
  * results/citeseer_embeddings.csv - Node embeddings (3327 x 32)
  * results/pubmed_embeddings.csv - Node embeddings (19717 x 32)
  * results/metrics_gnn_static.json - Detailed timing metrics

