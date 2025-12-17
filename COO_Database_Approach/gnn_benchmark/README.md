================================================================================
    COO GNN STATIC BENCHMARK - DATABASE I/O WORKFLOW
================================================================================


OVERVIEW
-----------------

2-layer Graph Convolutional Network (GCN) on citation networks using COO sparse
format with database I/O workflow.

Workflow: Load graph from CSV - Run GCN in memory - Save embeddings to CSV


COMMANDS
-----------------

Run from project root (DB_Project_MatMul/):

Generate synthetic graph data (run once):
  .\venv313\Scripts\python.exe COO_Implementation\gnn_benchmark\generate_data.py

Run benchmarks:
  .\venv313\Scripts\python.exe COO_Implementation\gnn_benchmark\gnn_static_benchmark.py --graph all
  .\venv313\Scripts\python.exe COO_Implementation\gnn_benchmark\gnn_static_benchmark.py --graph cora


================================================================================
RESULTS
================================================================================

Device: CPU (uses CPU for GNN operations with COO sparse format)
Architecture: 2-layer GCN (128 - 64 - 32 dimensions)

Graph         Vertices    Edges      Total Time    I/O Read    GNN Compute    I/O Write    I/O %
--------------------------------------------------------------------------------
Cora          2,708       10,832     0.289s        0.101s      0.012s         0.175s       95.7%
CiteSeer      3,327       13,308     0.407s        0.159s      0.013s         0.235s       96.8%


KEY FINDINGS
-----------------

  * I/O dominates: 95-97% of total time spent on disk operations
  * Compute is fast: GNN layers only 3-4% of total time (0.001-0.028s per forward pass)
  * Database workflow trade-off: Persistent storage adds significant overhead for small compute tasks
  * Scalability: PubMed (7x more nodes) shows ~14x slowdown, mostly from I/O


PERFORMANCE BREAKDOWN
---------------------

Cora (smallest):
  * Read graph + features: 0.101s (35%)
  * 2-layer GCN forward: 0.012s (4%)
  * Write embeddings: 0.175s (61%)

PubMed (largest):
  * Read graph + features: 3.316s (83%)
  * 2-layer GCN forward: 0.107s (3%)
  * Write embeddings: 0.550s (14%)


TECHNICAL NOTES
-----------------

  * Format: COO (scipy.sparse) for adjacency, dense PyTorch for features
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


