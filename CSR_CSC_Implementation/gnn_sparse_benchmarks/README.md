GNN Sparse Graph Benchmarks


Overview

Benchmarks for Graph Neural Network (GNN) operations comparing sparse vs dense performance on CPU and GPU.

Two Test Types:
1. Static Graphs - Sparse vs dense matrix multiplication
2. Dynamic Updates - Incremental edge addition (1, 2, or 3 edges) vs full recomputation


Structure

``
gnn_sparse_benchmarks/
+-- static_benchmarks/           # Static graph: Sparse vs Dense
|   +-- static_graph_benchmark_cpu.py (scipy CSR)
|   +-- static_graph_benchmark_gpu.py (PyTorch COO)
|   +-- README.md (static results)
|   +-- results/
+-- dynamic_benchmarks/          # Dynamic updates: Incremental vs Full Recomp
|   +-- dynamic_graph_benchmark_cpu.py (scipy CSR/LIL)
|   +-- dynamic_graph_benchmark_gpu.py (PyTorch COO)
|   +-- README.md (dynamic results)
|   +-- results/
+-- cpu_graph_benchmarks/ (legacy - to be removed)
+-- gpu_graph_benchmarks/ (legacy - to be removed)
`

Critical Format Note:
- CPU: Uses scipy CSR format directly (fully supported)
- GPU: Generates CSR initially, but PyTorch automatically converts to COO on GPU (PyTorch does NOT support CSR/CSC on GPU, only COO)


Quick Execution


Static Benchmarks
`bash
cd static_benchmarks

# CPU: 10,000 nodes, 99.9% sparse
python static_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-runs 3

# GPU: 10,000 nodes, 99.9% sparse
python static_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-runs 3
`


Dynamic Benchmarks
`bash
cd dynamic_benchmarks

# CPU: Add 3 edges
python dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-edges 3 --num-runs 3

# GPU: Add 2 edges
python dynamic_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-edges 2 --num-runs 3
`


Key Results


Static Graphs - Actual Results

10,000 Nodes @ 99.9% Sparsity:
- CPU Sparse (CSR): 0.0084s | Dense: 4.0845s | Speedup: 486.85x | Memory Ratio: 476.44x
- GPU Sparse (COO): 0.0222s | Dense: 0.1042s | Speedup: 4.69x | Memory Ratio: 333.52x

10,000 Nodes @ 99% Sparsity:
- CPU Sparse (CSR): 0.9195s | Dense: 4.3792s | Speedup: 4.76x | Memory Ratio: 49.99x
- GPU Sparse (COO): 0.1010s | Dense: 0.1239s | Speedup: 1.23x | Memory Ratio: 33.50x

Pattern: Ultra-high sparsity (99.9%) dramatically favors CPU sparse (486x vs 4.7x). At 99% sparsity, advantage drops significantly (CPU: 4.8x vs GPU: 1.2x). Sparse formats excel at 99.9%+.


Dynamic Updates
- Incremental consistently outperforms full recomputation
- CPU: 2-7x speedup (expected)
- GPU: 6-30x speedup (expected, larger graphs)


Output Files

Results saved as: {test}_{platform}_{nodes}nodes_{sparsity}pct[_{edges}edges]_results.{json,txt,csv}

Example: static_graph_cpu_1000nodes_99pct_results.json


Requirements

`bash
# CPU
pip install numpy scipy

# GPU (add to above)
pip install torch --index-url https://download.pytorch.org/whl/cu118
``


Hardware

- CPU: Standard processor, no special requirements
- GPU: NVIDIA with CUDA support (tested on RTX 5070 Ti, 5888 cores, 12GB VRAM)

See subdirectory READMEs for detailed results and analysis.


