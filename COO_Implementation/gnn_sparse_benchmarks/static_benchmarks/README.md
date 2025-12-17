Static Graph Benchmarks - COO Format

Benchmarks for static graph operations comparing sparse COO vs dense matrix operations on CPU and GPU.


Format

COO (Coordinate): Natural format for graphs - stores edge list as (row, col, value) triplets.

CPU Implementation Note: 
- Graphs generated in COO format
- Converted to CSR for multiplication (scipy limitation: COO × COO not supported)
- Conversion done once before timing to ensure fair benchmarking

GPU Implementation: 
- Uses PyTorch native COO sparse tensors
- No format conversion needed (PyTorch supports COO × COO directly)


Usage

``bash
# CPU benchmarks
python static_graph_benchmark_cpu.py --vertices 10000 --sparsity 99 --num-runs 3

# GPU benchmarks
python static_graph_benchmark_gpu.py --vertices 10000 --sparsity 99 --num-runs 3
`

Arguments:
- --vertices: Number of graph vertices
- --sparsity: Sparsity percentage (90, 99, 99.9)
- --num-runs: Number of runs for averaging

Results saved to: results/static_graph_{cpu/gpu}_coo_{nodes}nodes_{sparsity}pct_results.{json,txt,csv}`


Benchmark Results


Hardware Configuration
- CPU: AMD Ryzen 9 8940HX (16 cores used)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- CUDA: Version 13.0
- PyTorch: Version 2.9.1+cu130


CPU Results (COO ? CSR conversion) | Nodes | Sparsity | Edges | Sparse Time | Dense Time | Speedup | Memory Ratio |  | -------- | ---------- | ----------- | ------------- | ------------ | ---------- | -------------- |  | 5,000 | 99.0% | 250,000 | 0.085341s | 0.470971s | 5.52× | 25× |  | 5,000 | 99.9% | 24,999 | 0.000807s | 0.442243s | 547.74× | 250× |  | 10,000 | 99.0% | 1,000,000 | 0.984908s | 5.317314s | 5.40× | 25× |  | 10,000 | 99.9% | 99,999 | 0.005651s | 4.060758s | 718.62× | 250× |  | 20,000 | 99.0% | 4,000,000 | 7.455243s | 39.335607s | 5.28× | 25× |  | 20,000 | 99.9% | 399,999 | 0.052206s | 35.777243s | 685.30× | 250× | GPU Results (Native COO) | Nodes | Sparsity | Edges | Sparse Time | Dense Time | Speedup | Memory Ratio |  | -------- | ---------- | ----------- | ------------- | ------------ | ---------- | -------------- |  | 5,000 | 99.0% | 250,000 | 0.008873s | 0.012963s | 1.46× | 25× |  | 5,000 | 99.9% | 24,999 | 0.001974s | 0.013469s | 6.82× | 250× |  | 10,000 | 99.0% | 1,000,000 | 0.039951s | 0.100662s | 2.52× | 25× |  | 10,000 | 99.9% | 99,999 | 0.001712s | 0.100302s | 58.58× | 250× |  | 20,000 | 99.0% | 4,000,000 | OOM Error | N/A | N/A | 25× |  | 20,000 | 99.9% | 399,999 | 0.008706s | 1.273960s | 146.34× | 250× | Analysis

CPU Performance:
- Consistent 5-6× speedup at 99% sparsity across all sizes
- Dramatic 550-720× speedup at 99.9% sparsity
- Performance scales well with graph size

GPU Performance:
- Modest speedups at 99% sparsity (1.5-2.5×)
- Strong speedups at 99.9% sparsity (7-146×)
- 20,000 nodes at 99% hits GPU memory limit (requires 17.72 GB workspace)

Key Findings:
- Sparse operations essential for high sparsity graphs (99.9%)
- GPU benefits less than CPU at moderate sparsity (99%)
- Memory savings consistent (25-250×) regardless of platform
- Extreme sparsity (99.9%) shows GPU accelerates sparse by 146× vs dense

