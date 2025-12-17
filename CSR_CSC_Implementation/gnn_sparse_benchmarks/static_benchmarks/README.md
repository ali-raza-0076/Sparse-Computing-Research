Static Graph Benchmarks (CPU & GPU)


Overview

Compares sparse vs dense matrix multiplication for static graph adjacency matrices on both CPU and GPU.


Execution


CPU Static Benchmark (scipy CSR)
``bash
python static_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-runs 3
`


GPU Static Benchmark (PyTorch COO)
`bash
python static_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-runs 3
`


Arguments | Argument | Default | Description |  | ---------- | --------- | ------------- |  | --vertices | 1000 | Number of graph vertices |  | --sparsity | 99 | Sparsity percentage (90, 99, 99.9) |  | --num-runs | 3 | Benchmark runs to average | Format & Implementation


CPU
- Format: scipy CSR (Compressed Sparse Row) - fully supported on CPU
- Operation: CSR × CSR matrix multiplication
- Library: scipy.sparse


GPU
- Input Format: scipy CSR (generation only)
- GPU Execution Format: PyTorch COO (Coordinate)
- Conversion Flow: CSR ? .tocoo() ? torch.sparse_coo_tensor() ? GPU
- Critical Limitation: PyTorch does NOT support CSR/CSC on GPU, only COO
- Why COO Only: NVIDIA cuSPARSE supports CSR, but PyTorch doesn't expose this API
- Synchronization: torch.cuda.synchronize() ensures accurate timing


Hardware

GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (5,888 CUDA cores, 12GB VRAM)  
CUDA: 13.0  
PyTorch: 2.9.1+cu130


Actual Results


20,000 Nodes @ 99.9% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 0.0669s | 16.3417s | 244.17× | 476.40× |  | GPU | COO | 0.1551s | 0.9321s | 6.01× | 333.50× | 20,000 Nodes @ 99% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 7.1543s | 42.0243s | 5.87× | 50.13× |  | GPU | COO | 1.0043s | 0.8976s | 0.89× (DENSE WINS) | 33.50× | 10,000 Nodes @ 99.9% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 0.0084s | 4.0845s | 486.85× | 476.44× |  | GPU | COO | 0.0222s | 0.1042s | 4.69× | 333.52× | 10,000 Nodes @ 99% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 0.9195s | 4.3792s | 4.76× | 49.99× |  | GPU | COO | 0.1010s | 0.1239s | 1.23× | 33.50× | 5,000 Nodes @ 99.9% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 0.0010s | 0.4675s | 445.76× | 454.75× |  | GPU | COO | 0.0051s | 0.0142s | 2.78× | 333.51× | 5,000 Nodes @ 99% Sparsity | Platform | Format | Sparse Time | Dense Time | Speedup | Memory Ratio |  | ---------- | -------- | ------------- | ------------ | --------- | -------------- |  | CPU | CSR | 0.0930s | 0.4605s | 4.95× | 49.74× |  | GPU | COO | 0.0127s | 0.0138s | 1.09× | 33.49× | Key Insights

1. CRITICAL: GPU Sparse Loses at 99% with Large Graphs: At 20K nodes @ 99%, GPU sparse (1.00s) is SLOWER than dense (0.90s). Dense wins with 0.89× "speedup" (actually sparse slowdown).

2. Sparsity is Everything: CPU speedup ranges from 244-487× at 99.9% but only 4.76-5.87× at 99%. GPU drops from 2.78-6.01× (99.9%) to 0.89-1.23× (99%).

3. CPU Dominates Ultra-Sparse: At 99.9% sparsity, CPU CSR achieves 244-487× speedup vs GPU COO's 2.78-6.01×.

4. Graph Size Impact on GPU: Larger graphs help GPU sparse performance at 99.9% (5K: 2.78×, 10K: 4.69×, 20K: 6.01×). But at 99%, large graphs hurt GPU sparse (20K: dense wins).

5. Format Matters: CSR (CPU) is far more efficient than COO (GPU) for ultra-sparse operations due to better cache locality and lower overhead.

6. Memory Savings: Consistent across all tests (33-476× less memory for sparse), even when sparse is slower (20K @ 99%).

7. Practical Takeaway: Use CPU CSR for ultra-sparse graphs (<99.9%). GPU sparse only viable at 99.9%+ sparsity, and even then CPU dominates by 40-80×.


Output Files

`
results/static_graph_cpu_{nodes}nodes_{sparsity}pct_results.{json,txt,csv}
results/static_graph_gpu_{nodes}nodes_{sparsity}pct_results.{json,txt,csv}
``


