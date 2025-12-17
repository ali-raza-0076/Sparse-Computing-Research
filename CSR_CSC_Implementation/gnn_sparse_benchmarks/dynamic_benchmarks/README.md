Dynamic Graph Benchmarks (CPU & GPU)


Overview

Compares incremental edge addition vs full matrix recomputation for dynamic graph updates on both CPU and GPU.


Execution


CPU Dynamic Benchmark (scipy CSR/LIL)
``bash
# Add 1 edge
python dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-edges 1 --num-runs 3

# Add 3 edges
python dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99 --num-edges 3 --num-runs 3
`


GPU Dynamic Benchmark (PyTorch COO)
`bash
# Add 2 edges
python dynamic_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-edges 2 --num-runs 3

# Add 3 edges
python dynamic_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-edges 3 --num-runs 3
`


Arguments | Argument | Default | Description |  | ---------- | --------- | ------------- |  | --vertices | 1000 | Number of graph vertices |  | --sparsity | 99 | Sparsity percentage (90, 99, 99.9) |  | --num-edges | 1 | New edges to add (1, 2, or 3) |  | --num-runs | 3 | Benchmark runs to average | Format & Implementation


Edge Generation
- Selection: Edges generated randomly using random.randint() for row/col indices
- Values: All new edges have weight = 1.0
- Uniqueness: No duplicate checking (edges may overwrite existing connections)
- Determinism: Same random seed used across runs for consistency


CPU Implementation
- Format: scipy CSR (base) / LIL (incremental updates)
- Full Recomputation: 
  1. Convert CSR ? COO
  2. Append new edges as (row, col, value) tuples
  3. Rebuild CSR from expanded COO
- Incremental Update: 
  1. Convert CSR ? LIL (allows efficient element modification)
  2. Add edges: lil[row, col] += value
  3. Convert LIL ? CSR for final result
- Library: scipy.sparse


GPU Implementation
- Input Format: scipy CSR (converted to PyTorch COO)
- Full Recomputation: 
  1. Concatenate new edge indices to existing COO indices
  2. Concatenate new values to existing COO values
  3. Rebuild sparse COO tensor on GPU
- Incremental Update: 
  1. Convert COO ? dense matrix on GPU
  2. Add edges: dense[row, col] += value
  3. Convert dense ? sparse COO
- Critical Note: PyTorch does NOT support CSR/CSC on GPU, only COO
- Synchronization: torch.cuda.synchronize() before/after operations for accurate timing


Hardware

GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (5,888 CUDA cores, 12GB VRAM)  
CUDA: 13.0  
PyTorch: 2.9.1+cu130


Actual Results


20,000 Nodes @ 99.9% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 0.1669s | 0.0537s | 3.11× | Incremental |  | CPU | 2 | 0.1940s | 0.0792s | 2.45× | Incremental |  | CPU | 3 | 0.1711s | 0.0507s | 3.37× | Incremental |  | GPU | 1 | 0.0187s | 0.0456s | 0.41× | Full Recomp |  | GPU | 2 | 0.0187s | 0.0463s | 0.40× | Full Recomp |  | GPU | 3 | 0.0185s | 0.0454s | 0.41× | Full Recomp | 20,000 Nodes @ 99% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 1.7720s | 0.2455s | 7.22× | Incremental |  | CPU | 2 | 1.6840s | 0.2503s | 6.73× | Incremental |  | CPU | 3 | 1.7118s | 0.2420s | 7.07× | Incremental |  | GPU | 1 | 0.0510s | 0.0490s | 1.04× (WINS) | Incremental |  | GPU | 2 | 0.0423s | 0.0459s | 0.92× | Full Recomp |  | GPU | 3 | 0.0446s | 0.0438s | 1.02× (WINS) | Incremental | 10,000 Nodes @ 99.9% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 0.0435s | 0.0145s | 3.00× | Incremental |  | CPU | 2 | 0.0442s | 0.0147s | 3.01× | Incremental |  | CPU | 3 | 0.0411s | 0.0126s | 3.27× | Incremental |  | GPU | 1 | 0.0190s | 0.0278s | 0.68× | Full Recomp |  | GPU | 2 | 0.0181s | 0.0280s | 0.65× | Full Recomp |  | GPU | 3 | 0.0181s | 0.0287s | 0.63× | Full Recomp | 10,000 Nodes @ 99% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 0.4481s | 0.0767s | 5.84× | Incremental |  | CPU | 2 | 0.4228s | 0.0757s | 5.59× | Incremental |  | CPU | 3 | 0.4440s | 0.0718s | 6.18× | Incremental |  | GPU | 1 | 0.0236s | 0.0283s | 0.83× | Full Recomp |  | GPU | 2 | 0.0224s | 0.0250s | 0.90× | Full Recomp |  | GPU | 3 | 0.0235s | 0.0265s | 0.89× | Full Recomp | 5,000 Nodes @ 99.9% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 0.0117s | 0.0040s | 2.93× | Incremental |  | CPU | 2 | 0.0121s | 0.0040s | 3.05× | Incremental |  | CPU | 3 | 0.0102s | 0.0043s | 2.38× | Incremental |  | GPU | 1 | 0.0238s | 0.0300s | 0.79× | Full Recomp |  | GPU | 2 | 0.0210s | 0.0240s | 0.87× | Full Recomp |  | GPU | 3 | 0.0181s | 0.0220s | 0.82× | Full Recomp | 5,000 Nodes @ 99% Sparsity | Platform | Edges Added | Full Recomp Time | Incremental Time | Speedup | Winner |  | ---------- | ------------- | ------------------ | ------------------ | --------- | -------- |  | CPU | 1 | 0.1187s | 0.0230s | 5.16× | Incremental |  | CPU | 2 | 0.1063s | 0.0223s | 4.77× | Incremental |  | CPU | 3 | 0.1114s | 0.0226s | 4.92× | Incremental |  | GPU | 1 | 0.0202s | 0.0238s | 0.85× | Full Recomp |  | GPU | 2 | 0.0198s | 0.0228s | 0.87× | Full Recomp |  | GPU | 3 | 0.0186s | 0.0236s | 0.79× | Full Recomp | Key Insights

1. BREAKTHROUGH: GPU Incremental Crossover Found! 
   - At 20K nodes @ 99% sparsity, GPU incremental finally wins (1.02-1.04×)
   - But at 99.9% sparsity, GPU incremental still loses badly (0.40-0.41×)
   - GPU incremental requires large graphs (20K+) AND moderate sparsity (99%)

2. CPU Incremental Dominates: CPU achieves 2.38-7.22× speedup across all tests.
   - Best performance: 20K nodes @ 99% (6.73-7.22×) - large dense graphs maximize recomputation cost
   - Consistent advantage at all graph sizes and sparsities

3. Sparsity Impact:
   - CPU: Higher speedup at 99% (5.16-7.22×) vs 99.9% (2.38-3.37×) because denser graphs = more expensive recomputation
   - GPU: Only wins at 99%, loses at 99.9% even with 20K nodes. COO?dense overhead scales with matrix size

4. Graph Size Scaling:
   - CPU: Larger graphs improve incremental advantage (5K: 4.92×, 10K: 6.18×, 20K: 7.22× @ 99%)
   - GPU: Needs 20K+ nodes @ 99% to barely break even. At 99.9%, larger graphs make incremental WORSE (20K: 0.41× vs 10K: 0.68×)

5. Platform Comparison:
   - CPU CSR?LIL?CSR: Efficient at ALL tested sizes (5-20K nodes)
   - GPU COO?dense?sparse: Only viable at 20K+ nodes @ 99% sparsity

6. Practical Recommendations for 1-3 edge updates:
   - Always use CPU incremental (2.4-7.2× speedup)
   - Avoid GPU incremental unless: graph >20K nodes AND sparsity ~99% (not 99.9%)
   - For ultra-sparse graphs (99.9%), GPU full recomputation is 60% faster than incremental even at 20K nodes

3. Real-World Relevance: Social networks, knowledge graphs, and GNNs frequently add small numbers of edges—incremental updates are critical.

4. Batch Considerations: For 100+ edge updates, full recomputation may become competitive (not tested here).


Output Files

`
results/dynamic_graph_cpu_{nodes}nodes_{sparsity}pct_{edges}edges_results.{json,txt,csv}
results/dynamic_graph_gpu_{nodes}nodes_{sparsity}pct_{edges}edges_results.{json,txt,csv}
``


