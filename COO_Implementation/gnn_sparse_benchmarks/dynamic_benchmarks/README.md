Dynamic Graph Benchmarks - COO Format

Compares incremental edge addition vs full matrix recomputation for dynamic graph updates on CPU and GPU.


Format

COO (Coordinate): Natural format for graphs - stores edge list as (row, col, value) triplets.

CPU Implementation:
- Graphs generated in COO format
- Full recomputation: Append edges to COO arrays and rebuild
- Incremental: Convert COO?LIL (for efficient updates)?COO
- Note: scipy doesn't support direct COO in-place modifications

GPU Implementation:
- Native PyTorch COO sparse tensors
- Full recomputation: Concatenate new edges to COO indices/values and rebuild
- Incremental: Convert COO?dense (for in-place updates)?COO
- Note: PyTorch doesn't support in-place COO modifications


Usage

``bash
# CPU benchmarks
python dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-edges 3 --num-runs 3

# GPU benchmarks
python dynamic_graph_benchmark_gpu.py --vertices 10000 --sparsity 99.9 --num-edges 2 --num-runs 3
`

Arguments:
- --vertices: Number of graph vertices
- --sparsity: Sparsity percentage (90, 99, 99.9)
- --num-edges: New edges to add (1, 2, or 3)
- --num-runs: Number of runs for averaging

Results saved to: results/dynamic_graph_{cpu/gpu}_coo_{nodes}nodes_{sparsity}pct_{edges}edges_results.{json,txt,csv}`


Benchmark Results


Hardware Configuration
- CPU: AMD Ryzen 9 8940HX (16 cores)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (11.94 GB)
- CUDA: Version 13.0
- PyTorch: Version 2.9.1+cu130


5,000 Nodes @ 99% Sparsity (250,000 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 0.086478s | 0.027267s | 3.17× | Incremental |  | CPU | 2 | 0.088097s | 0.027930s | 3.15× | Incremental |  | CPU | 3 | 0.082690s | 0.029602s | 2.79× | Incremental |  | GPU | 1 | 0.016834s | 0.020033s | 0.84× | Full Recomp |  | GPU | 2 | 0.017753s | 0.020300s | 0.87× | Full Recomp |  | GPU | 3 | 0.017707s | 0.020735s | 0.85× | Full Recomp | 5,000 Nodes @ 99.9% Sparsity (24,999 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 0.008087s | 0.004529s | 1.79× | Incremental |  | CPU | 2 | 0.008647s | 0.004960s | 1.74× | Incremental |  | CPU | 3 | 0.010010s | 0.004834s | 2.07× | Incremental |  | GPU | 1 | 0.016586s | 0.019884s | 0.83× | Full Recomp |  | GPU | 2 | 0.016646s | 0.020935s | 0.80× | Full Recomp |  | GPU | 3 | 0.018322s | 0.020335s | 0.90× | Full Recomp | 10,000 Nodes @ 99% Sparsity (1,000,000 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 0.351879s | 0.103899s | 3.39× | Incremental |  | CPU | 2 | 0.339565s | 0.097629s | 3.48× | Incremental |  | CPU | 3 | 0.358276s | 0.110792s | 3.23× | Incremental |  | GPU | 1 | 0.045509s | 0.047524s | 0.96× | Full Recomp |  | GPU | 2 | 0.022370s | 0.028600s | 0.78× | Full Recomp |  | GPU | 3 | 0.042954s | 0.036743s | 1.17× | Incremental | 10,000 Nodes @ 99.9% Sparsity (99,999 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 0.038966s | 0.014054s | 2.77× | Incremental |  | CPU | 2 | 0.036606s | 0.016091s | 2.27× | Incremental |  | CPU | 3 | 0.036217s | 0.016579s | 2.18× | Incremental |  | GPU | 1 | 0.023222s | 0.040570s | 0.57× | Full Recomp |  | GPU | 2 | 0.017606s | 0.026087s | 0.67× | Full Recomp |  | GPU | 3 | 0.022529s | 0.036094s | 0.62× | Full Recomp | 20,000 Nodes @ 99% Sparsity (4,000,000 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 1.351105s | 0.346639s | 3.90× | Incremental |  | CPU | 2 | 1.406087s | 0.370062s | 3.80× | Incremental |  | CPU | 3 | 1.353696s | 0.343364s | 3.94× | Incremental |  | GPU | 1 | 0.040045s | 0.042814s | 0.94× | Full Recomp |  | GPU | 2 | 0.040850s | 0.044333s | 0.92× | Full Recomp |  | GPU | 3 | 0.040257s | 0.043014s | 0.94× | Full Recomp | 20,000 Nodes @ 99.9% Sparsity (399,999 edges) | Platform | Edges Added | Full Recomp | Incremental | Speedup | Winner |  | ---------- | ------------- | ------------- | ------------- | --------- | -------- |  | CPU | 1 | 0.131485s | 0.055658s | 2.36× | Incremental |  | CPU | 2 | 0.133921s | 0.056467s | 2.37× | Incremental |  | CPU | 3 | 0.134153s | 0.054852s | 2.45× | Incremental |  | GPU | 1 | 0.018330s | 0.040755s | 0.45× | Full Recomp |  | GPU | 2 | 0.017724s | 0.040471s | 0.44× | Full Recomp |  | GPU | 3 | 0.017854s | 0.040118s | 0.45× | Full Recomp | Analysis

CPU Performance:
- Incremental updates consistently outperform full recomputation across all configurations
- 5K nodes: 1.7-3.2× speedup
- 10K nodes: 2.2-3.5× speedup  
- 20K nodes: 2.4-3.9× speedup
- Speedup increases with graph size (larger graphs benefit more from incremental)

GPU Performance:
- Full recomputation dominates across nearly all configurations
- Only exception: 10K nodes, 99% sparsity, 3 edges (1.17× incremental)
- 20K nodes: Full recomp consistently faster (0.44-0.94×)
- COO?dense?COO conversion overhead exceeds small update benefits

Key Findings:
1. CPU: Incremental via LIL intermediate format highly effective (2-4× faster)
2. GPU: Dense conversion cost prohibitive for incremental COO updates
3. Crossover: Single GPU incremental win at 10K, 99%, 3 edges - not sustained at larger scale
4. Pattern: GPU incremental penalty worsens at extreme sparsity (99.9%: 0.44-0.45× vs 99%: 0.92-0.96×)

