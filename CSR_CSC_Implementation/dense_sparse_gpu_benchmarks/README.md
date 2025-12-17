================================================================================
    GPU SPARSE MATRIX BENCHMARKS (CSR/CSC)
================================================================================


PURPOSE
-----------------

Compare GPU (PyTorch/CUDA) sparse vs dense matrix operations using CSR and CSC
formats to evaluate GPU acceleration benefits for sparse computations.


SETUP
-----------------

Requirements:
  * NVIDIA GPU with CUDA support
  * PyTorch with CUDA enabled
  * Python 3.8+

Installation:
  pip install torch scipy numpy tabulate

Technical Note - Matrix Multiplication Order:
  When using non-square test matrices (e.g., 50,001 x 50,000), multiplication
  may use B * A instead of A * B to ensure dimension compatibility. For A * B to
  be valid, A.cols must equal B.rows. If this condition is not met, B * A is
  computed instead (where B.cols must equal A.rows). This ensures benchmarks run
  successfully with the available test data.


Execution

Run benchmarks with custom matrix size and sparsity:


Matrix Multiplication (CSR * CSC)
`bash
python multiplication_gpu_benchmark.py --size 1000 --sparsity 99 --num-runs 3
python multiplication_gpu_benchmark.py --size 2000 --sparsity 99.9 --num-runs 3
python multiplication_gpu_benchmark.py --size 3000 --sparsity 90 --num-runs 3
`


Matrix Addition (CSR + CSR)
`bash
python addition_gpu_benchmark.py --size 1000 --sparsity 99 --num-runs 3
python addition_gpu_benchmark.py --size 2000 --sparsity 99.9 --num-runs 3
python addition_gpu_benchmark.py --size 3000 --sparsity 90 --num-runs 3
`


Arguments
- --size: Matrix dimension (NxN)
- --sparsity: Sparsity percentage (e.g., 90, 99, 99.9)
- --num-runs: Number of runs for averaging (default: 3)
  - Why 3 runs? GPU performance varies due to thermal throttling, background processes, and GPU scheduler decisions
  - Running 3 times and averaging produces stable, reliable results and eliminates outliers
  - Each benchmark: 1 warmup run (not measured) + 3 timed runs (averaged)
  - Use --num-runs 1 for quick tests, 3-5 for production-quality benchmarks


Benchmark Methodology

Warmup Phase (CRITICAL for GPU accuracy):
- Every benchmark performs 1 warmup run before any timing begins
- Warmup run executes the full operation but timing is discarded
- Purpose:
  - Compiles CUDA kernels (JIT compilation on first execution)
  - Initializes GPU device state and memory allocators
  - Warms L1/L2 caches on GPU
  - Eliminates "cold start" penalties
- Only post-warmup runs are measured and averaged
- Without warmup: First run 10-100x slower due to kernel compilation

Timing Protocol:
- torch.cuda.synchronize() called before starting timer
- Operation executes on GPU (asynchronous by default)
- torch.cuda.synchronize() called after operation to block CPU until GPU completes
- time.perf_counter() captures high-resolution timing between synchronization points
- Prevents overlap between operations and ensures accurate GPU execution time

Data Generation:
- Matrices generated internally with specified size and sparsity
- Random seed fixed for reproducibility (42 for A, 123 for B)
- Values randomly selected from [1, 10] range


Test Configuration

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
  - CUDA Cores: 5,888
  - VRAM: 12 GB GDDR7
  - CUDA Version: 13.0
- Framework: PyTorch 2.9.1+cu130
- Sparse Format: 
  - Input: CSR * CSC for multiplication, CSR + CSR for addition
  - GPU Execution: COO (Coordinate format)
  - Important: PyTorch does not natively support CSR/CSC on GPU. Matrices are generated in CSR/CSC using scipy for optimal structure, then automatically converted to COO format via .tocoo() before GPU transfer. This is a PyTorch limitation - while NVIDIA's cuSPARSE library supports CSR on GPU, PyTorch only exposes COO sparse tensors for GPU operations.
- Parallelization: PyTorch automatically distributes operations across all CUDA cores using thread blocks and warps for parallel processing of sparse matrix elements
- Metrics: Execution time, speedup, memory usage


Results

Results are saved in the results/ directory with the following naming pattern:
- {operation}_gpu_{size}x{size}_{sparsity}pct_results.{json,txt,csv}


Example Filenames
- multiplication_gpu_1000x1000_99pct_results.json
- addition_gpu_2000x2000_99_9pct_results.txt
- multiplication_gpu_3000x3000_90pct_results.csv


Benchmark Results


Matrix Multiplication: GPU Sparse (CSR * CSC) vs Dense

| Size         | Sparsity | Non-Zeros | Sparse Time (s) | Dense Time (s) | Speedup | Winner | Memory Ratio |
| ------------ | -------- | --------- | --------------- | -------------- | ------- | ------ | ------------ |
| 1000x1000    | 90%      | 95,178    | 0.001874        | 0.000308       | 0.16x   | Dense  | 5.23x        |
| 1000x1000    | 99%      | 9,954     | 0.000788        | 0.000304       | 0.39x   | Dense  | 47.83x       |
| 1000x1000    | 99.9%    | 999       | 0.000544        | 0.000320       | 0.59x   | Dense  | 333.44x      |
| 2000x2000    | 90%      | 380,453   | 0.008030        | 0.001221       | 0.15x   | Dense  | 5.24x        |
| 2000x2000    | 99%      | 39,802    | 0.001538        | 0.001268       | 0.82x   | Dense  | 49.02x       |
| 2000x2000    | 99.9%    | 3,994     | 0.000855        | 0.001085       | 1.27x   | Sparse | 400.44x      |
| 3000x3000    | 90%      | 856,523   | 0.021328        | 0.003373       | 0.16x   | Dense  | 5.24x        |
| 3000x3000    | 99%      | 89,558    | 0.003838        | 0.003641       | 0.95x   | Dense  | 49.42x       |
| 3000x3000    | 99.9%    | 8,993     | 0.001977        | 0.003646       | 1.84x   | Sparse | 428.84x      |
| 10000x10000  | 90%      | 9,516,182 | 0.919717        | 0.138242       | 0.15x   | Dense  | 5.25x        |
| 10000x10000  | 99%      | 995,142   | 0.100388        | 0.141636       | 1.41x   | Sparse | 49.99x       |
| 10000x10000  | 99.9%    | 99,945    | 0.021959        | 0.126369       | 5.75x   | Sparse | 476.44x      |

Analysis: At 1000x1000, dense wins all configurations due to GPU sparse overhead. At larger sizes, sparse begins winning at higher sparsity: 2000x2000+ @ 99.9% shows 1.27-5.75x speedup. Best sparse performance: 5.75x at 10000x10000 @ 99.9%. Low sparsity (90%) continues favoring dense even at 10000x10000 (0.15x). Multiplication shows weaker GPU sparse benefits than addition. Sparse formats provide 5-476x memory savings.

---


Matrix Addition: GPU Sparse (CSR + CSR) vs Dense

| Size         | Sparsity | Non-Zeros | Sparse Time (s) | Dense Time (s) | Speedup | Winner | Memory Ratio |
| ------------ | -------- | --------- | --------------- | -------------- | ------- | ------ | ------------ |
| 1000x1000    | 90%      | 95,178    | 0.000130        | 0.000040       | 0.31x   | Dense  | 5.23x        |
| 1000x1000    | 99%      | 9,954     | 0.000067        | 0.000039       | 0.59x   | Dense  | 47.83x       |
| 1000x1000    | 99.9%    | 999       | 0.000043        | 0.000036       | 0.84x   | Dense  | 333.44x      |
| 2000x2000    | 90%      | 380,453   | 0.000277        | 0.000240       | 0.87x   | Dense  | 5.24x        |
| 2000x2000    | 99%      | 39,802    | 0.000254        | 0.000311       | 1.23x   | Sparse | 49.02x       |
| 2000x2000    | 99.9%    | 3,994     | 0.000043        | 0.000320       | 7.44x   | Sparse | 400.44x      |
| 3000x3000    | 90%      | 856,523   | 0.000459        | 0.000558       | 1.22x   | Sparse | 5.24x        |
| 3000x3000    | 99%      | 89,558    | 0.000206        | 0.000450       | 2.19x   | Sparse | 49.42x       |
| 3000x3000    | 99.9%    | 8,993     | 0.000044        | 0.000432       | 9.79x   | Sparse | 428.84x      |
| 10000x10000  | 90%      | 9,516,182 | 0.003843        | 0.005567       | 1.45x   | Sparse | 5.25x        |
| 10000x10000  | 99%      | 995,142   | 0.000554        | 0.005730       | 10.34x  | Sparse | 49.99x       |
| 10000x10000  | 99.9%    | 99,945    | 0.000129        | 0.005098       | 39.61x  | Sparse | 476.44x      |

Analysis: At 1000x1000, dense wins all configurations. At 2000x2000 and larger, sparse wins decisively with 1.22-39.61x speedup at 90% and higher sparsity. Addition shows stronger GPU sparse performance than multiplication across all matrix sizes. Best sparse performance: 39.61x at 10000x10000 @ 99.9%. Sparse formats provide 5-476x memory savings. Performance scales dramatically with matrix size at high sparsity.

---


Output Format

Each benchmark run produces three files:

1. JSON (_results.json): Machine-readable results with all metrics
2. TXT (_results.txt): Human-readable summary with formatted output
3. CSV (*_results.csv): Spreadsheet-compatible format for analysis


Metrics Captured
- Matrix size and actual sparsity
- Actual number of non-zeros
- GPU sparse operation time
- GPU dense operation time
- Speedup ratio (Dense/Sparse)
- Winner (Sparse or Dense)
- Memory usage (sparse vs dense)
- GPU name and CUDA version


Notes

- Benchmarks use PyTorch's sparse tensor operations
- CSR format is converted to COO internally by PyTorch for GPU operations
- Results may vary based on GPU architecture and CUDA version
- Warmup runs are performed before timing to ensure accurate measurements
- Multiple runs are averaged to reduce variance


Matrix Size Limitation

Maximum tested size: 10,000 x 10,000

Larger matrices (50,000 x 50,000 and above) were attempted but failed due to RAM limitations, not GPU memory constraints:

- Problem: Dense matrix creation (.toarray()) occurs in CPU RAM before GPU transfer
- Memory requirement: 50,000 x 50,000 dense matrix = 10 GB RAM (float32)
- System limit: Exceeded available system RAM during dense matrix generation
- Impact: Benchmark crashed during .toarray() conversion before reaching GPU operations

The bottleneck is CPU RAM for dense matrix creation, not GPU VRAM. Sparse matrices at these sizes fit easily in memory (99.9% sparsity = ~200 MB), but the corresponding dense comparison matrix cannot be created. This demonstrates a key advantage of sparse formats: they enable working with matrix dimensions that would be impossible in dense format due to memory constraints.


Comparison with CPU Benchmarks

For direct comparison with CPU performance, see:
- ../dense_sparse_cpu_benchmarks/single_threaded/README.md - Single-threaded CPU results
- ../dense_sparse_cpu_benchmarks/BENCHMARK_RESULTS.md` - Complete CPU benchmark summary



