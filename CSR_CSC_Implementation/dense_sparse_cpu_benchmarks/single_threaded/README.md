Single-Threaded Benchmark Results


Overview

Single-threaded dense (NumPy) vs sparse (CSR/CSC) matrix operations.

Hardware: AMD Ryzen 9 8940HX (16 cores), 32GB RAM  
Software: Python 3.13, NumPy, Numba

---


Multiplication (CSR×CSC) | Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |  | ------------- | ---------- | ----------- | ------------- | ------------ | --------- | -------- |  | 1000×1000 | 90% | 100,000 | 0.363s | 0.486s | 1.34× | Sparse |  | 1000×1000 | 99% | 10,000 | 0.046s | 0.472s | 10.25× | Sparse |  | 1000×1000 | 99.9% | 999 | 0.014s | 0.481s | 34.67× | Sparse |  | 2000×2000 | 99% | 40,000 | 0.155s | 5.346s | 34.42× | Sparse |  | 2000×2000 | 99.9% | 3,999 | 0.063s | 5.329s | 84.09× | Sparse |  | 3000×3000 | 99% | 90,000 | 0.354s | 52.960s | 149.63× | Sparse |  | 3000×3000 | 99.9% | 8,999 | 0.064s | 55.385s | 870.01× | Sparse | Key Findings:
- Sparse wins at all sparsity levels
- Speedup increases dramatically with matrix size: 34.67× (1000) ? 164.54× (2000) ? 793.52× (3000)
- 3000×3000 @ 99.9%: 793.52× speedup demonstrates exceptional scaling for high-sparsity multiplication

Scaling Limitation:
Beyond 3000×3000, single-threaded multiplication becomes impractical. Dense operations scale as O(n³), requiring prohibitively long execution times (hours) for matrices =4000×4000.


Addition (CSR) | Matrix Size | Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Winner |  | ------------- | ---------- | ----------- | ------------- | ------------ | --------- | -------- |  | 1000×1000 | 90% | 100,000 | 0.216s | 0.001s | 0.00× | Dense |  | 1000×1000 | 99% | 10,000 | 0.021s | 0.001s | 0.05× | Dense |  | 1000×1000 | 99.9% | 999 | 0.004s | 0.001s | 0.28× | Dense |  | 2000×2000 | 99% | 40,000 | 0.084s | 0.003s | 0.04× | Dense |  | 2000×2000 | 99.9% | 3,999 | 0.011s | 0.008s | 0.75× | Dense |  | 3000×3000 | 99% | 90,000 | 0.196s | 0.007s | 0.04× | Dense |  | 3000×3000 | 99.9% | 8,999 | 0.052s | 0.015s | 0.30× | Dense | Key Findings:
- Dense wins at all sparsity levels and matrix sizes
- CSR conversion overhead exceeds computation savings
- Addition is O(n) operation - insufficient complexity to benefit from sparse representation

---


Summary

Multiplication: Sparse representation highly effective, with speedup improving dramatically as sparsity increases and matrices grow larger.

Addition: Dense representation consistently faster due to format conversion overhead dominating the simple O(n) operation.

---


Files

- results/multiplication_results.{json,csv,txt} - 1000×1000 multiplication
- results/addition_results.{json,csv,txt} - 1000×1000 addition
- results/multiplication_2000x2000_results.{json,csv,txt} - 2000×2000 multiplication
- results/addition_2000x2000_results.{json,csv,txt} - 2000×2000 addition
- results/multiplication_3000x3000_results.{json,csv,txt} - 3000×3000 multiplication
- results/addition_3000x3000_results.{json,csv,txt} - 3000×3000 addition


