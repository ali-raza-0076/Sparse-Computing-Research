# Sparse Matrix Operations Benchmark Suite

A comprehensive benchmarking framework for sparse matrix operations (multiplication and addition) using CSR, CSC, and COO formats. Includes implementations for CPU (single-threaded and multicore), GPU acceleration, and GNN-specific benchmarks.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation Instructions](#installation-instructions)
4. [Running Benchmarks](#running-benchmarks)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### Windows

```powershell
# Install dependencies
python install_requirements.py

# Run all benchmarks (30-60 minutes)
python run_all_benchmarks.py
```

### Linux/WSL

```bash
# Install dependencies
python3 install_requirements.py

# Activate virtual environment (automatically created on Linux)
source venv/bin/activate

# Run all benchmarks
python3 run_all_benchmarks.py
```

### First-Time Setup

The installer automatically:
- Detects your operating system
- Creates virtual environment on Linux/WSL (handles externally-managed environment)
- Installs all required packages (NumPy, SciPy, Numba, PyTorch, etc.)
- Clears Numba cache to prevent issues

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher (Python 3.11+ recommended)
- **RAM**: 8GB (16GB+ recommended for large matrices)
- **Disk Space**: 5GB free

### Optional for GPU Benchmarks

- **GPU**: NVIDIA GPU with CUDA support
- **CUDA Toolkit**: 11.0+
- **VRAM**: 4GB+ recommended

### Cross-Platform Support

- Windows 10/11
- Linux (Ubuntu 20.04+)
- WSL 2 (Windows Subsystem for Linux)

## Installation Instructions

The `install_requirements.py` script handles all dependencies automatically.

### On Windows

1. Open PowerShell or Command Prompt
2. Navigate to project directory:
   ```powershell
   cd \path\to\DB_Project_MatMul
   ```
3. Run installer:
   ```powershell
   python install_requirements.py
   ```
4. Wait for installation to complete

### On Linux/WSL

1. Open terminal
2. Navigate to project directory:
   ```bash
   cd /path/to/DB_Project_MatMul
   ```
3. Run installer:
   ```bash
   python3 install_requirements.py
   ```
4. The script will:
   - Detect externally-managed Python environment
   - Automatically create `venv/` directory
   - Install all packages in the virtual environment
5. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
6. You're ready to run benchmarks!

### What Gets Installed

- **numpy** - Array operations
- **scipy** - Sparse matrix formats
- **numba** - JIT compilation for performance
- **torch** - GPU acceleration (PyTorch)
- **pandas** - Data handling
- **matplotlib** - Visualization (optional)

### Troubleshooting Installation

- **If venv creation fails on Linux**: `sudo apt install python3-venv`
- **If CUDA errors occur**: GPU benchmarks will be skipped automatically
- **If Numba cache errors occur**: Automatically cleared on script start

## Running Benchmarks

### Option 1: Run Everything (Recommended for First Time)

**Windows:**
```powershell
python run_all_benchmarks.py
```

**Linux/WSL:**
```bash
source venv/bin/activate
python3 run_all_benchmarks.py
```

This runs all benchmarks and creates detailed logs in the `output/` directory.  
**Total runtime**: 30-60 minutes depending on hardware.

## Troubleshooting

### Problem: "externally-managed-environment" error on Linux/WSL

**Solution**: Run `install_requirements.py` - it automatically creates a venv.  
Then activate it:
```bash
source venv/bin/activate
```

### Problem: ModuleNotFoundError for sparse_addition_coo or similar

**Solution**: Numba cache corruption. The scripts now automatically clear the cache at startup. If issues persist, manually delete `~/.numba_cache` and all `__pycache__` directories.

### Problem: "No module named 'torch'" or "No module named 'numba'"

**Solution**: Rerun `install_requirements.py`.  
On Linux/WSL: `source venv/bin/activate` first.

### Problem: CUDA out of memory errors

**Solution**: GPU benchmarks automatically skip if CUDA unavailable. Reduce matrix size with `--size` or `--vertices` arguments.

### Problem: Permission denied when writing results

**Solution**: Ensure you have write permissions in the project directory.  
On Linux: check with `ls -la`

### Problem: Scripts take too long to run

**Solution**: 
- Reduce `--num-runs` to 1 for faster testing
- Use smaller matrices: `--vertices 1000 --sparsity 99`

### Problem: Different results on Windows vs Linux

**Solution**: This is normal due to floating-point precision differences. Results should be within 1e-6 tolerance.

**License**: Apache License 2.0
