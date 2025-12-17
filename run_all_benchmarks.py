"""
Automated Script to Run All Benchmark Commands from README.md
This script executes all commands and creates beautifully formatted log files
"""

import subprocess
import os
import sys
import platform
import shutil
from datetime import datetime
from pathlib import Path
import glob
import re

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_DIR = PROJECT_ROOT / "input"

OUTPUT_DIR.mkdir(exist_ok=True)


def get_python_command():
    """Detect the correct Python command for this platform"""
    if platform.system() == "Linux":
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return "python3"
    else:
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)
        return "python"


def clear_numba_cache():
    """Clear all numba cache to prevent cross-platform errors"""
    cache_dirs_found = 0
    cache_files_deleted = 0
    
    try:
        for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
            try:
                numba_files = list(pycache_dir.glob("*.nbc")) + list(pycache_dir.glob("*.nbi"))
                if numba_files:
                    cache_dirs_found += 1
                    for cache_file in numba_files:
                        try:
                            cache_file.unlink()
                            cache_files_deleted += 1
                        except Exception:
                            pass
            except (OSError, PermissionError):
                continue
    except Exception:
        pass
    
    try:
        for numba_cache_dir in PROJECT_ROOT.rglob(".numba_cache"):
            try:
                shutil.rmtree(numba_cache_dir)
                cache_dirs_found += 1
            except Exception:
                continue
    except Exception:
        pass
    
    try:
        home = Path.home()
        for cache_dir in [home / ".cache" / "numba", home / ".numba_cache"]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dirs_found += 1
    except Exception:
        pass
    
    os.environ['NUMBA_DISABLE_CACHING'] = '1'
    if cache_dirs_found > 0:
        print(f"Cleared {cache_files_deleted} numba cache files from {cache_dirs_found} locations")


def discover_datasets():
    """Discover available datasets in the input folder"""
    datasets = {
        'dense_pairs': [],
        'sparse_pairs': []
    }
    
    dense_a_files = glob.glob(str(INPUT_DIR / "dense_matrix_a_*.csv"))
    for a_file in dense_a_files:
        match = re.search(r'dense_matrix_a_(.+)\.csv', os.path.basename(a_file))
        if match:
            dim_pattern = match.group(1)
            b_file = INPUT_DIR / f"dense_matrix_b_{dim_pattern}.csv"
            if b_file.exists():
                datasets['dense_pairs'].append({
                    'dim': dim_pattern,
                    'file_a': os.path.basename(a_file),
                    'file_b': os.path.basename(b_file)
                })
    
    sparse_a_files = glob.glob(str(INPUT_DIR / "matrix_a*.csv"))
    sparse_a_files = [f for f in sparse_a_files if not f.endswith('_transposed.csv') and 'dense' not in f]
    
    for a_file in sparse_a_files:
        a_basename = os.path.basename(a_file)
        match = re.search(r'matrix_a(.*)\.csv', a_basename)
        if match:
            suffix = match.group(1)
            b_file = INPUT_DIR / f"matrix_b{suffix}.csv"
            if b_file.exists():
                dim_name = suffix.replace('_', '') if suffix else 'default'
                datasets['sparse_pairs'].append({
                    'dim': dim_name,
                    'file_a': a_basename,
                    'file_b': os.path.basename(b_file)
                })
    
    return datasets

def create_log_header(command, description):
    """Create a beautifully formatted log header"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""================================================================================
                           COMMAND EXECUTION LOG
================================================================================

COMMAND EXECUTED:
{command}

DESCRIPTION:
{description}

DATE & TIME: {timestamp}

================================================================================
                                  OUTPUT START
================================================================================

"""

def create_log_footer(success=True, error_msg=""):
    """Create a beautifully formatted log footer"""
    status = "SUCCESS" if success else "FAILED"
    error_section = f"\nERROR MESSAGE:\n{error_msg}\n" if error_msg else ""
    
    return f"""
================================================================================
                                   OUTPUT END
================================================================================

EXECUTION STATUS: {status}{error_section}
================================================================================
                              END OF LOG FILE
================================================================================
"""

def run_command(command, log_filename, description, cwd=None):
    """Run a command and save output to a formatted log file"""
    print("=" * 80)
    print(f"Running: {command}")
    print("=" * 80)
    
    log_path = OUTPUT_DIR / log_filename
    
    try:
        if cwd is None:
            cwd = PROJECT_ROOT
        
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['NO_COLOR'] = '1'
        env['TERM'] = 'dumb'
        env['TQDM_DISABLE'] = '1'
        
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        
        import re
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z/-_]|\[[0-?]*[ -/]*[@-~])')
        output = ansi_escape.sub('', output)
        
        output = output.replace('\r', '')
        
        log_content = create_log_header(command, description)
        log_content += output
        log_content += create_log_footer(success=(result.returncode == 0), 
                                         error_msg="" if result.returncode == 0 else f"Command exited with code {result.returncode}")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        print(f"[OK] Log saved to: {log_filename}")
        print()
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        log_content = create_log_header(command, description)
        log_content += "Command timed out after 10 minutes"
        log_content += create_log_footer(success=False, error_msg="Timeout after 10 minutes")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        print(f"[ERROR] Command timed out - Log saved to: {log_filename}")
        print()
        return False
        
    except Exception as e:
        log_content = create_log_header(command, description)
        log_content += f"Exception occurred: {str(e)}"
        log_content += create_log_footer(success=False, error_msg=str(e))
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        print(f"[ERROR] Error occurred - Log saved to: {log_filename}")
        print()
        return False



def main():
    """Main execution function"""
    clear_numba_cache()
    
    python_cmd = get_python_command()
    print(f"Using Python command: {python_cmd}\n")
    
    print("================================================================================")
    print("           STARTING EXECUTION OF ALL README.MD BENCHMARK COMMANDS               ")
    print("================================================================================")
    print()
    
    print("Discovering datasets in input/ folder...")
    datasets = discover_datasets()
    
    print(f"  Found {len(datasets['dense_pairs'])} dense matrix pairs:")
    for pair in datasets['dense_pairs']:
        print(f"    - {pair['dim']}: {pair['file_a']} + {pair['file_b']}")
    
    print(f"  Found {len(datasets['sparse_pairs'])} sparse matrix pairs:")
    for pair in datasets['sparse_pairs']:
        print(f"    - {pair['dim']}: {pair['file_a']} + {pair['file_b']}")
    print()
    
    if not datasets['dense_pairs'] and not datasets['sparse_pairs']:
        print("âš  WARNING: No dataset pairs found! Please check the input/ folder.")
        print("  Expected format:")
        print("    Dense: dense_matrix_a_XXX.csv + dense_matrix_b_XXX.csv")
        print("    Sparse: matrix_a_XXX.csv + matrix_b_XXX.csv")
        return
    
    dynamic_commands = []
    cmd_num = 2
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} CSR_CSC_Implementation/dense_sparse_cpu_benchmarks/single_threaded/multiplication_benchmark.py --size 1000 --sparsity 99.9 --num-runs 3",
            "log": f"{cmd_num:02d}_csr_csc_single_multiplication.txt",
            "desc": "CSR/CSC single-threaded sparse matrix multiplication (1000x1000, 99.9% sparsity)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} CSR_CSC_Implementation/dense_sparse_cpu_benchmarks/single_threaded/addition_benchmark.py --size 2000 --sparsity 99.9 --num-runs 3",
            "log": f"{cmd_num:02d}_csr_csc_single_addition.txt",
            "desc": "CSR/CSC single-threaded sparse matrix addition (2000x2000, 99.9% sparsity)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} CSR_CSC_Implementation/gnn_sparse_benchmarks/static_benchmarks/static_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-runs 3",
            "log": f"{cmd_num:02d}_csr_csc_gnn_static.txt",
            "desc": "CSR/CSC GNN static graph benchmark (10000 vertices, 99.9% sparsity)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} CSR_CSC_Implementation/gnn_sparse_benchmarks/dynamic_benchmarks/dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-edges 1 --num-runs 3",
            "log": f"{cmd_num:02d}_csr_csc_gnn_dynamic_1edge.txt",
            "desc": "CSR/CSC GNN dynamic graph benchmark (10000 vertices, 99.9% sparsity, 1 edge update)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} CSR_CSC_Implementation/gnn_sparse_benchmarks/dynamic_benchmarks/dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99 --num-edges 3 --num-runs 3",
            "log": f"{cmd_num:02d}_csr_csc_gnn_dynamic_3edges.txt",
            "desc": "CSR/CSC GNN dynamic graph benchmark (10000 vertices, 99% sparsity, 3 edge updates)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} COO_Implementation/dense_sparse_cpu_benchmarks/single_threaded/addition_benchmark.py --num-runs 3",
            "log": f"{cmd_num:02d}_coo_single_addition.txt",
            "desc": "COO single-threaded sparse matrix addition (all sparsity levels)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} COO_Implementation/dense_sparse_cpu_benchmarks/single_threaded/multiplication_benchmark.py --num-runs 3",
            "log": f"{cmd_num:02d}_coo_single_multiplication.txt",
            "desc": "COO single-threaded sparse matrix multiplication (all sparsity levels)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} COO_Implementation/dense_sparse_cpu_benchmarks/multicore_parallel/multiplication_multiprocess_benchmark.py --size 2000 --sparsity 99.9 --num-runs 1",
            "log": f"{cmd_num:02d}_coo_multiprocess_multiplication.txt",
            "desc": "COO multiprocess sparse matrix multiplication (2000x2000, 99.9% sparsity, 16 cores)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} COO_Implementation/gnn_sparse_benchmarks/static_benchmarks/static_graph_benchmark_cpu.py --vertices 10000 --sparsity 99 --num-runs 3",
            "log": f"{cmd_num:02d}_coo_gnn_static.txt",
            "desc": "COO GNN static graph benchmark (10000 vertices, 99% sparsity)"
        }
    ])
    cmd_num += 1
    
    dynamic_commands.extend([
        {
            "cmd": f"{python_cmd} COO_Implementation/gnn_sparse_benchmarks/dynamic_benchmarks/dynamic_graph_benchmark_cpu.py --vertices 10000 --sparsity 99.9 --num-edges 3 --num-runs 3",
            "log": f"{cmd_num:02d}_coo_gnn_dynamic.txt",
            "desc": "COO GNN dynamic graph benchmark (10000 vertices, 99.9% sparsity, 3 edge updates)"
        }
    ])
    cmd_num += 1
    
    for pair in datasets['sparse_pairs']:
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/addition_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_csr_csc_db_addition_{pair['dim']}.txt",
            "desc": f"CSR/CSC database sparse addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/multiplication_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_csr_csc_db_multiplication_{pair['dim']}.txt",
            "desc": f"CSR/CSC database sparse multiplication ({pair['dim']})"
        })
        cmd_num += 1
    
    for pair in datasets['dense_pairs']:
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/addition_dense_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_csr_csc_db_dense_addition_{pair['dim']}.txt",
            "desc": f"CSR/CSC database dense addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/multiplication_dense_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_csr_csc_db_dense_multiplication_{pair['dim']}.txt",
            "desc": f"CSR/CSC database dense multiplication ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/multicore_parallel/addition_dense_parallel_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_csr_csc_db_parallel_addition_{pair['dim']}.txt",
            "desc": f"CSR/CSC database parallel dense addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} CSR_CSC_Database_Approach/dense_sparse_cpu_benchmarks/multicore_parallel/multiplication_dense_parallel_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_csr_csc_db_parallel_multiplication_{pair['dim']}.txt",
            "desc": f"CSR/CSC database parallel dense multiplication ({pair['dim']})"
        })
        cmd_num += 1
    
    for pair in datasets['sparse_pairs']:
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/addition_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_coo_db_addition_{pair['dim']}.txt",
            "desc": f"COO database sparse addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/multiplication_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']}",
            "log": f"{cmd_num:02d}_coo_db_multiplication_{pair['dim']}.txt",
            "desc": f"COO database sparse multiplication ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/multicore_parallel/addition_parallel_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_coo_db_parallel_addition_{pair['dim']}.txt",
            "desc": f"COO database parallel sparse addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/multicore_parallel/multiplication_parallel_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_coo_db_parallel_multiplication_{pair['dim']}.txt",
            "desc": f"COO database parallel sparse multiplication ({pair['dim']})"
        })
        cmd_num += 1
    
    for pair in datasets['dense_pairs']:
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/addition_dense_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_coo_db_dense_addition_{pair['dim']}.txt",
            "desc": f"COO database dense addition ({pair['dim']})"
        })
        cmd_num += 1
        
        dynamic_commands.append({
            "cmd": f"{python_cmd} COO_Database_Approach/dense_sparse_cpu_benchmarks/single_threaded/multiplication_dense_benchmark.py --input_a input/{pair['file_a']} --input_b input/{pair['file_b']} --num-runs 1",
            "log": f"{cmd_num:02d}_coo_db_dense_multiplication_{pair['dim']}.txt",
            "desc": f"COO database dense multiplication ({pair['dim']})"
        })
        cmd_num += 1
    
    total_commands = len(dynamic_commands)
    successful = 0
    failed = 0
    
    print(f"Total commands to execute: {total_commands}")
    print()
    
    for idx, cmd_info in enumerate(dynamic_commands, start=2):
        print(f"\n[Command {idx}/{total_commands + 1}]")
        success = run_command(cmd_info["cmd"], cmd_info["log"], cmd_info["desc"])
        if success:
            successful += 1
        else:
            failed += 1
    
    print()
    print("================================================================================")
    print("                    EXECUTION COMPLETE - SUMMARY                                ")
    print("================================================================================")
    print(f"  Total Commands: {total_commands + 1} (including pip install)")
    print(f"  Successful: {successful + 1}")
    print(f"  Failed: {failed}")
    print(f"  Output Location: ./output/")
    print(f"  Dense Dataset Pairs: {len(datasets['dense_pairs'])}")
    print(f"  Sparse Dataset Pairs: {len(datasets['sparse_pairs'])}")
    print("================================================================================")
    print()
    print("All log files have been saved with beautifully formatted output!")
    print()
    print("Each log file contains:")
    print("  1. The exact command executed")
    print("  2. Clear visual separators (double dash lines)")
    print("  3. Complete command output")
    print("  4. Professional formatting")
    print()
    print("Check the output folder to review all results!")

if __name__ == "__main__":
    main()


