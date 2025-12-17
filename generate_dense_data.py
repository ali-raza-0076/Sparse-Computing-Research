"""
Generate 2000×2000 dense matrices for database I/O benchmarking.

Creates fully dense matrices (all 4 million entries filled) in COO format (i,j,v).
Saved as CSV for database I/O workflow testing.
"""
import numpy as np
import csv
import os

def generate_dense_matrix_coo(rows, cols, output_file, seed=42):
    """
    Generate a dense matrix in COO format (every position has a value).
    
    Args:
        rows: Number of rows
        cols: Number of columns
        output_file: Path to save CSV file
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print(f"Generating {rows}×{cols} dense matrix in COO format...")
    print(f"Total entries: {rows * cols:,}")
    
    values = np.random.randint(-100, 100, size=(rows, cols))
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        entries = 0
        for i in range(rows):
            for j in range(cols):
                writer.writerow([i, j, int(values[i, j])])
                entries += 1
            if (i + 1) % 200 == 0:
                print(f"  Written {entries:,} / {rows * cols:,} entries ({(i+1)/rows*100:.1f}%)")
    
    print(f"✓ Generated {entries:,} entries")
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ File size: {file_size_mb:.1f} MB")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'input')
    os.makedirs(output_dir, exist_ok=True)
    
    size = 300
    
    print("="*70)
    print("DENSE MATRIX GENERATION FOR DATABASE I/O BENCHMARKS")
    print("="*70)
    print(f"\nMatrix size: {size}×{size}")
    print(f"Format: Dense (all {size*size:,} entries filled)")
    print(f"Storage: COO format (i,j,v) in CSV")
    print()
    
    matrix_a_file = os.path.join(output_dir, f'dense_matrix_a_{size}x{size}.csv')
    generate_dense_matrix_coo(size, size, matrix_a_file, seed=42)
    
    print()
    
    matrix_b_file = os.path.join(output_dir, f'dense_matrix_b_{size}x{size}.csv')
    generate_dense_matrix_coo(size, size, matrix_b_file, seed=123)
    
    print()
    print("="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {matrix_a_file}")
    print(f"  2. {matrix_b_file}")
    print(f"\nThese matrices can be used for:")
    print(f"  - COO format: Direct usage (already in i,j,v format)")
    print(f"  - CSR/CSC format: Convert using matrix_formats.py")
    print(f"  - Database I/O benchmarks: Addition and multiplication")
    print()


if __name__ == '__main__':
    main()
