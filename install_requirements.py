#!/usr/bin/env python3
"""
Python Requirements Installer for Sparse Matrix Operations Project
Simply run: python install_requirements.py
"""

import sys
import subprocess
import os
import platform
import venv

def print_header():
    """Print a nice header"""
    print()
    print("=" * 80)
    print("          Python Requirements Installation")
    print("          Sparse Matrix Operations Project")
    print("=" * 80)
    print()

def print_success():
    """Print success message"""
    print()
    print("=" * 80)
    print("          ✓ Installation Completed Successfully!")
    print("=" * 80)
    print()
    print("All required packages have been installed.")
    
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    python_cmd = "python" if platform.system() == "Windows" else "python3"
    
    if os.path.exists(venv_dir):
        print()
        print("IMPORTANT: Virtual environment created at ./venv")
        print()
        if platform.system() == "Windows":
            print("To activate the virtual environment:")
            print("  .\\venv\\Scripts\\activate")
            print()
            print("Then run benchmarks:")
            print("  python run_all_benchmarks.py")
        else:
            print("To activate the virtual environment:")
            print("  source venv/bin/activate")
            print()
            print("Then run benchmarks:")
            print("  python3 run_all_benchmarks.py")
        print()
    else:
        print("You can now run the benchmark scripts.")
        print()
        print("Quick start:")
        print(f"  • Run all CPU benchmarks:    {python_cmd} run_all_benchmarks.py")
        print(f"  • Generate test data:        {python_cmd} generate_dense_data.py")
        print()

def print_failure():
    """Print failure message"""
    print()
    print("=" * 80)
    print("          ✗ Installation Failed!")
    print("=" * 80)
    print()
    print("Some packages failed to install.")
    print("Please check the error messages above for details.")
    print()
    python_cmd = "python" if platform.system() == "Windows" else "python3"
    print("Common solutions:")
    print("  • Make sure you have internet connection")
    print("  • On Linux/WSL, install python3-venv: sudo apt install python3-venv")
    print("  • Update Python to the latest version")
    print(f"  • Try: {python_cmd} -m pip install --upgrade pip")
    print()

def check_python_version():
    """Check if Python version is sufficient"""
    print("[1/5] Checking Python version...")
    version = sys.version_info
    print(f"      ✓ Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"      ✗ ERROR: Python 3.8 or higher is required!")
        print(f"      Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    print()
    print("[2/5] Upgrading pip to latest version...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("      ✓ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print("      ⚠ Warning: Could not upgrade pip (continuing anyway)")
        return True
    except Exception as e:
        print(f"      ⚠ Warning: {str(e)} (continuing anyway)")
        return True

def check_requirements_file():
    """Check if requirements.txt exists"""
    print()
    print("[3/5] Checking for requirements.txt...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")
    
    if not os.path.exists(requirements_path):
        print(f"      ✗ ERROR: requirements.txt not found in {script_dir}")
        return None
    
    print(f"      ✓ Found: {requirements_path}")
    return requirements_path

def check_and_create_venv():
    """Check if virtual environment is needed and create it"""
    print()
    print("[4/5] Checking installation environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("      ✓ Already in a virtual environment")
        return sys.executable, False
    
    is_linux = platform.system() == "Linux"
    
    if is_linux:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "numpy"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "externally-managed-environment" in result.stderr:
                print("      ⚠ Externally-managed Python environment detected")
                print("      → Creating virtual environment...")
                needs_venv = True
            else:
                print("      ✓ System Python can install packages directly")
                return sys.executable, False
        except:
            needs_venv = True
    else:
        print("      ✓ Windows environment - no virtual environment needed")
        return sys.executable, False
    
    if needs_venv:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        venv_dir = os.path.join(script_dir, "venv")
        
        if platform.system() == "Windows":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")
        
        if os.path.exists(venv_dir):
            if os.path.exists(venv_python):
                print(f"      ✓ Virtual environment already exists: {venv_dir}")
                print("      → Using existing virtual environment")
            else:
                print(f"      ⚠ Corrupted virtual environment detected")
                print(f"      → Deleting corrupted venv...")
                import shutil
                try:
                    shutil.rmtree(venv_dir)
                except Exception as e:
                    print(f"      ✗ ERROR: Could not delete corrupted venv: {e}")
                    return None, False
                print(f"      → Creating new virtual environment: {venv_dir}")
                try:
                    venv.create(venv_dir, with_pip=True)
                    print("      ✓ Virtual environment created")
                except Exception as e:
                    print(f"      ✗ ERROR: Could not create virtual environment: {e}")
                    print()
                    print("      Try manually: python3 -m venv venv")
                    print("      Or install: sudo apt install python3-venv")
                    return None, False
        else:
            try:
                print(f"      → Creating virtual environment: {venv_dir}")
                venv.create(venv_dir, with_pip=True)
                print("      ✓ Virtual environment created")
            except Exception as e:
                print(f"      ✗ ERROR: Could not create virtual environment: {e}")
                print()
                print("      Try manually: python3 -m venv venv")
                print("      Or install: sudo apt install python3-venv")
                return None, False
        
        if not os.path.exists(venv_python):
            print(f"      ✗ ERROR: Virtual environment Python not found: {venv_python}")
            return None, False
        
        print(f"      ✓ Installing packages using: {venv_python}")
        return venv_python, True
    
    return sys.executable, False

def install_requirements(requirements_path, python_executable):
    """Install all packages from requirements.txt"""
    print()
    print("[5/5] Installing project dependencies...")
    print("      This may take several minutes...")
    print()
    
    try:
        subprocess.check_call(
            [python_executable, "-m", "pip", "install", "-r", requirements_path]
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"      ✗ ERROR: {str(e)}")
        return False

def main():
    """Main installation function"""
    print_header()
    
    if not check_python_version():
        print_failure()
        input("\nPress Enter to exit...")
        return 1
    
    if not upgrade_pip():
        print_failure()
        input("\nPress Enter to exit...")
        return 1
    
    requirements_path = check_requirements_file()
    if requirements_path is None:
        print_failure()
        input("\nPress Enter to exit...")
        return 1
    
    python_executable, venv_created = check_and_create_venv()
    if python_executable is None:
        print_failure()
        input("\nPress Enter to exit...")
        return 1
    
    if not install_requirements(requirements_path, python_executable):
        print_failure()
        input("\nPress Enter to exit...")
        return 1
    
    print_success()
    input("Press Enter to exit...")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        input("\nPress Enter to exit...")
        sys.exit(1)
