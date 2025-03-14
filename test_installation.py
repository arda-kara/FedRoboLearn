#!/usr/bin/env python
"""
Test script to verify the FL-for-DR installation.
"""
import os
import sys
import importlib
from colorama import init, Fore, Style

# Initialize colorama
init()

# Add the current directory to the Python path
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def check_module(module_name):
    """
    Check if a module can be imported.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        True if the module can be imported, False otherwise
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def print_status(message, success):
    """
    Print a status message with color.
    
    Args:
        message: Message to print
        success: Whether the status is successful
    """
    if success:
        print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def main():
    """Main entry point."""
    print(f"{Fore.CYAN}Testing FL-for-DR installation...{Style.RESET_ALL}\n")
    
    # Check if the package is installed
    package_installed = check_module("src")
    print_status("FL-for-DR package is installed", package_installed)
    
    if not package_installed:
        print(f"\n{Fore.YELLOW}The FL-for-DR package is not installed or not in the Python path.{Style.RESET_ALL}")
        print("Please try one of the following solutions:")
        print("  1. Run: python install.py")
        print("  2. Add the project directory to your PYTHONPATH:")
        print(f"     export PYTHONPATH={current_dir}:$PYTHONPATH  # Linux/Mac")
        print(f"     set PYTHONPATH={current_dir};%PYTHONPATH%  # Windows")
        return
    
    # Check dependencies
    dependencies = [
        "torch", "numpy", "matplotlib", "tqdm", "yaml", "requests",
        "flask", "cryptography", "tensorboard", "PIL", "sklearn",
        "flwr", "paho.mqtt", "colorama"
    ]
    
    print("\nChecking dependencies:")
    all_deps_installed = True
    
    for dep in dependencies:
        try:
            if dep == "yaml":
                # Special case for PyYAML
                import yaml
                installed = True
            elif dep == "PIL":
                # Special case for Pillow
                from PIL import Image
                installed = True
            elif dep == "sklearn":
                # Special case for scikit-learn
                from sklearn import __version__
                installed = True
            else:
                installed = check_module(dep)
            
            print_status(f"{dep}", installed)
            
            if not installed:
                all_deps_installed = False
        except Exception as e:
            print_status(f"{dep} (Error: {str(e)})", False)
            all_deps_installed = False
    
    # Check if directories exist
    print("\nChecking directories:")
    directories = ["logs", "data", "data/robot1", "data/robot2", "data/robot3"]
    all_dirs_exist = True
    
    for directory in directories:
        exists = os.path.isdir(directory)
        print_status(f"{directory}/", exists)
        
        if not exists:
            all_dirs_exist = False
    
    # Check if config file exists
    config_exists = os.path.isfile("config.yaml")
    print_status("config.yaml", config_exists)
    
    # Print summary
    print("\nSummary:")
    if package_installed and all_deps_installed and all_dirs_exist and config_exists:
        print(f"{Fore.GREEN}FL-for-DR is correctly installed!{Style.RESET_ALL}")
        print("\nYou can run the simulation with:")
        print("  python run.py --num-robots 3")
    else:
        print(f"{Fore.YELLOW}There are some issues with the installation.{Style.RESET_ALL}")
        print("Please fix the issues above and try again.")
        print("\nIf the src module is not found, try:")
        print(f"  export PYTHONPATH={current_dir}:$PYTHONPATH  # Linux/Mac")
        print(f"  set PYTHONPATH={current_dir};%PYTHONPATH%  # Windows")


if __name__ == "__main__":
    main() 