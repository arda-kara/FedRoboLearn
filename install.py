#!/usr/bin/env python
"""
Installation script for FL-for-DR.
"""
import os
import sys
import subprocess
import argparse


def install_package(dev_mode=False):
    """
    Install the package.
    
    Args:
        dev_mode: Whether to install in development mode
    """
    # First install requirements
    print("Installing requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Then install the package
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if dev_mode:
        cmd.append("-e")
    
    cmd.append(".")
    
    print(f"Installing package with: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Verify installation
    try:
        subprocess.run([sys.executable, "-c", "import src"], check=True)
        print("Package installation verified successfully!")
    except subprocess.CalledProcessError:
        print("WARNING: Package installation could not be verified. You may need to add the project directory to your PYTHONPATH.")


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["logs", "data/robot1", "data/robot2", "data/robot3"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Install FL-for-DR")
    parser.add_argument("--dev", action="store_true", help="Install in development mode")
    args = parser.parse_args()
    
    print("Installing FL-for-DR...")
    
    # Install package
    try:
        install_package(args.dev)
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    print("\nInstallation complete!")
    print("\nTo test the installation:")
    print("  python test_installation.py")
    print("\nTo run the simulation:")
    print("  python run.py --num-robots 3")


if __name__ == "__main__":
    main() 