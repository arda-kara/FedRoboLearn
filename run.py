#!/usr/bin/env python
"""
Script to run the FL-for-DR simulation.
"""
import os
import sys
import argparse
import subprocess


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run FL-for-DR Simulation")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to simulate")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    args = parser.parse_args()
    
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.dirname(__file__))
    
    # Set up environment
    env = os.environ.copy()
    
    # Add the project root to PYTHONPATH to ensure src module is accessible
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    # Run the simulation
    cmd = [
        sys.executable, 
        os.path.join(project_root, "examples", "run_simulation.py"),
        "--num-robots", str(args.num_robots),
        "--config", args.config,
        "--rounds", str(args.rounds)
    ]
    
    print(f"Running simulation with {args.num_robots} robots for {args.rounds} rounds...")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main() 