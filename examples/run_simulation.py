"""
Example script to run a federated learning simulation with multiple robots.
"""
import os
import sys
import time
import subprocess
import argparse
from multiprocessing import Process, Event

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from src to verify the path is correct
try:
    import src
    print(f"Successfully imported src module from {src.__file__}")
except ImportError as e:
    print(f"Error importing src module: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def run_coordinator(config_path, shutdown_event=None):
    """Run the coordinator process."""
    # Add the parent directory to the Python path in the subprocess
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{parent_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = parent_dir
    
    coordinator_script = os.path.join(parent_dir, "src", "coordinator", "main.py")
    cmd = [sys.executable, coordinator_script, "--config", config_path]
    
    print(f"Starting coordinator with command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Start the coordinator process
    process = subprocess.Popen(cmd, env=env)
    
    # Wait for shutdown event if provided
    if shutdown_event:
        while not shutdown_event.is_set():
            # Check if process is still running
            if process.poll() is not None:
                print(f"Coordinator process exited with code {process.returncode}")
                break
            time.sleep(0.5)
        
        # Terminate the process if it's still running
        if process.poll() is None:
            print("Terminating coordinator process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Coordinator process did not terminate, killing...")
                process.kill()
    else:
        # If no shutdown event, just wait for the process to complete
        process.wait()


def run_robot(robot_id, config_path, num_rounds, shutdown_event=None):
    """Run a robot client process."""
    # Add the parent directory to the Python path in the subprocess
    env = os.environ.copy()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{parent_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = parent_dir
    
    robot_script = os.path.join(parent_dir, "src", "robot", "main.py")
    cmd = [
        sys.executable, robot_script,
        "--id", str(robot_id),
        "--config", config_path,
        "--rounds", str(num_rounds)
    ]
    
    print(f"Starting robot {robot_id} with command: {' '.join(cmd)}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")
    
    # Start the robot process
    process = subprocess.Popen(cmd, env=env)
    
    # Wait for shutdown event if provided
    if shutdown_event:
        while not shutdown_event.is_set():
            # Check if process is still running
            if process.poll() is not None:
                print(f"Robot {robot_id} process exited with code {process.returncode}")
                break
            time.sleep(0.5)
        
        # Terminate the process if it's still running
        if process.poll() is None:
            print(f"Terminating robot {robot_id} process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Robot {robot_id} process did not terminate, killing...")
                process.kill()
    else:
        # If no shutdown event, just wait for the process to complete
        process.wait()


def main():
    """Main entry point for the simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FL-for-DR Simulation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--num-robots", type=int, default=3, help="Number of robots to simulate")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    args = parser.parse_args()
    
    # Create a shutdown event for graceful termination
    shutdown_event = Event()
    
    # Create processes
    processes = []
    
    # Start coordinator
    coordinator_process = Process(
        target=run_coordinator, 
        args=(args.config, shutdown_event)
    )
    coordinator_process.daemon = True
    processes.append(coordinator_process)
    
    # Start robots
    for i in range(args.num_robots):
        robot_process = Process(
            target=run_robot, 
            args=(i, args.config, args.rounds, shutdown_event)
        )
        robot_process.daemon = True
        processes.append(robot_process)
    
    # Start all processes
    print(f"Starting coordinator and {args.num_robots} robots for {args.rounds} rounds...")
    
    for p in processes:
        p.start()
        time.sleep(1)  # Stagger start times
    
    # Wait for processes to complete
    try:
        # Wait for all processes to complete
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping all processes...")
        # Signal all processes to shut down
        shutdown_event.set()
        
        # Wait for processes to terminate
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                print(f"Process {p.name} did not terminate, killing...")
                p.terminate()


if __name__ == "__main__":
    main() 