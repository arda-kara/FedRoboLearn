"""
Main entry point for the robot client.
"""
import os
import argparse
import torch
from src.robot.client import RobotClient
from src.utils.config import load_config, get_config_value
from src.utils.data_utils import get_dataset, get_data_loaders, create_iid_partition, create_non_iid_partition


def main():
    """Main entry point for the robot client."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FL-for-DR Robot Client")
    parser.add_argument("--id", type=int, required=True, help="Robot ID")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Path to log directory")
    parser.add_argument("--rounds", type=int, help="Number of rounds to run")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create robot client
    client = RobotClient(
        robot_id=args.id,
        config_path=args.config,
        data_path=args.data_path,
        log_dir=args.log_dir
    )
    
    # Set up data loaders
    setup_data_loaders(client, config, args.id)
    
    # Run federated learning
    client.run(num_rounds=args.rounds)


def setup_data_loaders(client, config, robot_id):
    """
    Set up data loaders for the robot client.
    
    Args:
        client: Robot client
        config: Configuration dictionary
        robot_id: Robot ID
    """
    # Get simulation parameters
    simulation_enabled = get_config_value(config, "simulation.enabled", True)
    
    if simulation_enabled:
        # Get simulation parameters
        dataset_name = get_config_value(config, "simulation.dataset", "cifar10")
        num_robots = get_config_value(config, "simulation.num_robots", 3)
        distribution = get_config_value(config, "simulation.data_distribution", "iid")
        alpha = get_config_value(config, "simulation.non_iid_alpha", 0.5)
        batch_size = get_config_value(config, "training.batch_size", 32)
        
        # Set random seed for reproducibility
        random_seed = get_config_value(config, "simulation.random_seed", 42)
        torch.manual_seed(random_seed)
        
        # Get dataset
        data_dir = os.path.join("data", dataset_name)
        train_dataset = get_dataset(dataset_name, data_dir, train=True)
        test_dataset = get_dataset(dataset_name, data_dir, train=False)
        
        # Create partitions
        if distribution.lower() == "iid":
            train_partitions = create_iid_partition(train_dataset, num_robots)
            test_partitions = create_iid_partition(test_dataset, num_robots)
        elif distribution.lower() == "non_iid":
            train_partitions = create_non_iid_partition(train_dataset, num_robots, alpha)
            test_partitions = create_non_iid_partition(test_dataset, num_robots, alpha)
        else:
            raise ValueError(f"Distribution {distribution} not supported")
        
        # Get data loaders for this robot
        if robot_id < num_robots:
            train_loader = get_data_loaders(
                train_dataset, train_partitions[robot_id], batch_size, shuffle=True
            )
            test_loader = get_data_loaders(
                test_dataset, test_partitions[robot_id], batch_size, shuffle=False
            )
            
            # Set data loaders
            client.set_data_loaders(train_loader, test_loader)
        else:
            raise ValueError(f"Robot ID {robot_id} exceeds number of robots {num_robots}")
    else:
        # In a real-world scenario, each robot would load its own data
        # This is just a placeholder for custom data loading
        raise NotImplementedError("Custom data loading not implemented yet")


if __name__ == "__main__":
    main() 