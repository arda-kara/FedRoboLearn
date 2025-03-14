"""
Main entry point for the coordinator.
"""
import argparse
from src.coordinator.coordinator import Coordinator


def main():
    """Main entry point for the coordinator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FL-for-DR Coordinator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Path to log directory")
    args = parser.parse_args()
    
    # Create coordinator
    coordinator = Coordinator(
        config_path=args.config,
        log_dir=args.log_dir
    )
    
    # Run coordinator
    coordinator.run()


if __name__ == "__main__":
    main() 