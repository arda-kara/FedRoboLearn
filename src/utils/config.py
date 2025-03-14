"""
Configuration utilities for FL-for-DR.
"""
import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_config_value(config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
    """
    Get a value from nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Path to the key using dot notation (e.g., "model.learning_rate")
        default: Default value to return if key is not found
        
    Returns:
        Value at the specified key path or default if not found
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def update_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Update a value in nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Path to the key using dot notation (e.g., "model.learning_rate")
        value: New value to set
        
    Returns:
        Updated configuration dictionary
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the nested dictionary containing the key to update
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Update the value
    current[keys[-1]] = value
    
    return config 