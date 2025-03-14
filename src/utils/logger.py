"""
Logging utilities for FL-for-DR.
"""
import os
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime


class Logger:
    """
    Logger class for FL-for-DR.
    """
    
    def __init__(
        self, 
        name: str, 
        log_dir: str = "logs", 
        level: str = "info",
        use_console: bool = True,
        use_file: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level (debug, info, warning, error)
            use_console: Whether to log to console
            use_file: Whether to log to file
        """
        self.name = name
        self.log_dir = log_dir
        self.level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }
        self.level = self.level_map.get(level.lower(), logging.INFO)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        if use_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if use_file:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics dictionary.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step or round number
        """
        step_str = f" (Step {step})" if step is not None else ""
        self.info(f"Metrics{step_str}: {metrics}")


class TensorboardLogger:
    """
    Tensorboard logger for FL-for-DR.
    """
    
    def __init__(self, log_dir: str = "logs/tensorboard"):
        """
        Initialize the Tensorboard logger.
        
        Args:
            log_dir: Directory to save Tensorboard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join(log_dir, timestamp)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.enabled = True
        except ImportError:
            print("Tensorboard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to Tensorboard.
        
        Args:
            tag: Data identifier
            value: Value to log
            step: Global step value
        """
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """
        Log multiple scalars to Tensorboard.
        
        Args:
            main_tag: Parent name for the tags
            tag_scalar_dict: Dictionary of tag names and values
            step: Global step value
        """
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self) -> None:
        """Close the Tensorboard writer."""
        if self.enabled:
            self.writer.close() 