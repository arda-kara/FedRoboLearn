"""
Robot client for FL-for-DR.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import base64
import io
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from collections import OrderedDict

from src.utils.logger import Logger
from src.utils.config import load_config, get_config_value
from src.models.models import get_model
from src.communication.http_comm import HttpClient
from src.communication.mqtt_comm import MqttClient


class RobotClient:
    """
    Robot client for federated learning.
    """
    
    def __init__(
        self,
        robot_id: int,
        config_path: str = "config.yaml",
        data_path: Optional[str] = None,
        log_dir: str = "logs"
    ):
        """
        Initialize the robot client.
        
        Args:
            robot_id: ID of the robot
            config_path: Path to the configuration file
            data_path: Path to the data directory
            log_dir: Path to the log directory
        """
        self.robot_id = robot_id
        self.config = load_config(config_path)
        self.data_path = data_path or f"data/robot{robot_id}"
        
        # Set up logger
        log_level = get_config_value(self.config, "logging.level", "info")
        self.logger = Logger(f"robot_{robot_id}", log_dir, log_level)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Set up communication
        self._setup_communication()
        
        # Set up model
        self._setup_model()
        
        # Set up training
        self._setup_training()
        
        # Set up data
        self.train_loader = None
        self.val_loader = None
        
        # Set up metrics
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Set up state
        self.round = 0
        self.is_training = False
    
    def _setup_communication(self) -> None:
        """Set up communication with the coordinator."""
        protocol = get_config_value(self.config, "communication.protocol", "http")
        
        if protocol.lower() == "http":
            host = get_config_value(self.config, "communication.coordinator_host", "localhost")
            port = get_config_value(self.config, "communication.coordinator_port", 8000)
            use_tls = get_config_value(self.config, "communication.use_tls", False)
            timeout = get_config_value(self.config, "communication.timeout", 30)
            
            self.comm = HttpClient(host, port, use_tls, timeout)
            self.logger.info(f"Using HTTP communication with coordinator at {host}:{port}")
        
        elif protocol.lower() == "mqtt":
            broker = get_config_value(self.config, "communication.mqtt_broker", "localhost")
            port = get_config_value(self.config, "communication.mqtt_port", 1883)
            use_tls = get_config_value(self.config, "communication.use_tls", False)
            timeout = get_config_value(self.config, "communication.timeout", 30)
            
            self.comm = MqttClient(self.robot_id, broker, port, use_tls, timeout)
            self.logger.info(f"Using MQTT communication with broker at {broker}:{port}")
        
        else:
            raise ValueError(f"Communication protocol {protocol} not supported")
    
    def _setup_model(self) -> None:
        """Set up the model."""
        model_name = get_config_value(self.config, "model.name", "cnn")
        input_shape = get_config_value(self.config, "model.input_shape", [3, 32, 32])
        output_dim = get_config_value(self.config, "model.output_dim", 10)
        hidden_layers = get_config_value(self.config, "model.hidden_layers", [64, 128])
        
        self.model = get_model(model_name, input_shape, output_dim, hidden_layers)
        self.model.to(self.device)
        
        self.logger.info(f"Using model: {model_name}")
    
    def _setup_training(self) -> None:
        """Set up training parameters."""
        # Get training parameters from config
        self.batch_size = get_config_value(self.config, "training.batch_size", 32)
        self.learning_rate = get_config_value(self.config, "training.learning_rate", 0.01)
        self.optimizer_name = get_config_value(self.config, "training.optimizer", "sgd")
        self.momentum = get_config_value(self.config, "training.momentum", 0.9)
        self.weight_decay = get_config_value(self.config, "training.weight_decay", 0.0001)
        self.local_epochs = get_config_value(self.config, "training.local_epochs", 2)
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up optimizer
        if self.optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")
        
        # Set up scheduler
        scheduler_name = get_config_value(self.config, "training.scheduler", "step")
        
        if scheduler_name.lower() == "step":
            step_size = get_config_value(self.config, "training.step_size", 5)
            gamma = get_config_value(self.config, "training.gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.local_epochs
            )
        elif scheduler_name.lower() == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Scheduler {scheduler_name} not supported")
    
    def set_data_loaders(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> None:
        """
        Set the data loaders.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Log dataset sizes
        num_train_samples = len(train_loader.dataset)
        self.logger.info(f"Training dataset size: {num_train_samples}")
        
        if val_loader is not None:
            num_val_samples = len(val_loader.dataset)
            self.logger.info(f"Validation dataset size: {num_val_samples}")
    
    def register(self) -> Dict[str, Any]:
        """
        Register with the coordinator.
        
        Returns:
            Response from the coordinator
        """
        # Prepare metadata
        metadata = {
            "device": str(self.device),
            "model": get_config_value(self.config, "model.name", "cnn"),
            "dataset_size": len(self.train_loader.dataset) if self.train_loader else 0
        }
        
        # Register with coordinator
        response = self.comm.register(self.robot_id, metadata)
        
        self.logger.info(f"Registered with coordinator: {response}")
        
        return response
    
    def get_global_model(self) -> None:
        """Get the global model from the coordinator."""
        response = self.comm.get_global_model()
        
        if response.get("status") == "success":
            # Update round
            self.round = response.get("round", 0)
            
            # Deserialize model
            model_base64 = response.get("model")
            model_bytes = base64.b64decode(model_base64)
            model_state_dict = torch.load(io.BytesIO(model_bytes))
            
            # Load model parameters
            self.model.load_state_dict(model_state_dict)
            
            self.logger.info(f"Received global model for round {self.round}")
        else:
            self.logger.error(f"Failed to get global model: {response}")
    
    def train(self) -> Dict[str, float]:
        """
        Train the model on local data.
        
        Returns:
            Training metrics
        """
        if self.train_loader is None:
            self.logger.error("No training data loader set")
            return {}
        
        self.is_training = True
        self.model.train()
        
        # Training metrics
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Train for local epochs
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
                
                # Log batch progress
                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Epoch: {epoch+1}/{self.local_epochs} | "
                        f"Batch: {batch_idx+1}/{len(self.train_loader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Acc: {100.*epoch_correct/epoch_total:.2f}%"
                    )
            
            # Update training metrics
            train_loss += epoch_loss / len(self.train_loader)
            correct += epoch_correct
            total += epoch_total
            
            # Step scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Calculate average metrics
        avg_train_loss = train_loss / self.local_epochs
        train_accuracy = 100. * correct / total
        
        # Store metrics
        self.metrics["train_loss"].append(avg_train_loss)
        self.metrics["train_accuracy"].append(train_accuracy)
        
        # Log training results
        self.logger.info(
            f"Training completed for round {self.round} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Acc: {train_accuracy:.2f}%"
        )
        
        self.is_training = False
        
        return {
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy
        }
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Evaluation metrics
        """
        if self.val_loader is None:
            self.logger.warning("No validation data loader set")
            return {}
        
        self.model.eval()
        
        # Evaluation metrics
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        
        # Store metrics
        self.metrics["val_loss"].append(avg_val_loss)
        self.metrics["val_accuracy"].append(val_accuracy)
        
        # Log validation results
        self.logger.info(
            f"Validation for round {self.round} | "
            f"Loss: {avg_val_loss:.4f} | "
            f"Acc: {val_accuracy:.2f}%"
        )
        
        return {
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        }
    
    def send_update(self) -> Dict[str, Any]:
        """
        Send model update to the coordinator.
        
        Returns:
            Response from the coordinator
        """
        # Get model parameters
        model_update = self.model.state_dict()
        
        # Prepare metrics
        metrics = {
            "train_loss": self.metrics["train_loss"][-1] if self.metrics["train_loss"] else 0.0,
            "train_accuracy": self.metrics["train_accuracy"][-1] if self.metrics["train_accuracy"] else 0.0
        }
        
        if self.metrics["val_loss"]:
            metrics["val_loss"] = self.metrics["val_loss"][-1]
            metrics["val_accuracy"] = self.metrics["val_accuracy"][-1]
        
        # Prepare metadata
        metadata = {
            "dataset_size": len(self.train_loader.dataset) if self.train_loader else 0,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }
        
        # Send update to coordinator
        response = self.comm.send_update(self.robot_id, model_update, metrics, metadata)
        
        self.logger.info(f"Sent model update for round {self.round}: {response}")
        
        return response
    
    def federated_learning_round(self) -> None:
        """Perform one round of federated learning."""
        # Get global model
        self.get_global_model()
        
        # Train on local data
        train_metrics = self.train()
        
        # Evaluate on validation data
        val_metrics = self.evaluate()
        
        # Send update to coordinator
        self.send_update()
        
        # Increment round
        self.round += 1
    
    def run(self, num_rounds: Optional[int] = None) -> None:
        """
        Run the federated learning process.
        
        Args:
            num_rounds: Number of rounds to run (if None, run indefinitely)
        """
        # Register with coordinator
        self.register()
        
        # Run for specified number of rounds or indefinitely
        round_count = 0
        max_rounds = num_rounds or float('inf')
        
        while round_count < max_rounds:
            try:
                self.logger.info(f"Starting federated learning round {round_count+1}")
                self.federated_learning_round()
                round_count += 1
                
                # Sleep between rounds
                time.sleep(1)
            
            except KeyboardInterrupt:
                self.logger.info("Interrupted by user")
                break
            
            except Exception as e:
                self.logger.error(f"Error in round {round_count+1}: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
        self.logger.info(f"Completed {round_count} rounds of federated learning") 