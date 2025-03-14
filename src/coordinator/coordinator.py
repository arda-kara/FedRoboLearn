"""
Coordinator for FL-for-DR.
"""
import os
import time
import torch
import threading
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from collections import OrderedDict

from src.utils.logger import Logger, TensorboardLogger
from src.utils.config import load_config, get_config_value
from src.models.models import get_model
from src.models.federated import aggregate_models
from src.communication.http_comm import HttpServer
from src.communication.mqtt_comm import MqttCoordinator


class Coordinator:
    """
    Coordinator for federated learning.
    """
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        log_dir: str = "logs"
    ):
        """
        Initialize the coordinator.
        
        Args:
            config_path: Path to the configuration file
            log_dir: Path to the log directory
        """
        self.config = load_config(config_path)
        
        # Set up logger
        log_level = get_config_value(self.config, "logging.level", "info")
        self.logger = Logger("coordinator", log_dir, log_level)
        
        # Set up tensorboard logger if enabled
        use_tensorboard = get_config_value(self.config, "logging.use_tensorboard", True)
        if use_tensorboard:
            self.tb_logger = TensorboardLogger(os.path.join(log_dir, "tensorboard"))
        else:
            self.tb_logger = None
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Set up communication
        self._setup_communication()
        
        # Set up model
        self._setup_model()
        
        # Set up federated learning parameters
        self._setup_federated_params()
        
        # Set up state
        self.round = 0
        self.robots = {}
        self.updates = {}
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Set up threading
        self.aggregation_thread = None
        self.running = False
    
    def _setup_communication(self) -> None:
        """Set up communication server."""
        protocol = get_config_value(self.config, "communication.protocol", "http")
        
        if protocol.lower() == "http":
            host = get_config_value(self.config, "communication.coordinator_host", "0.0.0.0")
            port = get_config_value(self.config, "communication.coordinator_port", 8000)
            use_tls = get_config_value(self.config, "communication.use_tls", False)
            
            self.comm = HttpServer(host, port, use_tls)
            self.logger.info(f"Using HTTP server at {host}:{port}")
        
        elif protocol.lower() == "mqtt":
            broker = get_config_value(self.config, "communication.mqtt_broker", "localhost")
            port = get_config_value(self.config, "communication.mqtt_port", 1883)
            use_tls = get_config_value(self.config, "communication.use_tls", False)
            
            self.comm = MqttCoordinator(broker, port, use_tls)
            self.logger.info(f"Using MQTT coordinator with broker at {broker}:{port}")
        
        else:
            raise ValueError(f"Communication protocol {protocol} not supported")
    
    def _setup_model(self) -> None:
        """Set up the global model."""
        model_name = get_config_value(self.config, "model.name", "cnn")
        input_shape = get_config_value(self.config, "model.input_shape", [3, 32, 32])
        output_dim = get_config_value(self.config, "model.output_dim", 10)
        hidden_layers = get_config_value(self.config, "model.hidden_layers", [64, 128])
        
        self.model = get_model(model_name, input_shape, output_dim, hidden_layers)
        self.model.to(self.device)
        
        self.logger.info(f"Using model: {model_name}")
        
        # Set initial model parameters in the communication server
        self.comm.set_global_model(self.model.state_dict())
    
    def _setup_federated_params(self) -> None:
        """Set up federated learning parameters."""
        self.rounds = get_config_value(self.config, "federated.rounds", 10)
        self.min_clients = get_config_value(self.config, "federated.min_clients", 2)
        self.min_sample_size = get_config_value(self.config, "federated.min_sample_size", 10)
        self.aggregation_method = get_config_value(self.config, "federated.aggregation_method", "fedavg")
        self.client_fraction = get_config_value(self.config, "federated.client_fraction", 1.0)
        self.proximal_mu = get_config_value(self.config, "federated.proximal_mu", 0.01)
        
        self.logger.info(f"Federated learning parameters:")
        self.logger.info(f"  Rounds: {self.rounds}")
        self.logger.info(f"  Min clients: {self.min_clients}")
        self.logger.info(f"  Aggregation method: {self.aggregation_method}")
    
    def aggregate_updates(self) -> None:
        """Aggregate model updates from robots."""
        # Get updates from communication server
        updates = self.comm.get_updates()
        
        if len(updates) < self.min_clients:
            self.logger.warning(
                f"Not enough clients for aggregation: {len(updates)} < {self.min_clients}"
            )
            return
        
        self.logger.info(f"Aggregating updates from {len(updates)} robots")
        
        # Extract model updates and weights
        model_updates = []
        weights = []
        
        for robot_id, update in updates.items():
            model_update = update["model_update"]
            metadata = update.get("metadata", {})
            
            # Get dataset size for weighting
            dataset_size = metadata.get("dataset_size", 0)
            
            if dataset_size >= self.min_sample_size:
                model_updates.append(model_update)
                weights.append(dataset_size)
            else:
                self.logger.warning(
                    f"Robot {robot_id} has too few samples: {dataset_size} < {self.min_sample_size}"
                )
        
        if not model_updates:
            self.logger.warning("No valid updates for aggregation")
            return
        
        # Aggregate model updates
        if self.aggregation_method.lower() == "fedavg":
            global_model = aggregate_models(
                model_updates, weights=weights, method="fedavg"
            )
        elif self.aggregation_method.lower() == "fedprox":
            global_model = aggregate_models(
                model_updates, self.model.state_dict(),
                weights=weights, method="fedprox", mu=self.proximal_mu
            )
        else:
            raise ValueError(f"Aggregation method {self.aggregation_method} not supported")
        
        # Update global model
        self.model.load_state_dict(global_model)
        
        # Update global model in communication server
        self.comm.set_global_model(global_model)
        
        # Aggregate metrics
        self._aggregate_metrics(updates)
        
        # Clear updates
        self.comm.clear_updates()
        
        # Increment round
        self.round += 1
        self.comm.increment_round()
        
        self.logger.info(f"Completed round {self.round}/{self.rounds}")
        
        # Log status
        if hasattr(self.comm, "broadcast_status"):
            self.comm.broadcast_status()
    
    def _aggregate_metrics(self, updates: Dict[int, Dict[str, Any]]) -> None:
        """
        Aggregate metrics from robot updates.
        
        Args:
            updates: Dictionary mapping robot IDs to their updates
        """
        # Extract metrics
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        for robot_id, update in updates.items():
            metrics = update.get("metrics", {})
            
            if "train_loss" in metrics:
                train_losses.append(metrics["train_loss"])
            
            if "train_accuracy" in metrics:
                train_accuracies.append(metrics["train_accuracy"])
            
            if "val_loss" in metrics:
                val_losses.append(metrics["val_loss"])
            
            if "val_accuracy" in metrics:
                val_accuracies.append(metrics["val_accuracy"])
        
        # Calculate average metrics
        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            self.metrics["train_loss"].append(avg_train_loss)
            
            if self.tb_logger:
                self.tb_logger.log_scalar("train/loss", avg_train_loss, self.round)
        
        if train_accuracies:
            avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
            self.metrics["train_accuracy"].append(avg_train_accuracy)
            
            if self.tb_logger:
                self.tb_logger.log_scalar("train/accuracy", avg_train_accuracy, self.round)
        
        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
            self.metrics["val_loss"].append(avg_val_loss)
            
            if self.tb_logger:
                self.tb_logger.log_scalar("val/loss", avg_val_loss, self.round)
        
        if val_accuracies:
            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            self.metrics["val_accuracy"].append(avg_val_accuracy)
            
            if self.tb_logger:
                self.tb_logger.log_scalar("val/accuracy", avg_val_accuracy, self.round)
        
        # Log metrics
        log_metrics = {}
        
        if train_losses:
            log_metrics["train_loss"] = self.metrics["train_loss"][-1]
        
        if train_accuracies:
            log_metrics["train_accuracy"] = self.metrics["train_accuracy"][-1]
        
        if val_losses:
            log_metrics["val_loss"] = self.metrics["val_loss"][-1]
        
        if val_accuracies:
            log_metrics["val_accuracy"] = self.metrics["val_accuracy"][-1]
        
        self.logger.log_metrics(log_metrics, self.round)
    
    def _aggregation_loop(self) -> None:
        """Main aggregation loop."""
        self.logger.info("Starting aggregation loop")
        
        while self.running and self.round < self.rounds:
            try:
                # Wait for updates
                time.sleep(5)
                
                # Check if we have enough updates
                updates = self.comm.get_updates()
                
                if len(updates) >= self.min_clients:
                    self.logger.info(f"Received {len(updates)} updates, aggregating...")
                    self.aggregate_updates()
                else:
                    self.logger.info(
                        f"Waiting for more updates: {len(updates)}/{self.min_clients}"
                    )
            
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
        
        self.logger.info("Aggregation loop completed")
    
    def start(self) -> None:
        """Start the coordinator."""
        self.logger.info("Starting coordinator")
        
        # Start communication server
        if isinstance(self.comm, HttpServer):
            # Start HTTP server in a separate thread
            self.server_thread = threading.Thread(
                target=self.comm.run, kwargs={"debug": False}
            )
            self.server_thread.daemon = True
            self.server_thread.start()
        elif isinstance(self.comm, MqttCoordinator):
            # Connect to MQTT broker
            self.comm.connect()
        
        # Start aggregation loop
        self.running = True
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
    
    def stop(self) -> None:
        """Stop the coordinator."""
        self.logger.info("Stopping coordinator")
        
        # Stop aggregation loop
        self.running = False
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        
        # Stop communication server
        if isinstance(self.comm, MqttCoordinator):
            self.comm.disconnect()
        
        # Close tensorboard logger
        if self.tb_logger:
            self.tb_logger.close()
    
    def run(self) -> None:
        """Run the coordinator until completion."""
        try:
            self.start()
            
            # Wait for completion
            while self.running and self.round < self.rounds:
                time.sleep(1)
            
            self.logger.info(f"Federated learning completed after {self.round} rounds")
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        finally:
            self.stop() 