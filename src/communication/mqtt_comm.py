"""
MQTT-based communication module for FL-for-DR.
"""
import json
import time
import torch
import base64
import io
import threading
import paho.mqtt.client as mqtt
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from collections import OrderedDict


class MqttClient:
    """
    MQTT client for communication with the coordinator.
    """
    
    def __init__(
        self,
        robot_id: int,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        use_tls: bool = False,
        timeout: int = 30
    ):
        """
        Initialize the MQTT client.
        
        Args:
            robot_id: ID of the robot
            broker_host: Hostname of the MQTT broker
            broker_port: Port of the MQTT broker
            use_tls: Whether to use TLS
            timeout: Connection timeout in seconds
        """
        self.robot_id = robot_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.use_tls = use_tls
        self.timeout = timeout
        
        # Create MQTT client
        self.client = mqtt.Client(f"robot_{robot_id}")
        
        # Set up TLS if needed
        if use_tls:
            self.client.tls_set()
        
        # Set up topics
        self.register_topic = "fl/register"
        self.global_model_topic = "fl/global_model"
        self.update_topic = f"fl/update/{robot_id}"
        self.status_topic = "fl/status"
        
        # Set up response handlers
        self.responses = {}
        self.callbacks = {}
        
        # Set up message handler
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client connects to the broker.
        """
        if rc == 0:
            print(f"Robot {self.robot_id} connected to MQTT broker")
            # Subscribe to global model updates
            self.client.subscribe(self.global_model_topic)
            # Subscribe to status updates
            self.client.subscribe(self.status_topic)
        else:
            print(f"Connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """
        Callback for when a message is received.
        """
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        # Store response
        self.responses[topic] = payload
        
        # Call callback if registered
        if topic in self.callbacks:
            self.callbacks[topic](payload)
    
    def connect(self) -> None:
        """
        Connect to the MQTT broker.
        """
        self.client.connect(self.broker_host, self.broker_port, self.timeout)
        self.client.loop_start()
    
    def disconnect(self) -> None:
        """
        Disconnect from the MQTT broker.
        """
        self.client.loop_stop()
        self.client.disconnect()
    
    def register(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register the robot with the coordinator.
        
        Args:
            metadata: Metadata about the robot (e.g., dataset size, capabilities)
            
        Returns:
            Response from the coordinator
        """
        data = {
            "robot_id": self.robot_id,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        # Publish registration message
        self.client.publish(self.register_topic, json.dumps(data))
        
        # Wait for response (in a real implementation, use a more robust approach)
        time.sleep(1)
        
        return self.responses.get(self.register_topic, {"status": "unknown"})
    
    def get_global_model(self, callback: Optional[Callable] = None) -> None:
        """
        Register a callback for when a new global model is received.
        
        Args:
            callback: Function to call when a new global model is received
        """
        if callback:
            self.callbacks[self.global_model_topic] = callback
    
    def send_update(
        self,
        model_update: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a model update to the coordinator.
        
        Args:
            model_update: Model parameters or gradients
            metrics: Training metrics (e.g., loss, accuracy)
            metadata: Additional metadata
        """
        # Serialize model update
        model_bytes = io.BytesIO()
        torch.save(model_update, model_bytes)
        model_bytes.seek(0)
        model_base64 = base64.b64encode(model_bytes.read()).decode('utf-8')
        
        data = {
            "robot_id": self.robot_id,
            "model_update": model_base64,
            "metrics": metrics,
            "timestamp": time.time()
        }
        
        if metadata is not None:
            data["metadata"] = metadata
        
        # Publish update
        self.client.publish(self.update_topic, json.dumps(data))
    
    def get_status(self, callback: Optional[Callable] = None) -> None:
        """
        Register a callback for when a status update is received.
        
        Args:
            callback: Function to call when a status update is received
        """
        if callback:
            self.callbacks[self.status_topic] = callback


class MqttCoordinator:
    """
    MQTT coordinator for federated learning.
    """
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        use_tls: bool = False
    ):
        """
        Initialize the MQTT coordinator.
        
        Args:
            broker_host: Hostname of the MQTT broker
            broker_port: Port of the MQTT broker
            use_tls: Whether to use TLS
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.use_tls = use_tls
        
        # Create MQTT client
        self.client = mqtt.Client("fl_coordinator")
        
        # Set up TLS if needed
        if use_tls:
            self.client.tls_set()
        
        # Set up topics
        self.register_topic = "fl/register"
        self.global_model_topic = "fl/global_model"
        self.update_topic = "fl/update/+"  # + is a wildcard for robot_id
        self.status_topic = "fl/status"
        
        # Set up state
        self.robots = {}
        self.global_model = None
        self.round = 0
        self.updates = {}
        
        # Set up message handler
        self.client.on_message = self._on_message
        self.client.on_connect = self._on_connect
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client connects to the broker.
        """
        if rc == 0:
            print("Coordinator connected to MQTT broker")
            # Subscribe to registration messages
            self.client.subscribe(self.register_topic)
            # Subscribe to update messages from all robots
            self.client.subscribe(self.update_topic)
        else:
            print(f"Connection failed with code {rc}")
    
    def _on_message(self, client, userdata, msg):
        """
        Callback for when a message is received.
        """
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        
        if topic == self.register_topic:
            # Handle registration
            robot_id = payload.get("robot_id")
            metadata = payload.get("metadata", {})
            self.robots[robot_id] = metadata
            print(f"Robot {robot_id} registered")
        
        elif topic.startswith("fl/update/"):
            # Handle update
            robot_id = int(topic.split("/")[-1])
            model_update_base64 = payload.get("model_update")
            metrics = payload.get("metrics", {})
            metadata = payload.get("metadata", {})
            
            # Deserialize model update
            model_bytes = base64.b64decode(model_update_base64)
            model_update = torch.load(io.BytesIO(model_bytes))
            
            # Store update
            self.updates[robot_id] = {
                "model_update": model_update,
                "metrics": metrics,
                "metadata": metadata
            }
            
            print(f"Received update from robot {robot_id}")
    
    def connect(self) -> None:
        """
        Connect to the MQTT broker.
        """
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_start()
    
    def disconnect(self) -> None:
        """
        Disconnect from the MQTT broker.
        """
        self.client.loop_stop()
        self.client.disconnect()
    
    def broadcast_global_model(self, model: Any) -> None:
        """
        Broadcast the global model to all robots.
        
        Args:
            model: Global model parameters
        """
        self.global_model = model
        
        # Serialize model
        model_bytes = io.BytesIO()
        torch.save(model, model_bytes)
        model_bytes.seek(0)
        model_base64 = base64.b64encode(model_bytes.read()).decode('utf-8')
        
        data = {
            "round": self.round,
            "model": model_base64,
            "timestamp": time.time()
        }
        
        # Publish global model
        self.client.publish(self.global_model_topic, json.dumps(data))
    
    def broadcast_status(self) -> None:
        """
        Broadcast the current status to all robots.
        """
        data = {
            "round": self.round,
            "num_robots": len(self.robots),
            "num_updates": len(self.updates),
            "timestamp": time.time()
        }
        
        # Publish status
        self.client.publish(self.status_topic, json.dumps(data))
    
    def get_updates(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all updates received from robots.
        
        Returns:
            Dictionary mapping robot IDs to their updates
        """
        return self.updates
    
    def clear_updates(self) -> None:
        """Clear all updates."""
        self.updates = {}
    
    def increment_round(self) -> None:
        """Increment the round counter."""
        self.round += 1 