"""
HTTP-based communication module for FL-for-DR.
"""
import json
import requests
import torch
import base64
import io
import pickle
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import OrderedDict


class HttpClient:
    """
    HTTP client for communication with the coordinator.
    """
    
    def __init__(
        self,
        coordinator_host: str = "localhost",
        coordinator_port: int = 8000,
        use_tls: bool = False,
        timeout: int = 30
    ):
        """
        Initialize the HTTP client.
        
        Args:
            coordinator_host: Hostname of the coordinator
            coordinator_port: Port of the coordinator
            use_tls: Whether to use HTTPS
            timeout: Connection timeout in seconds
        """
        self.protocol = "https" if use_tls else "http"
        self.base_url = f"{self.protocol}://{coordinator_host}:{coordinator_port}"
        self.timeout = timeout
    
    def register(self, robot_id: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register the robot with the coordinator.
        
        Args:
            robot_id: ID of the robot
            metadata: Metadata about the robot (e.g., dataset size, capabilities)
            
        Returns:
            Response from the coordinator
        """
        url = f"{self.base_url}/register"
        data = {
            "robot_id": robot_id,
            "metadata": metadata
        }
        
        response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def get_global_model(self) -> Dict[str, Any]:
        """
        Get the global model from the coordinator.
        
        Returns:
            Global model parameters and metadata
        """
        url = f"{self.base_url}/global_model"
        
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def send_update(
        self,
        robot_id: int,
        model_update: Dict[str, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a model update to the coordinator.
        
        Args:
            robot_id: ID of the robot
            model_update: Model parameters or gradients
            metrics: Training metrics (e.g., loss, accuracy)
            metadata: Additional metadata
            
        Returns:
            Response from the coordinator
        """
        url = f"{self.base_url}/update"
        
        # Serialize model update
        model_bytes = io.BytesIO()
        torch.save(model_update, model_bytes)
        model_bytes.seek(0)
        model_base64 = base64.b64encode(model_bytes.read()).decode('utf-8')
        
        data = {
            "robot_id": robot_id,
            "model_update": model_base64,
            "metrics": metrics
        }
        
        if metadata is not None:
            data["metadata"] = metadata
        
        response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the federated learning process.
        
        Returns:
            Status information
        """
        url = f"{self.base_url}/status"
        
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return response.json()


class HttpServer:
    """
    HTTP server for the coordinator.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        use_tls: bool = False
    ):
        """
        Initialize the HTTP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            use_tls: Whether to use HTTPS
        """
        self.host = host
        self.port = port
        self.use_tls = use_tls
        
        # Import Flask here to avoid dependency if not used
        from flask import Flask, request, jsonify
        
        self.app = Flask(__name__)
        self.robots = {}
        self.global_model = None
        self.round = 0
        self.updates = {}
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes."""
        from flask import request, jsonify
        
        @self.app.route('/register', methods=['POST'])
        def register():
            data = request.json
            robot_id = data.get('robot_id')
            metadata = data.get('metadata', {})
            
            self.robots[robot_id] = metadata
            
            return jsonify({
                "status": "success",
                "message": f"Robot {robot_id} registered successfully",
                "round": self.round
            })
        
        @self.app.route('/global_model', methods=['GET'])
        def get_global_model():
            if self.global_model is None:
                return jsonify({
                    "status": "error",
                    "message": "Global model not available"
                }), 404
            
            # Serialize model
            model_bytes = io.BytesIO()
            torch.save(self.global_model, model_bytes)
            model_bytes.seek(0)
            model_base64 = base64.b64encode(model_bytes.read()).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "round": self.round,
                "model": model_base64
            })
        
        @self.app.route('/update', methods=['POST'])
        def receive_update():
            data = request.json
            robot_id = data.get('robot_id')
            model_update_base64 = data.get('model_update')
            metrics = data.get('metrics', {})
            metadata = data.get('metadata', {})
            
            # Deserialize model update
            model_bytes = base64.b64decode(model_update_base64)
            model_update = torch.load(io.BytesIO(model_bytes))
            
            # Store update
            self.updates[robot_id] = {
                "model_update": model_update,
                "metrics": metrics,
                "metadata": metadata
            }
            
            return jsonify({
                "status": "success",
                "message": f"Update from robot {robot_id} received",
                "round": self.round
            })
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            return jsonify({
                "status": "success",
                "round": self.round,
                "num_robots": len(self.robots),
                "num_updates": len(self.updates)
            })
    
    def set_global_model(self, model: Any) -> None:
        """
        Set the global model.
        
        Args:
            model: Global model parameters
        """
        self.global_model = model
    
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
    
    def run(self, debug: bool = False) -> None:
        """
        Run the server.
        
        Args:
            debug: Whether to run in debug mode
        """
        if self.use_tls:
            # For HTTPS, you need to provide SSL context
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug,
                ssl_context='adhoc'  # Use 'adhoc' for development, provide cert/key for production
            )
        else:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug
            ) 