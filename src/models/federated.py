"""
Federated learning algorithms for FL-for-DR.
"""
import copy
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict


def fedavg_aggregate(
    local_models_params: List[OrderedDict],
    weights: Optional[List[float]] = None
) -> OrderedDict:
    """
    Federated Averaging (FedAvg) algorithm for model aggregation.
    
    Args:
        local_models_params: List of local model parameters (state_dicts)
        weights: Weights for each model (e.g., based on number of samples)
                If None, equal weights are used
    
    Returns:
        Aggregated model parameters
    """
    if not local_models_params:
        raise ValueError("No local models provided for aggregation")
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(local_models_params)] * len(local_models_params)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Initialize with the first model's parameters
    global_params = copy.deepcopy(local_models_params[0])
    
    # Set all parameters to 0
    for key in global_params.keys():
        global_params[key] = global_params[key] * 0.0
    
    # Weighted average of parameters
    for local_params, weight in zip(local_models_params, weights):
        for key in global_params.keys():
            global_params[key] += local_params[key] * weight
    
    return global_params


def fedprox_aggregate(
    local_models_params: List[OrderedDict],
    global_model_params: OrderedDict,
    mu: float = 0.01,
    weights: Optional[List[float]] = None
) -> OrderedDict:
    """
    FedProx algorithm for model aggregation with proximal term.
    
    Args:
        local_models_params: List of local model parameters (state_dicts)
        global_model_params: Global model parameters from previous round
        mu: Proximal term weight
        weights: Weights for each model (e.g., based on number of samples)
                If None, equal weights are used
    
    Returns:
        Aggregated model parameters
    """
    if not local_models_params:
        raise ValueError("No local models provided for aggregation")
    
    # Use equal weights if not provided
    if weights is None:
        weights = [1.0 / len(local_models_params)] * len(local_models_params)
    else:
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    # Initialize with the first model's parameters
    global_params = copy.deepcopy(local_models_params[0])
    
    # Set all parameters to 0
    for key in global_params.keys():
        global_params[key] = global_params[key] * 0.0
    
    # Weighted average of parameters with proximal term
    for local_params, weight in zip(local_models_params, weights):
        for key in global_params.keys():
            # Add proximal term: (local_params - global_params) * mu
            proximal_term = (local_params[key] - global_model_params[key]) * mu
            global_params[key] += (local_params[key] - proximal_term) * weight
    
    return global_params


def aggregate_models(
    local_models_params: List[OrderedDict],
    global_model_params: Optional[OrderedDict] = None,
    method: str = "fedavg",
    weights: Optional[List[float]] = None,
    **kwargs
) -> OrderedDict:
    """
    Aggregate local models using the specified method.
    
    Args:
        local_models_params: List of local model parameters (state_dicts)
        global_model_params: Global model parameters from previous round
                            (required for some methods like FedProx)
        method: Aggregation method ("fedavg", "fedprox")
        weights: Weights for each model
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Aggregated model parameters
    """
    if method.lower() == "fedavg":
        return fedavg_aggregate(local_models_params, weights)
    elif method.lower() == "fedprox":
        if global_model_params is None:
            raise ValueError("Global model parameters required for FedProx")
        
        mu = kwargs.get("mu", 0.01)
        return fedprox_aggregate(local_models_params, global_model_params, mu, weights)
    else:
        raise ValueError(f"Aggregation method {method} not supported")


def model_to_tensor(model_params: OrderedDict) -> torch.Tensor:
    """
    Convert model parameters to a single tensor for efficient transfer.
    
    Args:
        model_params: Model parameters (state_dict)
    
    Returns:
        Flattened tensor of all parameters
    """
    # Extract all parameters and flatten them
    tensors = []
    for param in model_params.values():
        tensors.append(param.view(-1))
    
    # Concatenate all tensors
    return torch.cat(tensors)


def tensor_to_model(
    tensor: torch.Tensor,
    model_template: OrderedDict
) -> OrderedDict:
    """
    Convert a flattened tensor back to model parameters.
    
    Args:
        tensor: Flattened tensor of all parameters
        model_template: Template model parameters with correct shapes
    
    Returns:
        Model parameters (state_dict)
    """
    model_params = copy.deepcopy(model_template)
    
    # Calculate the number of elements in each parameter
    sizes = []
    for param in model_template.values():
        sizes.append(param.numel())
    
    # Split the tensor and reshape each part
    start_idx = 0
    for i, (key, param) in enumerate(model_params.items()):
        end_idx = start_idx + sizes[i]
        model_params[key] = tensor[start_idx:end_idx].reshape(param.shape)
        start_idx = end_idx
    
    return model_params


def compress_model(
    model_params: OrderedDict,
    compression_ratio: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, torch.Size]]:
    """
    Compress model parameters for efficient transfer.
    
    Args:
        model_params: Model parameters (state_dict)
        compression_ratio: Ratio of parameters to keep (0.1 = 10%)
    
    Returns:
        Tuple of (compressed tensor, shapes dictionary)
    """
    # Convert model to tensor
    tensor = model_to_tensor(model_params)
    
    # Store shapes for reconstruction
    shapes = {key: param.shape for key, param in model_params.items()}
    
    # Calculate number of parameters to keep
    num_params = tensor.numel()
    k = int(num_params * compression_ratio)
    
    # Get top-k values by magnitude
    _, indices = torch.topk(tensor.abs(), k)
    
    # Create sparse tensor (indices and values)
    values = tensor[indices]
    
    # Return compressed representation
    return (indices, values, num_params), shapes


def decompress_model(
    compressed_data: Tuple[torch.Tensor, torch.Tensor, int],
    shapes: Dict[str, torch.Size],
    model_template: OrderedDict
) -> OrderedDict:
    """
    Decompress model parameters.
    
    Args:
        compressed_data: Tuple of (indices, values, total_size)
        shapes: Dictionary of parameter shapes
        model_template: Template model parameters
    
    Returns:
        Decompressed model parameters (state_dict)
    """
    indices, values, total_size = compressed_data
    
    # Create full tensor with zeros
    tensor = torch.zeros(total_size, device=values.device)
    
    # Fill in the values at the specified indices
    tensor[indices] = values
    
    # Convert back to model parameters
    return tensor_to_model(tensor, model_template) 