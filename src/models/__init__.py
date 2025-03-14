"""
Model modules for FL-for-DR.
"""
from src.models.models import SimpleCNN, MLP, ResNet18, get_model
from src.models.federated import (
    fedavg_aggregate, fedprox_aggregate, aggregate_models,
    model_to_tensor, tensor_to_model, compress_model, decompress_model
)

__all__ = [
    'SimpleCNN', 'MLP', 'ResNet18', 'get_model',
    'fedavg_aggregate', 'fedprox_aggregate', 'aggregate_models',
    'model_to_tensor', 'tensor_to_model', 'compress_model', 'decompress_model'
] 