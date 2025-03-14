"""
Data utilities for FL-for-DR.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Any, Optional, Union


def get_dataset(
    dataset_name: str,
    data_dir: str = "data",
    train: bool = True,
    transform: Optional[Any] = None
) -> Dataset:
    """
    Get a dataset by name.
    
    Args:
        dataset_name: Name of the dataset (e.g., cifar10, mnist)
        data_dir: Directory to store the dataset
        train: Whether to get the training set
        transform: Transforms to apply to the dataset
        
    Returns:
        Dataset object
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Default transforms if none provided
    if transform is None:
        if dataset_name.lower() in ["cifar10", "cifar100"]:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:  # MNIST and others
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
    
    # Get the dataset
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset


def create_iid_partition(
    dataset: Dataset,
    num_partitions: int
) -> List[List[int]]:
    """
    Create IID (Independent and Identically Distributed) partitions of a dataset.
    
    Args:
        dataset: Dataset to partition
        num_partitions: Number of partitions to create
        
    Returns:
        List of indices for each partition
    """
    num_items = len(dataset)
    indices = list(range(num_items))
    np.random.shuffle(indices)
    
    partition_size = num_items // num_partitions
    partitions = []
    
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size if i < num_partitions - 1 else num_items
        partitions.append(indices[start_idx:end_idx])
    
    return partitions


def create_non_iid_partition(
    dataset: Dataset,
    num_partitions: int,
    alpha: float = 0.5
) -> List[List[int]]:
    """
    Create non-IID partitions of a dataset using Dirichlet distribution.
    
    Args:
        dataset: Dataset to partition
        num_partitions: Number of partitions to create
        alpha: Concentration parameter for Dirichlet distribution
              (smaller alpha -> more non-IID)
        
    Returns:
        List of indices for each partition
    """
    # Get labels for all data points
    if isinstance(dataset, datasets.MNIST) or isinstance(dataset, datasets.CIFAR10) or \
       isinstance(dataset, datasets.CIFAR100) or isinstance(dataset, datasets.FashionMNIST):
        labels = np.array(dataset.targets)
    else:
        # Try to get labels from dataset
        try:
            labels = np.array([y for _, y in dataset])
        except:
            raise ValueError("Cannot extract labels from dataset")
    
    num_classes = len(np.unique(labels))
    num_items = len(dataset)
    indices = list(range(num_items))
    
    # Group indices by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Distribute data using Dirichlet distribution
    partitions = [[] for _ in range(num_partitions)]
    
    # For each class, distribute its indices according to Dirichlet distribution
    for class_idx, class_list in enumerate(class_indices):
        # Generate Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        
        # Calculate number of items per partition for this class
        num_per_partition = [int(p * len(class_list)) for p in proportions]
        
        # Adjust to ensure all items are distributed
        num_per_partition[-1] = len(class_list) - sum(num_per_partition[:-1])
        
        # Distribute indices
        class_list_copy = class_list.copy()
        np.random.shuffle(class_list_copy)
        
        start_idx = 0
        for partition_idx, num_items in enumerate(num_per_partition):
            partitions[partition_idx].extend(
                class_list_copy[start_idx:start_idx + num_items]
            )
            start_idx += num_items
    
    # Shuffle each partition
    for partition in partitions:
        np.random.shuffle(partition)
    
    return partitions


def get_data_loaders(
    dataset: Dataset,
    partition_indices: List[int],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for a specific partition of a dataset.
    
    Args:
        dataset: Dataset to create loader from
        partition_indices: Indices for the partition
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the partition
    """
    subset = Subset(dataset, partition_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def distribute_data(
    dataset_name: str,
    num_robots: int,
    data_dir: str = "data",
    distribution: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Dict[int, Dict[str, DataLoader]]:
    """
    Distribute data among robots.
    
    Args:
        dataset_name: Name of the dataset
        num_robots: Number of robots
        data_dir: Directory to store the dataset
        distribution: Data distribution type ('iid' or 'non_iid')
        alpha: Concentration parameter for non-IID distribution
        batch_size: Batch size for DataLoaders
        train_ratio: Ratio of training data to total data
        
    Returns:
        Dictionary mapping robot IDs to their train and test DataLoaders
    """
    # Get the dataset
    full_dataset = get_dataset(dataset_name, data_dir, train=True)
    
    # Split into train and validation sets for each robot
    num_items = len(full_dataset)
    train_size = int(train_ratio * num_items)
    val_size = num_items - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create partitions
    if distribution.lower() == "iid":
        train_partitions = create_iid_partition(train_dataset, num_robots)
        val_partitions = create_iid_partition(val_dataset, num_robots)
    elif distribution.lower() == "non_iid":
        train_partitions = create_non_iid_partition(train_dataset, num_robots, alpha)
        val_partitions = create_non_iid_partition(val_dataset, num_robots, alpha)
    else:
        raise ValueError(f"Distribution {distribution} not supported")
    
    # Create DataLoaders for each robot
    robot_data = {}
    for robot_id in range(num_robots):
        robot_data[robot_id] = {
            "train": get_data_loaders(
                full_dataset, train_partitions[robot_id], batch_size, shuffle=True
            ),
            "val": get_data_loaders(
                full_dataset, val_partitions[robot_id], batch_size, shuffle=False
            )
        }
    
    return robot_data 