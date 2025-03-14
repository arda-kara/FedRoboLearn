# FL-for-DR Architecture

This document describes the architecture of the Federated Learning for Decentralized Robots (FL-for-DR) system.

## Overview

FL-for-DR enables a set of physical or simulated robots to jointly learn a model while keeping their local data private. Instead of sharing raw data, robots only exchange model parameters or gradients, addressing privacy and bandwidth constraints in multi-robot learning scenarios.

## System Components

The system consists of three main components:

1. **Robot Local Module**: Handles local data collection, preprocessing, and model training on each robot.
2. **Federated Coordinator**: Aggregates model updates from all robots and computes a global model.
3. **Communication Layer**: Manages secure and efficient exchange of model parameters between robots and coordinator.

## Architecture Diagram

```
                                 ┌─────────────────┐
                                 │                 │
                                 │   Coordinator   │
                                 │                 │
                                 └────────┬────────┘
                                          │
                                          │ Global Model
                                          │ Parameters
                                          │
                 ┌─────────────┬──────────┴───────────┬─────────────┐
                 │             │                      │             │
         ┌───────▼───────┐    ┌▼───────────────┐    ┌─▼─────────────┐
         │               │    │                │    │               │
         │   Robot 1     │    │    Robot 2     │    │    Robot 3    │
         │               │    │                │    │               │
         └───────┬───────┘    └────────┬───────┘    └───────┬───────┘
                 │                     │                    │
         ┌───────▼───────┐    ┌────────▼───────┐    ┌───────▼───────┐
         │  Local Data 1  │    │  Local Data 2  │    │  Local Data 3 │
         └───────────────┘    └────────────────┘    └───────────────┘
```

## Federated Learning Process

The federated learning process follows these steps:

1. **Initialization**:
   - The coordinator initializes a global model.
   - The global model is distributed to all robots.

2. **Local Training**:
   - Each robot trains the model on its local dataset.
   - Local updates (model parameters or gradients) are computed.

3. **Model Update Transfer**:
   - Robots send their local updates to the coordinator.
   - Only model parameters are transferred, not raw data.

4. **Aggregation**:
   - The coordinator aggregates the updates using a federated learning algorithm (e.g., FedAvg).
   - A new global model is computed.

5. **Global Model Distribution**:
   - The coordinator sends the new global model back to all robots.

6. **Iteration**:
   - Steps 2-5 are repeated for multiple rounds until convergence.

## Communication Protocols

The system supports two communication protocols:

1. **HTTP**: REST-based communication using HTTP/HTTPS.
2. **MQTT**: Lightweight publish-subscribe protocol suitable for resource-constrained devices.

## Security and Privacy

- **Data Privacy**: Raw data never leaves the robots.
- **Secure Communication**: TLS encryption for parameter exchange.
- **Model Security**: Parameters can be compressed or quantized to reduce bandwidth.

## Implementation Details

The system is implemented in Python using PyTorch for model training and inference. The main components are:

- **Robot Client**: Handles local training and communication with the coordinator.
- **Coordinator**: Manages the federated learning process and aggregates model updates.
- **Models**: Neural network architectures for various tasks.
- **Communication**: HTTP and MQTT implementations for parameter exchange.
- **Utils**: Configuration, logging, and data handling utilities. 