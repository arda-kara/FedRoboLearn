# Federated Learning for Decentralized Robots (FL-for-DR)

A framework for enabling multiple robots to collaboratively train machine learning models without sharing raw data.

## Project Overview

FL-for-DR enables a set of physical or simulated robots to jointly learn a model (for perception, navigation, or manipulation) while keeping their local data private. Instead of sharing raw data, robots only exchange model parameters or gradients, addressing privacy and bandwidth constraints in multi-robot learning scenarios.

### Key Features

- **Privacy Preservation**: Robots keep their raw data local, sharing only model updates
- **Bandwidth Efficiency**: Minimizes communication overhead in distributed robotic systems
- **Collaborative Learning**: Robots benefit from each other's experiences without direct data sharing
- **Scalable Architecture**: Supports multiple robots with different data distributions
- **Fault Tolerance**: System continues functioning even if some robots disconnect

## Architecture

The system consists of three main components:

1. **Robot Local Module**: Handles local data collection, preprocessing, and model training on each robot.
2. **Federated Coordinator**: Aggregates model updates from all robots and computes a global model.
3. **Communication Layer**: Manages secure and efficient exchange of model parameters between robots and coordinator.

![Architecture Diagram](docs/architecture.png)

## Performance Results

Our implementation has been tested on image classification tasks using the CIFAR-10 dataset. In a simulation with 3 robots over 10 federated learning rounds, we achieved:

- **Final Validation Accuracy**: 65.91%
- **Training Accuracy**: 68.62%
- **Convergence**: Steady improvement from initial ~15% to final ~66% accuracy
- **Generalization**: Small gap between training and validation accuracy

### Learning Progress

| Round | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|---------------------|---------------|-----------------|
| 1     | 39.18%           | 49.80%              | 1.658         | 1.372           |
| 3     | 59.13%           | 62.76%              | 1.138         | 1.035           |
| 5     | 66.20%           | 64.92%              | 0.951         | 0.985           |
| 7     | 68.24%           | 65.91%              | 0.896         | 0.961           |
| 10    | 68.62%           | 65.91%              | 0.889         | 0.957           |

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Additional requirements in `requirements.txt`

### Installation

#### Option 1: Using the installation script

```bash
# Clone the repository
git clone https://github.com/yourusername/FL-for-DR.git
cd FL-for-DR

# Run the installation script
python install.py

# For development mode (editable installation)
python install.py --dev
```

#### Option 2: Manual installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FL-for-DR.git
cd FL-for-DR

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Create necessary directories
mkdir -p logs data/robot1 data/robot2 data/robot3
```

### Testing the Installation

```bash
# Test if the installation is working correctly
python test_installation.py
```

### Running a Simulation

#### Option 1: Using the run script (recommended)

```bash
# Run a simulation with 3 robots for 10 rounds
python run.py --num-robots 3 --rounds 10
```

#### Option 2: Using the example script directly

```bash
# Run a simulation with 3 robots for 10 rounds
python examples/run_simulation.py --num-robots 3 --rounds 10
```

#### Option 3: Manual startup

Start the coordinator and robots manually in separate terminals:

1. Start the coordinator:
```bash
python src/coordinator/main.py
```

2. Start multiple robot clients:
```bash
python src/robot/main.py --id 1 --data-path data/robot1
python src/robot/main.py --id 2 --data-path data/robot2
# Add more robots as needed
```

## Project Structure

```
FL-for-DR/
├── src/
│   ├── coordinator/       # Federated learning server/coordinator
│   ├── robot/             # Robot client implementation
│   ├── models/            # ML model definitions
│   ├── utils/             # Shared utilities
│   └── communication/     # Communication protocols
├── data/                  # Sample datasets for testing
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
├── examples/              # Example configurations and demos
├── install.py             # Installation script
├── run.py                 # Script to run the simulation
└── test_installation.py   # Script to test the installation
```

## Applications

While our implementation uses CIFAR-10 for demonstration, the same approach can be applied to various real-world robotics tasks:

1. **Object Recognition**: Robots identifying objects in different environments
2. **Human-Robot Interaction**: Learning to recognize gestures, faces, or emotions
3. **Navigation**: Identifying obstacles, landmarks, or navigable paths
4. **Anomaly Detection**: Identifying unusual or dangerous situations
5. **Task Learning**: Learning to perform specific tasks from visual input

## Future Work

- Implement additional federated learning algorithms beyond FedAvg
- Add privacy-preserving mechanisms like differential privacy
- Support heterogeneous robot architectures and capabilities
- Implement more sophisticated model architectures
- Evaluate performance on real-world robotic tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Federated Learning research community
- Robot Operating System (ROS) community 