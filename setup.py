from setuptools import setup, find_packages

setup(
    name="fl-for-dr",
    version="0.1.0",
    packages=["src", "src.coordinator", "src.robot", "src.models", "src.utils", "src.communication"],
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "requests>=2.26.0",
        "flask>=2.0.0",
        "cryptography>=36.0.0",
        "tensorboard>=2.7.0",
        "pillow>=8.3.0",
        "scikit-learn>=1.0.0",
        "pytest>=6.2.5",
        "flwr>=1.0.0",
        "paho-mqtt>=1.6.0",
        "colorama>=0.4.4",
    ],
    python_requires=">=3.8",
) 