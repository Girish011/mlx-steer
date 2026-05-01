"""MLX Steer - Activation steering for local LLMs on Apple Silicon."""

__version__ = "0.1.0"

from mlx_steer.monitor.chi import CHIMonitor
from mlx_steer.steering.compute_vectors import compute_steering_vectors
from mlx_steer.steering.contrastive_pairs import get_all_pairs
from mlx_steer.steering.injector import SteeringEngine

__all__ = [
    "CHIMonitor",
    "SteeringEngine",
    "compute_steering_vectors",
    "get_all_pairs",
]