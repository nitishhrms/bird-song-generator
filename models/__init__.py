"""
Bird Song Generator - Model Implementations
"""

from .gan import WaveGANGenerator, WaveGANDiscriminator
from .diffusion import DiffusionModel
from .vae import VAE

__all__ = [
    'WaveGANGenerator',
    'WaveGANDiscriminator',
    'DiffusionModel',
    'VAE'
]
