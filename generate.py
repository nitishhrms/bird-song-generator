"""
Unified Generation Script

Generate bird songs using trained models (WaveGAN, Diffusion, or VAE).
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.gan import WaveGANGenerator
from models.diffusion import DiffusionModel
from models.vae import VAE
from utils.audio import save_audio, spectrogram_to_audio, plot_waveform, plot_spectrogram
import numpy as np


def generate_wavegan(checkpoint_path, num_samples=10, output_dir='generated', device='cpu'):
    """Generate samples using WaveGAN"""
    
    print(f"Loading WaveGAN from {checkpoint_path}...")
    
    # Load model
    generator = WaveGANGenerator(latent_dim=100, output_length=16384).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate from random latent vector
            z = torch.randn(1, 100, device=device)
            audio = generator(z)
            
            # Save audio
            audio_np = audio.cpu().numpy().squeeze()
            save_path = output_dir / f'wavegan_sample_{i+1}.wav'
            save_audio(audio_np, save_path, sr=22050)
            
            # Save waveform plot
            plot_path = output_dir / f'wavegan_sample_{i+1}_waveform.png'
            plot_waveform(audio_np, sr=22050, save_path=plot_path)
    
    print(f"Generated {num_samples} samples in {output_dir}")


def generate_diffusion(checkpoint_path, num_samples=10, output_dir='generated', device='cpu'):
    """Generate samples using Diffusion model"""
    
    print(f"Loading Diffusion model from {checkpoint_path}...")
    
    # Load model
    model = DiffusionModel(spectrogram_shape=(128, 128), timesteps=1000).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples (this may take a while)...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate spectrogram
            spec = model.sample(batch_size=1, device=device)
            spec_np = spec.cpu().numpy().squeeze()
            
            # Denormalize
            spec_np = (spec_np + 1) / 2 * 80 - 80
            
            # Convert to audio
            audio = spectrogram_to_audio(spec_np, sr=22050)
            
            # Save audio
            save_path = output_dir / f'diffusion_sample_{i+1}.wav'
            save_audio(audio, save_path, sr=22050)
            
            # Save spectrogram plot
            plot_path = output_dir / f'diffusion_sample_{i+1}_spec.png'
            plot_spectrogram(spec_np, sr=22050, save_path=plot_path)
    
    print(f"Generated {num_samples} samples in {output_dir}")


def generate_vae(checkpoint_path, num_samples=10, output_dir='generated', device='cpu'):
    """Generate samples using VAE"""
    
    print(f"Loading VAE from {checkpoint_path}...")
    
    # Load model
    model = VAE(input_shape=(1, 128, 128), latent_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate from random latent vector
            spec = model.sample(num_samples=1, device=device)
            spec_np = spec.cpu().numpy().squeeze()
            
            # Denormalize
            spec_np = (spec_np + 1) / 2 * 80 - 80
            
            # Convert to audio
            audio = spectrogram_to_audio(spec_np, sr=22050)
            
            # Save audio
            save_path = output_dir / f'vae_sample_{i+1}.wav'
            save_audio(audio, save_path, sr=22050)
            
            # Save spectrogram plot
            plot_path = output_dir / f'vae_sample_{i+1}_spec.png'
            plot_spectrogram(spec_np, sr=22050, save_path=plot_path)
    
    print(f"Generated {num_samples} samples in {output_dir}")


def interpolate_vae(checkpoint_path, num_steps=10, output_dir='generated', device='cpu'):
    """Generate interpolation between two random points in VAE latent space"""
    
    print(f"Loading VAE from {checkpoint_path}...")
    
    model = VAE(input_shape=(1, 128, 128), latent_dim=128).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = Path(output_dir) / 'interpolation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_steps} interpolation steps...")
    
    with torch.no_grad():
        # Sample two random latent vectors
        z1 = torch.randn(1, 128, device=device)
        z2 = torch.randn(1, 128, device=device)
        
        # Interpolate
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z = (1 - alpha) * z1 + alpha * z2
            
            # Generate
            spec = model.decode(z)
            spec_np = spec.cpu().numpy().squeeze()
            spec_np = (spec_np + 1) / 2 * 80 - 80
            audio = spectrogram_to_audio(spec_np, sr=22050)
            
            # Save
            save_path = output_dir / f'interpolation_step_{i+1:02d}.wav'
            save_audio(audio, save_path, sr=22050)
    
    print(f"Generated interpolation in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bird songs using trained models')
    
    parser.add_argument('--model', type=str, required=True, choices=['gan', 'diffusion', 'vae'],
                        help='Model type to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Output directory for generated samples')
    parser.add_argument('--interpolate', action='store_true',
                        help='Generate interpolation (VAE only)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.model == 'gan':
        generate_wavegan(args.checkpoint, args.num_samples, args.output_dir, args.device)
    elif args.model == 'diffusion':
        generate_diffusion(args.checkpoint, args.num_samples, args.output_dir, args.device)
    elif args.model == 'vae':
        if args.interpolate:
            interpolate_vae(args.checkpoint, args.num_samples, args.output_dir, args.device)
        else:
            generate_vae(args.checkpoint, args.num_samples, args.output_dir, args.device)
