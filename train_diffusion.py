"""
Training Script for Diffusion Model

Trains DDPM model on bird song spectrograms.
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.diffusion import DiffusionModel
from utils.dataset import BirdSongDataset, create_dataloader
from utils.training import Trainer, ExperimentLogger, get_optimizer, set_seed, count_parameters
from utils.audio import spectrogram_to_audio, save_audio, plot_spectrogram
import numpy as np


def train_diffusion(args):
    """Train Diffusion model"""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Experiment logger
    logger = ExperimentLogger('diffusion', log_dir='experiments')
    logger.log_config(vars(args))
    experiment_dir = logger.get_experiment_dir()
    
    # Create model
    print("\nInitializing Diffusion model...")
    model = DiffusionModel(
        spectrogram_shape=(args.spec_height, args.spec_width),
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = get_optimizer(model, 'adam', lr=args.lr)
    
    # Dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = BirdSongDataset(
        data_dir=args.data_dir,
        mode='spectrogram',
        sr=args.sample_rate,
        spec_shape=(args.spec_height, args.spec_width),
        n_mels=args.spec_height,
        augment=args.augment,
        cache_spectrograms=args.cache_spectrograms
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Trainer
    trainer = Trainer(model, device, checkpoint_dir=experiment_dir / 'checkpoints')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, spectrograms in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            
            # Forward pass
            loss = model(spectrograms)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Logging
            trainer.global_step += 1
            if batch_idx % args.log_interval == 0:
                trainer.log_scalar('train/loss', loss.item())
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")
        
        logger.log_metrics(epoch, {'loss': avg_loss})
        
        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(batch_size=4, device=device)
                
                # Save samples
                sample_dir = experiment_dir / 'samples'
                sample_dir.mkdir(exist_ok=True)
                
                for i, sample in enumerate(samples):
                    # Convert spectrogram to audio
                    spec = sample.cpu().numpy().squeeze()
                    
                    # Denormalize (assuming minmax normalization to [-1, 1])
                    spec = (spec + 1) / 2 * 80 - 80  # Approximate dB range
                    
                    # Convert to audio
                    audio = spectrogram_to_audio(spec, sr=args.sample_rate)
                    
                    # Save audio
                    save_path = sample_dir / f'epoch_{epoch+1}_sample_{i}.wav'
                    save_audio(audio, save_path, sr=args.sample_rate)
                    
                    # Save spectrogram plot
                    plot_path = sample_dir / f'epoch_{epoch+1}_sample_{i}_spec.png'
                    plot_spectrogram(spec, sr=args.sample_rate, save_path=plot_path)
                
                # Log to TensorBoard
                trainer.log_image('samples/spectrogram', samples[0], step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            trainer.save_checkpoint(
                f'diffusion_epoch_{epoch+1}.pt',
                optimizer_state_dict=optimizer.state_dict()
            )
    
    print("\nTraining completed!")
    print(f"Results saved to: {experiment_dir}")
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Diffusion model for bird song generation')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/bird_songs',
                        help='Path to bird song audio directory')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate')
    parser.add_argument('--spec_height', type=int, default=128,
                        help='Spectrogram height (mel bands)')
    parser.add_argument('--spec_width', type=int, default=128,
                        help='Spectrogram width (time frames)')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--cache_spectrograms', action='store_true',
                        help='Cache spectrograms in memory')
    
    # Model
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending beta value')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Sample generation interval (epochs)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Checkpoint save interval (epochs)')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    train_diffusion(args)
