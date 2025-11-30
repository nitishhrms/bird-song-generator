"""
Training Script for WaveGAN

Trains WaveGAN model on bird song audio using raw waveforms.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.gan import WaveGANGenerator, WaveGANDiscriminator, gradient_penalty
from utils.dataset import BirdSongDataset, create_dataloader
from utils.training import Trainer, ExperimentLogger, get_optimizer, set_seed, count_parameters
from utils.audio import save_audio

def train_wavegan(args):
    """Train WaveGAN model"""
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment logger
    logger = ExperimentLogger('wavegan', log_dir='experiments')
    logger.log_config(vars(args))
    experiment_dir = logger.get_experiment_dir()
    
    # Create models
    print("\nInitializing models...")
    generator = WaveGANGenerator(
        latent_dim=args.latent_dim,
        output_length=args.audio_length,
        channels=1,
        model_size=args.model_size
    ).to(device)
    
    discriminator = WaveGANDiscriminator(
        input_length=args.audio_length,
        channels=1,
        model_size=args.model_size,
        use_phase_shuffle=True
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    
    # Optimizers
    optimizer_g = get_optimizer(generator, 'adam', lr=args.lr_g, betas=(0.5, 0.9))
    optimizer_d = get_optimizer(discriminator, 'adam', lr=args.lr_d, betas=(0.5, 0.9))
    
    # Dataset and dataloader
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = BirdSongDataset(
        data_dir=args.data_dir,
        mode='waveform',
        sr=args.sample_rate,
        audio_length=args.audio_length,
        augment=args.augment
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Trainer
    trainer = Trainer(generator, device, checkpoint_dir=experiment_dir / 'checkpoints')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_g = 0
        epoch_loss_d = 0
        
        for batch_idx, real_audio in enumerate(dataloader):
            real_audio = real_audio.to(device)
            batch_size = real_audio.size(0)
            
            # Train Discriminator
            for _ in range(args.n_critic):
                optimizer_d.zero_grad()
                
                # Generate fake audio
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake_audio = generator(z).detach()
                
                # Discriminator scores
                real_score = discriminator(real_audio)
                fake_score = discriminator(fake_audio)
                
                # Wasserstein loss with gradient penalty
                gp = gradient_penalty(discriminator, real_audio, fake_audio, device)
                loss_d = -real_score.mean() + fake_score.mean() + args.lambda_gp * gp
                
                loss_d.backward()
                optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_audio = generator(z)
            fake_score = discriminator(fake_audio)
            
            loss_g = -fake_score.mean()
            
            loss_g.backward()
            optimizer_g.step()
            
            # Accumulate losses
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            
            # Log to TensorBoard
            trainer.global_step += 1
            if batch_idx % args.log_interval == 0:
                trainer.log_scalar('train/loss_g', loss_g.item())
                trainer.log_scalar('train/loss_d', loss_d.item())
                trainer.log_scalar('train/real_score', real_score.mean().item())
                trainer.log_scalar('train/fake_score', fake_score.mean().item())
                trainer.log_scalar('train/gradient_penalty', gp.item())
        
        # Epoch statistics
        avg_loss_g = epoch_loss_g / len(dataloader)
        avg_loss_d = epoch_loss_d / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss_G: {avg_loss_g:.4f} | Loss_D: {avg_loss_d:.4f}")
        
        # Log metrics
        logger.log_metrics(epoch, {
            'loss_g': avg_loss_g,
            'loss_d': avg_loss_d
        })
        
        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(4, args.latent_dim, device=device)
                samples = generator(z)
                
                # Save audio samples
                sample_dir = experiment_dir / 'samples'
                sample_dir.mkdir(exist_ok=True)
                
                for i, sample in enumerate(samples):
                    save_path = sample_dir / f'epoch_{epoch+1}_sample_{i}.wav'
                    save_audio(sample, save_path, sr=args.sample_rate)
                
                # Log to TensorBoard
                trainer.log_audio('samples/generated', samples[0], sample_rate=args.sample_rate)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            trainer.save_checkpoint(
                f'wavegan_epoch_{epoch+1}.pt',
                discriminator_state_dict=discriminator.state_dict(),
                optimizer_g_state_dict=optimizer_g.state_dict(),
                optimizer_d_state_dict=optimizer_d.state_dict()
            )
    
    print("\nTraining completed!")
    print(f"Results saved to: {experiment_dir}")
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train WaveGAN for bird song generation')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/bird_songs',
                        help='Path to bird song audio directory')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate')
    parser.add_argument('--audio_length', type=int, default=16384,
                        help='Audio length in samples')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension')
    parser.add_argument('--model_size', type=int, default=64,
                        help='Base model size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                        help='Discriminator learning rate')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty weight')
    
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
    
    train_wavegan(args)
