"""
Training Script for VAE

Trains VAE model on bird song spectrograms.
"""

import argparse
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.vae import VAE
from utils.dataset import BirdSongDataset, create_dataloader
from utils.training import Trainer, ExperimentLogger, get_optimizer, set_seed, count_parameters
from utils.audio import spectrogram_to_audio, save_audio, plot_spectrogram


def train_vae(args):
    """Train VAE model"""
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Experiment logger
    logger = ExperimentLogger('vae', log_dir='experiments')
    logger.log_config(vars(args))
    experiment_dir = logger.get_experiment_dir()
    
    # Create model
    print("\nInitializing VAE model...")
    model = VAE(
        input_shape=(1, args.spec_height, args.spec_width),
        latent_dim=args.latent_dim,
        beta=args.beta
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
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, spectrograms in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            
            # Forward pass
            recon_spectrograms, mu, logvar = model(spectrograms)
            
            # Compute loss
            loss_dict = model.loss_function(recon_spectrograms, spectrograms, mu, logvar)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['recon_loss'].item()
            epoch_kl_loss += loss_dict['kl_loss'].item()
            
            # Logging
            trainer.global_step += 1
            if batch_idx % args.log_interval == 0:
                trainer.log_scalar('train/loss', loss.item())
                trainer.log_scalar('train/recon_loss', loss_dict['recon_loss'].item())
                trainer.log_scalar('train/kl_loss', loss_dict['kl_loss'].item())
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | KL: {avg_kl_loss:.4f}")
        
        logger.log_metrics(epoch, {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        })
        
        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                # Generate from random latent vectors
                samples = model.sample(num_samples=4, device=device)
                
                # Save samples
                sample_dir = experiment_dir / 'samples'
                sample_dir.mkdir(exist_ok=True)
                
                for i, sample in enumerate(samples):
                    # Convert spectrogram to audio
                    spec = sample.cpu().numpy().squeeze()
                    
                    # Denormalize
                    spec = (spec + 1) / 2 * 80 - 80
                    
                    # Convert to audio
                    audio = spectrogram_to_audio(spec, sr=args.sample_rate)
                    
                    # Save audio
                    save_path = sample_dir / f'epoch_{epoch+1}_sample_{i}.wav'
                    save_audio(audio, save_path, sr=args.sample_rate)
                    
                    # Save spectrogram plot
                    plot_path = sample_dir / f'epoch_{epoch+1}_sample_{i}_spec.png'
                    plot_spectrogram(spec, sr=args.sample_rate, save_path=plot_path)
                
                # Also save reconstructions
                test_batch = next(iter(dataloader)).to(device)
                reconstructions = model.reconstruct(test_batch[:4])
                
                for i, recon in enumerate(reconstructions):
                    spec = recon.cpu().numpy().squeeze()
                    spec = (spec + 1) / 2 * 80 - 80
                    audio = spectrogram_to_audio(spec, sr=args.sample_rate)
                    
                    save_path = sample_dir / f'epoch_{epoch+1}_recon_{i}.wav'
                    save_audio(audio, save_path, sr=args.sample_rate)
                
                # Log to TensorBoard
                trainer.log_image('samples/generated', samples[0], step=epoch)
                trainer.log_image('samples/reconstruction', reconstructions[0], step=epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            trainer.save_checkpoint(
                f'vae_epoch_{epoch+1}.pt',
                optimizer_state_dict=optimizer.state_dict()
            )
    
    print("\nTraining completed!")
    print(f"Results saved to: {experiment_dir}")
    trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE for bird song generation')
    
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
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta value for beta-VAE (KL weight)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
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
    
    train_vae(args)
