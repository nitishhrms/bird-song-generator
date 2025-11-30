"""
Bird Song Generator - Improved Implementation
Based on working Colab notebook approach

Uses:
- diffusers Mel class for audio processing
- DDIM sampling (simpler than DDPM)
- Spectrograms as grayscale images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

# Check if diffusers audio_diffusion is available
try:
    from diffusers.pipelines.audio_diffusion.mel import Mel
    HAS_AUDIO_DIFFUSION = True
    print("Using diffusers Mel class")
except ImportError:
    # Use our standalone implementation
    from utils.mel_processor import Mel
    HAS_AUDIO_DIFFUSION = False
    print("Using standalone Mel class (diffusers audio_diffusion not available)")

from datasets import load_dataset
import torchaudio.transforms as AT
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader
import random


# ============================================================================
# Audio Processing with Mel Spectrogram
# ============================================================================

class MelProcessor:
    """Handles conversion between audio and mel spectrograms"""
    def __init__(self, x_res=128, y_res=128, sample_rate=16000):
        self.x_res = x_res
        self.y_res = y_res
        self.sample_rate = sample_rate
        self.mel = Mel(x_res, y_res, sample_rate)
    
    def audio_to_image(self, audio_array):
        """Convert audio array to PIL image"""
        audio_tensor = torch.tensor(audio_array).to(torch.float32)
        self.mel.load_audio(raw_audio=np.array(audio_tensor))
        num_slices = self.mel.get_number_of_slices()
        slice_idx = random.randint(0, max(0, num_slices-1))
        im = self.mel.audio_slice_to_image(slice_idx)
        return im
    
    def image_to_audio(self, image):
        """Convert PIL image back to audio"""
        return self.mel.image_to_audio(image)


# ============================================================================
# Dataset
# ============================================================================

class BirdSongImageDataset(Dataset):
    """Dataset that converts bird songs to spectrogram images"""
    def __init__(self, dataset, mel_processor, source_sr=32000, target_sr=16000, 
                 x_res=128, y_res=128):
        self.dataset = dataset
        self.mel_processor = mel_processor
        self.resampler = AT.Resample(source_sr, target_sr, dtype=torch.float32)
        self.x_res = x_res
        self.y_res = y_res
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get audio
        audio_array = self.dataset[idx]['audio']['array']
        
        # Resample
        audio_tensor = torch.tensor(audio_array).to(torch.float32)
        audio_tensor = self.resampler(audio_tensor)
        
        # Convert to mel spectrogram image
        self.mel_processor.mel.load_audio(raw_audio=np.array(audio_tensor))
        num_slices = self.mel_processor.mel.get_number_of_slices()
        slice_idx = random.randint(0, max(0, num_slices-1))
        im = self.mel_processor.mel.audio_slice_to_image(slice_idx)
        
        # Convert to tensor and normalize
        im = TF.to_tensor(im) - 0.5  # Range: [-0.5, 0.5]
        im = TF.resize(im, (self.x_res, self.y_res))
        
        return im


# ============================================================================
# Noise Schedule
# ============================================================================

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    """Create linear beta schedule"""
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def noisify(x, timesteps=1000):
    """Add noise to images for training"""
    device = x.device
    bs = x.shape[0]
    
    # Random timesteps
    t = torch.randint(0, timesteps, (bs,), device=device).long()
    
    # Noise schedule
    _, _, alphas_cumprod = make_beta_schedule(timesteps)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # Add noise
    noise = torch.randn_like(x)
    alpha_t = alphas_cumprod[t][:, None, None, None]
    x_noisy = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise
    
    return (x_noisy, t), noise


# ============================================================================
# DDIM Sampling
# ============================================================================

@torch.no_grad()
def ddim_sample(model, shape, steps=100, eta=1.0, clamp=1.0, device='cuda'):
    """DDIM sampling for generation"""
    model.eval()
    
    # Start from random noise
    x = torch.randn(shape, device=device)
    
    # Noise schedule
    timesteps = 1000
    _, _, alphas_cumprod = make_beta_schedule(timesteps)
    alphas_cumprod = alphas_cumprod.to(device)
    
    # Sample indices
    step_indices = torch.linspace(0, timesteps-1, steps).long()
    
    for i in tqdm(reversed(range(steps)), desc='Sampling'):
        t_idx = step_indices[i]
        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model((x, t))
        
        # DDIM step
        alpha_t = alphas_cumprod[t_idx]
        alpha_t_prev = alphas_cumprod[step_indices[i-1]] if i > 0 else torch.tensor(1.0)
        
        # Predict x0
        x_0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        if clamp:
            x_0_pred = x_0_pred.clamp(-clamp, clamp)
        
        # Compute variance
        sigma = eta * ((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)).sqrt()
        
        # Update x
        if i > 0:
            noise = torch.randn_like(x)
            x = alpha_t_prev.sqrt() * x_0_pred + (1 - alpha_t_prev - sigma**2).sqrt() * noise_pred + sigma * noise
        else:
            x = alpha_t_prev.sqrt() * x_0_pred
    
    return x


# ============================================================================
# Training
# ============================================================================

def train_simple_diffusion(
    data_dir='data/bird_songs',
    epochs=15,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    save_dir='experiments_simple',
    x_res=128,
    y_res=128
):
    """Train the diffusion model"""
    
    # Setup
    from pathlib import Path
    import glob
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Mel processor
    mel_processor = MelProcessor(x_res, y_res, sample_rate=16000)
    
    # Load audio files directly from directory (avoid datasets library cache issues)
    print("Loading audio files from directory...")
    
    audio_files = glob.glob(str(Path(data_dir) / "*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {data_dir}")
        print("Please run: python download_dataset.py --split train")
        return
    
    # Create simple dataset class for local files
    class LocalAudioDataset(Dataset):
        def __init__(self, audio_files, mel_processor, resampler, x_res, y_res):
            self.audio_files = audio_files
            self.mel_processor = mel_processor
            self.resampler = resampler
            self.x_res = x_res
            self.y_res = y_res
        
        def __len__(self):
            return len(self.audio_files)
        
        def __getitem__(self, idx):
            # Load audio
            import soundfile as sf
            audio, sr = sf.read(self.audio_files[idx])
            
            # Resample if needed
            if sr != self.mel_processor.sample_rate:
                audio_tensor = torch.tensor(audio).to(torch.float32)
                audio_tensor = self.resampler(audio_tensor)
                audio = np.array(audio_tensor)
            
            # Convert to mel spectrogram image
            self.mel_processor.mel.load_audio(raw_audio=audio)
            num_slices = self.mel_processor.mel.get_number_of_slices()
            slice_idx = random.randint(0, max(0, num_slices-1))
            im = self.mel_processor.mel.audio_slice_to_image(slice_idx)
            
            # Convert to tensor and normalize
            im = TF.to_tensor(im) - 0.5  # Range: [-0.5, 0.5]
            im = TF.resize(im, (self.x_res, self.y_res))
            
            return im
    
    # Create resampler
    source_sr = 22050  # Assuming your audio files are 22050 Hz
    target_sr = 16000
    resampler = AT.Resample(source_sr, target_sr, dtype=torch.float32)
    
    # Split into train/val
    random.shuffle(audio_files)
    split_idx = int(len(audio_files) * 0.9)
    train_files = audio_files[:split_idx]
    val_files = audio_files[split_idx:]
    
    # Create datasets
    train_ds = LocalAudioDataset(train_files, mel_processor, resampler, x_res, y_res)
    val_ds = LocalAudioDataset(val_files, mel_processor, resampler, x_res, y_res)
    
    # Dataloaders with noisify collate
    def collate_fn(batch):
        x = torch.stack(batch)
        return noisify(x)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn=collate_fn)  # num_workers=0 for Windows
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_fn)
    
    # Model - SimpleDiffusion matching Colab implementation
    from models.simple_diffusion import SimpleDiffusion, init_ddpm
    model = SimpleDiffusion(
        in_channels=1,
        out_channels=1,
        nfs=(32, 64, 128, 256, 512),
        num_layers=1,
        attn_chans=0,
        n_mids=6
    ).to(device)
    
    # Initialize weights
    init_ddpm(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for (x_noisy, t), noise_target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            x_noisy = x_noisy.to(device)
            t = t.to(device)
            noise_target = noise_target.to(device)
            
            # Forward - SimpleDiffusion takes (x, t) directly
            optimizer.zero_grad()
            
            # Get noise prediction
            noise_pred = model(x_noisy, t)
            loss = criterion(noise_pred, noise_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path / f'checkpoint_epoch_{epoch+1}.pt')
            
            # Generate samples
            print("Generating samples...")
            samples = ddim_sample(model, (4, 1, x_res, y_res), steps=100, device=device)
            samples = (samples + 0.5).clamp(0, 1)
            
            # Save samples
            for i, sample in enumerate(samples):
                im_array = (sample[0].cpu().numpy() * 255).astype(np.uint8)
                im = Image.fromarray(im_array, mode='L')
                im.save(save_path / f'sample_epoch_{epoch+1}_{i}.png')
                
                # Convert to audio
                audio = mel_processor.image_to_audio(im)
                import soundfile as sf
                sf.write(save_path / f'sample_epoch_{epoch+1}_{i}.wav', audio, 16000)
    
    print("Training complete!")
    return model, mel_processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_simple_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
