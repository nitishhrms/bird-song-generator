"""
Dataset Classes for Bird Song Audio

Supports both raw audio (WaveGAN) and spectrogram (Diffusion/VAE) formats.
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from .audio import (
    load_audio, 
    audio_to_spectrogram, 
    normalize_spectrogram,
    pad_or_trim_audio,
    augment_audio
)


class BirdSongDataset(Dataset):
    """
    Dataset for bird song audio files
    
    Supports two modes:
    1. 'waveform': Returns raw audio waveforms (for WaveGAN)
    2. 'spectrogram': Returns mel spectrograms (for Diffusion/VAE)
    
    Args:
        data_dir: Directory containing audio files
        mode: 'waveform' or 'spectrogram'
        sr: Sample rate
        duration: Audio duration in seconds
        audio_length: Target audio length in samples (for waveform mode)
        n_mels: Number of mel bands (for spectrogram mode)
        spec_shape: Target spectrogram shape (height, width) for spectrogram mode
        augment: Whether to apply data augmentation
        cache_spectrograms: Cache spectrograms in memory for faster training
    """
    def __init__(
        self,
        data_dir,
        mode='waveform',
        sr=22050,
        duration=None,
        audio_length=16384,
        n_mels=128,
        spec_shape=(128, 128),
        augment=False,
        cache_spectrograms=False
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sr = sr
        self.duration = duration
        self.audio_length = audio_length
        self.n_mels = n_mels
        self.spec_shape = spec_shape
        self.augment = augment
        self.cache_spectrograms = cache_spectrograms
        
        # Find all audio files
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            self.audio_files.extend(list(self.data_dir.glob(f'**/{ext}')))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {data_dir}")
        
        print(f"Found {len(self.audio_files)} audio files in {data_dir}")
        print(f"Mode: {mode}")
        
        # Cache for spectrograms
        self.spectrogram_cache = {} if cache_spectrograms else None
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, _ = load_audio(audio_path, sr=self.sr, duration=self.duration)
        
        # Apply augmentation
        if self.augment:
            pitch_shift = np.random.randint(-2, 3)
            time_stretch = np.random.uniform(0.9, 1.1)
            audio = augment_audio(audio, self.sr, pitch_shift, time_stretch, add_noise=True)
        
        if self.mode == 'waveform':
            # Return raw audio waveform
            audio = pad_or_trim_audio(audio, self.audio_length)
            audio = torch.FloatTensor(audio).unsqueeze(0)  # Add channel dimension
            return audio
        
        elif self.mode == 'spectrogram':
            # Check cache first
            if self.cache_spectrograms and idx in self.spectrogram_cache:
                return self.spectrogram_cache[idx]
            
            # Convert to spectrogram
            spec = audio_to_spectrogram(audio, self.sr, n_mels=self.n_mels)
            
            # Resize to target shape
            spec = self._resize_spectrogram(spec, self.spec_shape)
            
            # Normalize to [-1, 1]
            spec = normalize_spectrogram(spec, method='minmax')
            
            # Convert to tensor
            spec = torch.FloatTensor(spec).unsqueeze(0)  # Add channel dimension
            
            # Cache if enabled
            if self.cache_spectrograms:
                self.spectrogram_cache[idx] = spec
            
            return spec
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _resize_spectrogram(self, spec, target_shape):
        """Resize spectrogram to target shape"""
        import cv2
        
        if spec.shape == target_shape:
            return spec
        
        # Use OpenCV for resizing (or fallback to numpy)
        try:
            spec_resized = cv2.resize(spec, (target_shape[1], target_shape[0]))
        except:
            # Fallback: simple interpolation
            from scipy.ndimage import zoom
            zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
            spec_resized = zoom(spec, zoom_factors, order=1)
        
        return spec_resized


class RandomNoiseDataset(Dataset):
    """
    Dataset that generates random noise (for testing)
    
    Args:
        size: Number of samples
        shape: Shape of each sample (e.g., (1, 128, 128) for spectrograms)
    """
    def __init__(self, size=1000, shape=(1, 128, 128)):
        self.size = size
        self.shape = shape
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(*self.shape)


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Create a DataLoader from a dataset
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )


if __name__ == "__main__":
    # Test dataset
    print("Testing BirdSongDataset...")
    
    # Create dummy data directory
    data_dir = Path("data/bird_songs")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if any audio files exist
    if len(list(data_dir.glob("*.wav"))) == 0:
        print(f"\nNo audio files found in {data_dir}")
        print("Please add bird song audio files (.wav, .mp3, .flac, .ogg) to the data directory")
        print("\nYou can download bird songs from:")
        print("  - Xeno-canto: https://www.xeno-canto.org/")
        print("  - Kaggle bird song datasets")
        print("  - Cornell Lab of Ornithology: https://www.birds.cornell.edu/")
    else:
        # Test waveform mode
        print("\n1. Testing waveform mode (for WaveGAN)...")
        dataset_wave = BirdSongDataset(data_dir, mode='waveform', audio_length=16384)
        sample = dataset_wave[0]
        print(f"   Waveform shape: {sample.shape}")
        
        # Test spectrogram mode
        print("\n2. Testing spectrogram mode (for Diffusion/VAE)...")
        dataset_spec = BirdSongDataset(data_dir, mode='spectrogram', spec_shape=(128, 128))
        sample = dataset_spec[0]
        print(f"   Spectrogram shape: {sample.shape}")
        
        # Test dataloader
        print("\n3. Testing DataLoader...")
        dataloader = create_dataloader(dataset_wave, batch_size=4, num_workers=0)
        batch = next(iter(dataloader))
        print(f"   Batch shape: {batch.shape}")
    
    print("\nDataset module ready!")
