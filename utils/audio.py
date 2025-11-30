"""
Audio Processing Utilities

Handles both raw audio waveforms and spectrogram conversions.
- WaveGAN uses raw audio directly (1D waveforms)
- Diffusion and VAE use spectrograms (2D time-frequency representations)
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def load_audio(file_path, sr=22050, duration=None, mono=True):
    """
    Load audio file
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        duration: Duration to load in seconds (None for full file)
        mono: Convert to mono
    
    Returns:
        audio: Audio waveform as numpy array
        sr: Sample rate
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration, mono=mono)
    return audio, sample_rate


def save_audio(audio, file_path, sr=22050):
    """
    Save audio to file
    
    Args:
        audio: Audio waveform (numpy array or torch tensor)
        file_path: Output file path
        sr: Sample rate
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # Ensure audio is in correct shape
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Normalize to [-1, 1]
    audio = np.clip(audio, -1, 1)
    
    sf.write(file_path, audio, sr)
    print(f"Saved audio to {file_path}")


def audio_to_spectrogram(audio, sr=22050, n_fft=1024, hop_length=512, n_mels=128):
    """
    Convert raw audio to mel spectrogram
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel bands
    
    Returns:
        mel_spec: Mel spectrogram in dB scale
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def spectrogram_to_audio(mel_spec_db, sr=22050, n_fft=1024, hop_length=512, n_iter=32):
    """
    Convert mel spectrogram back to audio using Griffin-Lim algorithm
    
    Args:
        mel_spec_db: Mel spectrogram in dB scale
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_iter: Number of Griffin-Lim iterations
    
    Returns:
        audio: Reconstructed audio waveform
    """
    # Convert from dB to power
    mel_spec = librosa.db_to_power(mel_spec_db)
    
    # Invert mel spectrogram to linear spectrogram
    spec = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=sr,
        n_fft=n_fft
    )
    
    # Use Griffin-Lim to reconstruct phase and audio
    audio = librosa.griffinlim(
        spec,
        n_iter=n_iter,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    return audio


def normalize_spectrogram(spec, method='minmax'):
    """
    Normalize spectrogram to [-1, 1] or [0, 1]
    
    Args:
        spec: Spectrogram
        method: 'minmax' or 'standard'
    
    Returns:
        Normalized spectrogram
    """
    if method == 'minmax':
        # Normalize to [-1, 1]
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = 2 * (spec - spec_min) / (spec_max - spec_min + 1e-8) - 1
    elif method == 'standard':
        # Standardize (zero mean, unit variance)
        spec_norm = (spec - spec.mean()) / (spec.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return spec_norm


def denormalize_spectrogram(spec_norm, original_min, original_max):
    """
    Denormalize spectrogram from [-1, 1] back to original range
    
    Args:
        spec_norm: Normalized spectrogram
        original_min: Original minimum value
        original_max: Original maximum value
    
    Returns:
        Denormalized spectrogram
    """
    spec = (spec_norm + 1) / 2 * (original_max - original_min) + original_min
    return spec


def plot_waveform(audio, sr=22050, title="Waveform", save_path=None):
    """
    Plot audio waveform
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        title: Plot title
        save_path: Path to save plot (optional)
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio)) / sr
    plt.plot(time, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved waveform plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram(spec, sr=22050, hop_length=512, title="Spectrogram", save_path=None):
    """
    Plot spectrogram
    
    Args:
        spec: Spectrogram (2D array)
        sr: Sample rate
        hop_length: Hop length used for STFT
        title: Plot title
        save_path: Path to save plot (optional)
    """
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    
    if spec.ndim > 2:
        spec = spec.squeeze()
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        spec,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(real_audio, fake_audio, sr=22050, save_path=None):
    """
    Plot side-by-side comparison of real and generated audio
    
    Args:
        real_audio: Real audio waveform
        fake_audio: Generated audio waveform
        sr: Sample rate
        save_path: Path to save plot (optional)
    """
    if isinstance(real_audio, torch.Tensor):
        real_audio = real_audio.cpu().numpy()
    if isinstance(fake_audio, torch.Tensor):
        fake_audio = fake_audio.cpu().numpy()
    
    real_audio = real_audio.squeeze()
    fake_audio = fake_audio.squeeze()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Real waveform
    time_real = np.arange(len(real_audio)) / sr
    axes[0, 0].plot(time_real, real_audio)
    axes[0, 0].set_title('Real Audio - Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Fake waveform
    time_fake = np.arange(len(fake_audio)) / sr
    axes[0, 1].plot(time_fake, fake_audio)
    axes[0, 1].set_title('Generated Audio - Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Real spectrogram
    real_spec = audio_to_spectrogram(real_audio, sr)
    img1 = axes[1, 0].imshow(real_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title('Real Audio - Spectrogram')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Frequency')
    plt.colorbar(img1, ax=axes[1, 0])
    
    # Fake spectrogram
    fake_spec = audio_to_spectrogram(fake_audio, sr)
    img2 = axes[1, 1].imshow(fake_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title('Generated Audio - Spectrogram')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Frequency')
    plt.colorbar(img2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def pad_or_trim_audio(audio, target_length):
    """
    Pad or trim audio to target length
    
    Args:
        audio: Audio waveform
        target_length: Target length in samples
    
    Returns:
        Processed audio of target length
    """
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Trim
        audio = audio[:target_length]
    
    return audio


def augment_audio(audio, sr=22050, pitch_shift_steps=0, time_stretch_rate=1.0, add_noise=False):
    """
    Apply data augmentation to audio
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        pitch_shift_steps: Number of semitones to shift pitch
        time_stretch_rate: Time stretching factor (1.0 = no change)
        add_noise: Whether to add random noise
    
    Returns:
        Augmented audio
    """
    # Pitch shift
    if pitch_shift_steps != 0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift_steps)
    
    # Time stretch
    if time_stretch_rate != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=time_stretch_rate)
    
    # Add noise
    if add_noise:
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
    
    return audio


if __name__ == "__main__":
    print("Audio utilities module")
    print("=" * 50)
    print("\nSupports two processing modes:")
    print("1. RAW AUDIO (WaveGAN): Direct 1D waveform processing")
    print("2. SPECTROGRAMS (Diffusion/VAE): 2D time-frequency representations")
    print("\nKey functions:")
    print("  - load_audio() / save_audio()")
    print("  - audio_to_spectrogram() / spectrogram_to_audio()")
    print("  - plot_waveform() / plot_spectrogram()")
    print("  - augment_audio()")
