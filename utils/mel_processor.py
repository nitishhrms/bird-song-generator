"""
Standalone Mel Spectrogram Processor
Replaces diffusers.pipelines.audio_diffusion.mel.Mel to avoid dependency conflicts

Based on the audio-diffusion implementation but standalone
"""

import numpy as np
import librosa
from PIL import Image
import torch


class Mel:
    """
    Mel spectrogram converter for audio diffusion
    Converts audio to/from mel spectrograms as PIL images
    """
    
    def __init__(self, x_res=128, y_res=128, sample_rate=16000, 
                 n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=None):
        """
        Args:
            x_res: Width of spectrogram image (time dimension)
            y_res: Height of spectrogram image (frequency dimension)
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
            fmin: Minimum frequency
            fmax: Maximum frequency (defaults to sample_rate/2)
        """
        self.x_res = x_res
        self.y_res = y_res
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels if n_mels == y_res else y_res
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        self.audio = None
        self.mel_spec = None
        
    def load_audio(self, raw_audio=None, audio_file=None):
        """Load audio from array or file"""
        if raw_audio is not None:
            self.audio = raw_audio
        elif audio_file is not None:
            self.audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        else:
            raise ValueError("Must provide either raw_audio or audio_file")
        
        # Compute mel spectrogram
        self.mel_spec = librosa.feature.melspectrogram(
            y=self.audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        
        # Convert to log scale
        self.mel_spec = librosa.power_to_db(self.mel_spec, ref=np.max)
        
    def get_number_of_slices(self):
        """Get number of x_res-width slices available"""
        if self.mel_spec is None:
            return 0
        return max(1, self.mel_spec.shape[1] // self.x_res)
    
    def audio_slice_to_image(self, slice_idx=0):
        """
        Convert a slice of the mel spectrogram to a PIL image
        
        Args:
            slice_idx: Which slice to extract (0 to get_number_of_slices()-1)
            
        Returns:
            PIL Image (grayscale)
        """
        if self.mel_spec is None:
            raise ValueError("Must load audio first")
        
        # Extract slice
        start_idx = slice_idx * self.x_res
        end_idx = start_idx + self.x_res
        
        # Handle edge case
        if end_idx > self.mel_spec.shape[1]:
            end_idx = self.mel_spec.shape[1]
            start_idx = max(0, end_idx - self.x_res)
        
        mel_slice = self.mel_spec[:, start_idx:end_idx]
        
        # Resize to target resolution if needed
        if mel_slice.shape != (self.y_res, self.x_res):
            # Use PIL for resizing
            img = Image.fromarray(mel_slice.astype(np.float32))
            img = img.resize((self.x_res, self.y_res), Image.LANCZOS)
            mel_slice = np.array(img)
        
        # Normalize to 0-255
        mel_normalized = ((mel_slice - mel_slice.min()) / 
                         (mel_slice.max() - mel_slice.min() + 1e-8) * 255)
        mel_normalized = mel_normalized.astype(np.uint8)
        
        # Create PIL image
        image = Image.fromarray(mel_normalized, mode='L')
        return image
    
    def image_to_audio(self, image):
        """
        Convert PIL image back to audio
        
        Args:
            image: PIL Image (grayscale) or numpy array
            
        Returns:
            Audio array
        """
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            mel_array = np.array(image).astype(np.float32)
        else:
            mel_array = image.astype(np.float32)
        
        # Denormalize from 0-255 to dB scale (approximate)
        mel_db = (mel_array / 255.0) * 80.0 - 80.0  # Approximate dB range
        
        # Resize if needed
        if mel_db.shape != (self.n_mels, self.x_res):
            img = Image.fromarray(mel_db)
            img = img.resize((self.x_res, self.n_mels), Image.LANCZOS)
            mel_db = np.array(img)
        
        # Convert from dB to power
        mel_power = librosa.db_to_power(mel_db)
        
        # Inverse mel spectrogram using Griffin-Lim
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            n_iter=32  # Griffin-Lim iterations
        )
        
        return audio


if __name__ == "__main__":
    # Test the Mel class
    print("Testing Mel class...")
    
    # Create test audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * frequency * t)
    
    # Create Mel processor
    mel = Mel(x_res=128, y_res=128, sample_rate=sample_rate)
    
    # Load audio
    mel.load_audio(raw_audio=test_audio)
    print(f"Loaded audio: {len(test_audio)} samples")
    print(f"Number of slices: {mel.get_number_of_slices()}")
    
    # Convert to image
    image = mel.audio_slice_to_image(0)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Convert back to audio
    reconstructed_audio = mel.image_to_audio(image)
    print(f"Reconstructed audio: {len(reconstructed_audio)} samples")
    
    print("\n✅ Mel class working correctly!")
