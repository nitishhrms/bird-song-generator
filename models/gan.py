"""
WaveGAN Implementation for Bird Song Generation

Based on "Adversarial Audio Synthesis" (Donahue et al., 2018)
Generates raw audio waveforms directly using 1D convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseShuffleLayer(nn.Module):
    """
    Phase shuffle layer for discriminator.
    Randomly shifts audio samples to prevent checkerboard artifacts.
    """
    def __init__(self, shift_range=2):
        super().__init__()
        self.shift_range = shift_range
    
    def forward(self, x):
        if not self.training or self.shift_range == 0:
            return x
        
        batch_size, channels, length = x.shape
        shift = torch.randint(-self.shift_range, self.shift_range + 1, (batch_size,), device=x.device)
        
        # Apply shifts
        shifted = []
        for i in range(batch_size):
            if shift[i] == 0:
                shifted.append(x[i:i+1])
            elif shift[i] > 0:
                # Shift right
                pad = torch.zeros(1, channels, shift[i].item(), device=x.device)
                shifted.append(torch.cat([pad, x[i:i+1, :, :-shift[i]]], dim=2))
            else:
                # Shift left
                pad = torch.zeros(1, channels, -shift[i].item(), device=x.device)
                shifted.append(torch.cat([x[i:i+1, :, -shift[i]:], pad], dim=2))
        
        return torch.cat(shifted, dim=0)


class WaveGANGenerator(nn.Module):
    """
    WaveGAN Generator
    
    Generates raw audio waveforms from latent vectors using transposed 1D convolutions.
    
    Args:
        latent_dim: Dimension of input latent vector (default: 100)
        output_length: Length of output audio in samples (default: 16384)
        channels: Number of audio channels (default: 1 for mono)
        model_size: Base number of filters (default: 64)
    """
    def __init__(self, latent_dim=100, output_length=16384, channels=1, model_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.model_size = model_size
        
        # Calculate initial size
        # We'll upsample 5 times (2^5 = 32x)
        self.init_size = output_length // 32  # 512 for 16384
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, model_size * 16 * self.init_size)
        
        # Transposed convolutions for upsampling
        self.conv_layers = nn.ModuleList([
            # 512 -> 1024
            nn.ConvTranspose1d(model_size * 16, model_size * 8, kernel_size=25, stride=2, padding=12, output_padding=1),
            # 1024 -> 2048
            nn.ConvTranspose1d(model_size * 8, model_size * 4, kernel_size=25, stride=2, padding=12, output_padding=1),
            # 2048 -> 4096
            nn.ConvTranspose1d(model_size * 4, model_size * 2, kernel_size=25, stride=2, padding=12, output_padding=1),
            # 4096 -> 8192
            nn.ConvTranspose1d(model_size * 2, model_size, kernel_size=25, stride=2, padding=12, output_padding=1),
            # 8192 -> 16384
            nn.ConvTranspose1d(model_size, channels, kernel_size=25, stride=2, padding=12, output_padding=1),
        ])
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(model_size * 8),
            nn.BatchNorm1d(model_size * 4),
            nn.BatchNorm1d(model_size * 2),
            nn.BatchNorm1d(model_size),
        ])
    
    def forward(self, z):
        """
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
        
        Returns:
            Generated audio waveform of shape (batch_size, channels, output_length)
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, self.model_size * 16, self.init_size)
        
        # Apply transposed convolutions with ReLU and batch norm
        for i, conv in enumerate(self.conv_layers[:-1]):
            x = conv(x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
        
        # Final layer with tanh activation
        x = self.conv_layers[-1](x)
        x = torch.tanh(x)
        
        # Ensure correct output length
        if x.shape[2] != self.output_length:
            x = x[:, :, :self.output_length]
        
        return x


class WaveGANDiscriminator(nn.Module):
    """
    WaveGAN Discriminator
    
    Classifies real vs. generated audio using 1D convolutions with phase shuffle.
    
    Args:
        input_length: Length of input audio in samples (default: 16384)
        channels: Number of audio channels (default: 1 for mono)
        model_size: Base number of filters (default: 64)
        use_phase_shuffle: Whether to use phase shuffle (default: True)
    """
    def __init__(self, input_length=16384, channels=1, model_size=64, use_phase_shuffle=True):
        super().__init__()
        self.model_size = model_size
        self.use_phase_shuffle = use_phase_shuffle
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            # 16384 -> 8192
            nn.Conv1d(channels, model_size, kernel_size=25, stride=2, padding=12),
            # 8192 -> 4096
            nn.Conv1d(model_size, model_size * 2, kernel_size=25, stride=2, padding=12),
            # 4096 -> 2048
            nn.Conv1d(model_size * 2, model_size * 4, kernel_size=25, stride=2, padding=12),
            # 2048 -> 1024
            nn.Conv1d(model_size * 4, model_size * 8, kernel_size=25, stride=2, padding=12),
            # 1024 -> 512
            nn.Conv1d(model_size * 8, model_size * 16, kernel_size=25, stride=2, padding=12),
        ])
        
        # Phase shuffle layers
        if use_phase_shuffle:
            self.phase_shuffle = nn.ModuleList([
                PhaseShuffleLayer(2) for _ in range(len(self.conv_layers))
            ])
        
        # Calculate final size
        final_size = input_length // (2 ** len(self.conv_layers))
        
        # Final classification layer
        self.fc = nn.Linear(model_size * 16 * final_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: Audio waveform of shape (batch_size, channels, input_length)
        
        Returns:
            Discriminator score of shape (batch_size, 1)
        """
        # Apply convolutions with LeakyReLU and optional phase shuffle
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.leaky_relu(x, 0.2)
            if self.use_phase_shuffle:
                x = self.phase_shuffle[i](x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def gradient_penalty(discriminator, real_data, fake_data, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        discriminator: Discriminator model
        real_data: Real audio samples
        fake_data: Generated audio samples
        device: torch device
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    
    # Random weight for interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    
    # Interpolate between real and fake
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # Get discriminator output
    d_interpolates = discriminator(interpolates)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator = WaveGANGenerator(latent_dim=100, output_length=16384).to(device)
    discriminator = WaveGANDiscriminator(input_length=16384).to(device)
    
    # Test forward pass
    z = torch.randn(4, 100).to(device)
    fake_audio = generator(z)
    print(f"Generated audio shape: {fake_audio.shape}")
    
    real_audio = torch.randn(4, 1, 16384).to(device)
    real_score = discriminator(real_audio)
    fake_score = discriminator(fake_audio)
    print(f"Real score shape: {real_score.shape}")
    print(f"Fake score shape: {fake_score.shape}")
    
    # Test gradient penalty
    gp = gradient_penalty(discriminator, real_audio, fake_audio.detach(), device)
    print(f"Gradient penalty: {gp.item()}")
    
    print("\nWaveGAN models initialized successfully!")
