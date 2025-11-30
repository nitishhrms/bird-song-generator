"""
Denoising Diffusion Probabilistic Model (DDPM) for Bird Song Generation

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
Adapted for audio spectrograms with U-Net architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize and compute Q, K, V
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, c, h * w).transpose(1, 2)
        k = k.view(b, c, h * w).transpose(1, 2)
        v = v.view(b, c, h * w).transpose(1, 2)
        
        # Compute attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)
        out = out.transpose(1, 2).view(b, c, h, w)
        out = self.proj(out)
        
        return x + out


class UNet(nn.Module):
    """Simplified U-Net architecture for diffusion model"""
    def __init__(self, in_channels=1, model_channels=32, out_channels=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, model_channels, time_emb_dim)
        self.down1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        self.enc2 = ResidualBlock(model_channels, model_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        self.enc3 = ResidualBlock(model_channels * 2, model_channels * 4, time_emb_dim)
        self.down3 = nn.Conv2d(model_channels * 4, model_channels * 4, 3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim),
            ResidualBlock(model_channels * 4, model_channels * 4, time_emb_dim),
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(model_channels * 4, model_channels * 4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(model_channels * 8, model_channels * 2, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(model_channels * 4, model_channels, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(model_channels * 2, model_channels, time_emb_dim)
        
        # Final
        self.final = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        x = self.down1(e1)
        
        e2 = self.enc2(x, t_emb)
        x = self.down2(e2)
        
        e3 = self.enc3(x, t_emb)
        x = self.down3(e3)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, t_emb)
        
        # Decoder with skip connections
        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, t_emb)
        
        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, t_emb)
        
        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, t_emb)
        
        # Final
        x = self.final(x)
        
        return x


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for audio spectrograms
    
    Args:
        spectrogram_shape: Tuple of (height, width) for spectrogram
        timesteps: Number of diffusion timesteps (default: 1000)
        beta_start: Starting beta value (default: 0.0001)
        beta_end: Ending beta value (default: 0.02)
    """
    def __init__(self, spectrogram_shape=(128, 128), timesteps=1000, 
                 beta_start=0.0001, beta_end=0.02):
        super().__init__()
        
        self.spectrogram_shape = spectrogram_shape
        self.timesteps = timesteps
        
        # U-Net model
        self.model = UNet(in_channels=1, out_channels=1)
        
        # Define beta schedule (linear)
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        
        # Pre-compute alpha values
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance',
                            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to x_start
        
        Args:
            x_start: Clean spectrograms
            t: Timesteps
            noise: Optional noise tensor
        
        Returns:
            Noisy spectrograms at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """
        Compute training loss
        
        Args:
            x_start: Clean spectrograms
            t: Timesteps
            noise: Optional noise tensor
        
        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Reverse diffusion: denoise one step
        
        Args:
            x: Noisy spectrogram at timestep t
            t: Current timestep
        
        Returns:
            Denoised spectrogram at timestep t-1
        """
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None]
        
        # Predict noise
        predicted_noise = self.model(x, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, batch_size=1, device='cpu'):
        """
        Generate spectrograms from random noise
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
        
        Returns:
            Generated spectrograms
        """
        shape = (batch_size, 1, *self.spectrogram_shape)
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x
    
    def forward(self, x, t=None):
        """
        Forward pass for training
        
        Args:
            x: Clean spectrograms
            t: Optional timesteps (randomly sampled if None)
        
        Returns:
            Loss value
        """
        if t is None:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        
        return self.p_losses(x, t)


if __name__ == "__main__":
    # Test the diffusion model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = DiffusionModel(spectrogram_shape=(128, 128), timesteps=1000).to(device)
    
    # Test forward pass (training)
    x = torch.randn(4, 1, 128, 128).to(device)
    loss = model(x)
    print(f"Training loss: {loss.item()}")
    
    # Test sampling (generation)
    samples = model.sample(batch_size=2, device=device)
    print(f"Generated samples shape: {samples.shape}")
    
    print("\nDiffusion model initialized successfully!")
