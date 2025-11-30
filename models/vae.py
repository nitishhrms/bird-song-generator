"""
Variational Autoencoder (VAE) for Bird Song Generation

Learns a latent representation of bird song spectrograms
and generates new samples by sampling from the latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    VAE Encoder: Compresses spectrograms to latent representation
    
    Args:
        input_shape: Tuple of (channels, height, width)
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
    """
    def __init__(self, input_shape=(1, 128, 128), latent_dim=128, 
                 hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build encoder layers
        modules = []
        in_channels = input_shape[0]
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_out = self.encoder(dummy)
            self.flatten_size = dummy_out.view(1, -1).shape[1]
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input spectrograms of shape (batch, channels, height, width)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder: Reconstructs spectrograms from latent representation
    
    Args:
        output_shape: Tuple of (channels, height, width)
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions (reversed from encoder)
    """
    def __init__(self, output_shape=(1, 128, 128), latent_dim=128,
                 hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        # Calculate initial spatial size after upsampling
        self.init_h = output_shape[1] // (2 ** len(hidden_dims))
        self.init_w = output_shape[2] // (2 ** len(hidden_dims))
        
        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.init_h * self.init_w)
        
        # Build decoder layers
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer to output channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[-1], output_shape[0], kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        """
        Decode latent vector to spectrogram
        
        Args:
            z: Latent vectors of shape (batch, latent_dim)
        
        Returns:
            Reconstructed spectrograms
        """
        x = self.fc(z)
        x = x.view(-1, 256, self.init_h, self.init_w)
        x = self.decoder(x)
        x = self.final_layer(x)
        
        # Ensure correct output size
        if x.shape[2:] != self.output_shape[1:]:
            x = F.interpolate(x, size=self.output_shape[1:], mode='bilinear', align_corners=False)
        
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder for bird song spectrograms
    
    Args:
        input_shape: Tuple of (channels, height, width)
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence term (beta-VAE)
    """
    def __init__(self, input_shape=(1, 128, 128), latent_dim=128, beta=1.0):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder and decoder
        hidden_dims = [32, 64, 128, 256]
        self.encoder = Encoder(input_shape, latent_dim, hidden_dims)
        self.decoder = Decoder(input_shape, latent_dim, list(reversed(hidden_dims)))
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through VAE
        
        Args:
            x: Input spectrograms
        
        Returns:
            recon_x: Reconstructed spectrograms
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss: reconstruction + KL divergence
        
        Args:
            recon_x: Reconstructed spectrograms
            x: Original spectrograms
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Dictionary with total loss and individual components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    @torch.no_grad()
    def sample(self, num_samples=1, device='cpu'):
        """
        Generate new spectrograms by sampling from latent space
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
        
        Returns:
            Generated spectrograms
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input spectrograms
        
        Args:
            x: Input spectrograms
        
        Returns:
            Reconstructed spectrograms
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)
    
    @torch.no_grad()
    def encode(self, x):
        """
        Encode input to latent space
        
        Args:
            x: Input spectrograms
        
        Returns:
            Latent vectors (using mean, not sampled)
        """
        mu, _ = self.encoder(x)
        return mu
    
    @torch.no_grad()
    def decode(self, z):
        """
        Decode latent vectors to spectrograms
        
        Args:
            z: Latent vectors
        
        Returns:
            Decoded spectrograms
        """
        return self.decoder(z)


if __name__ == "__main__":
    # Test the VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    vae = VAE(input_shape=(1, 128, 128), latent_dim=128, beta=1.0).to(device)
    
    # Test forward pass
    x = torch.randn(4, 1, 128, 128).to(device)
    recon_x, mu, logvar = vae(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon_x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test loss
    loss_dict = vae.loss_function(recon_x, x, mu, logvar)
    print(f"\nLoss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test sampling
    samples = vae.sample(num_samples=2, device=device)
    print(f"\nGenerated samples shape: {samples.shape}")
    
    # Test encoding/decoding
    z = vae.encode(x)
    decoded = vae.decode(z)
    print(f"Encoded latent shape: {z.shape}")
    print(f"Decoded shape: {decoded.shape}")
    
    print("\nVAE model initialized successfully!")
