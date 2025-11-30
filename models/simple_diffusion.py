"""
SimpleDiffusion Model - Based on Colab Implementation
Matches the working architecture from the reference notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps"""
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


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SelfAttention(nn.Module):
    """Self-attention layer"""
    def __init__(self, channels, attn_chans=8, transpose=False):
        super().__init__()
        self.channels = channels
        self.attn_chans = attn_chans
        self.transpose = transpose
        
        # Use LayerNorm for transformer blocks
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        # x shape: (B, L, C) for transformer blocks
        b, l, c = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)
        out = self.proj(out)
        
        return out


class ResBlock(nn.Module):
    """Residual block with time embedding - using LayerNorm for stability"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.SiLU()
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=1, 
                 add_down=True, attn_chans=0):
        super().__init__()
        
        self.resnets = nn.ModuleList([
            ResBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        
        self.attentions = nn.ModuleList([
            SelfAttention(out_channels, attn_chans) if attn_chans > 0 else nn.Identity()
            for _ in range(num_layers)
        ])
        
        if add_down:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()
        
        self.saved = []
    
    def forward(self, x, t_emb):
        self.saved = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x)
            self.saved.append(x)
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, prev_out_channels, out_channels, time_emb_dim,
                 num_layers=1, add_up=True, attn_chans=0):
        super().__init__()
        
        # First resnet gets concatenated input
        first_in_channels = in_channels + prev_out_channels
        
        self.resnets = nn.ModuleList([
            ResBlock(
                first_in_channels if i == 0 else out_channels + prev_out_channels,
                out_channels, 
                time_emb_dim
            )
            for i in range(num_layers)
        ])
        
        self.attentions = nn.ModuleList([
            SelfAttention(out_channels, attn_chans) if attn_chans > 0 else nn.Identity()
            for _ in range(num_layers)
        ])
        
        if add_up:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x, t_emb, saved):
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = saved.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, t_emb)
            x = attn(x)
        x = self.upsample(x)
        return x


def lin(ni, nf, norm=None):
    """Linear layer with optional normalization"""
    layers = [nn.Linear(ni, nf)]
    if norm:
        layers.append(norm(nf))
    layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def pre_conv(ni, nf, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True):
    """Convolution with normalization and activation"""
    layers = []
    if norm:
        layers.append(norm(ni))
    if act:
        layers.append(act())
    layers.append(nn.Conv2d(ni, nf, ks, stride=stride, padding=ks//2, bias=bias))
    return nn.Sequential(*layers)


class LearnEmbSS(nn.Module):
    """Learned embedding scale and shift"""
    def __init__(self, sz, ni):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(sz, ni))
        self.shift = nn.Parameter(torch.zeros(sz, ni))

    def forward(self, x):
        return x * self.scale + self.shift


def _mlp(ni, nh):
    """MLP block"""
    return nn.Sequential(
        nn.Linear(ni, nh),
        nn.GELU(),
        nn.LayerNorm(nh),
        nn.Linear(nh, ni)
    )


class EmbTransformerBlk(nn.Module):
    """Transformer block with time embedding"""
    def __init__(self, n_emb, ni, attn_chans=8):
        super().__init__()
        self.attn = SelfAttention(ni, attn_chans=attn_chans, transpose=False)
        self.mlp = _mlp(ni, ni * 4)
        self.nrm1 = nn.LayerNorm(ni)
        self.nrm2 = nn.LayerNorm(ni)
        self.emb_proj = nn.Linear(n_emb, ni * 2)

    def forward(self, x, t):
        emb = self.emb_proj(F.silu(t))[:, None]
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = x + self.attn(self.nrm1(x))
        x = x * (1 + scale) + shift
        return x + self.mlp(self.nrm2(x))


class SimpleDiffusion(nn.Module):
    """
    SimpleDiffusion model matching Colab implementation
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for grayscale)
        nfs: Channel multipliers for each level
        num_layers: Number of residual blocks per level
        attn_chans: Attention channels (0 to disable)
        attn_start: Level to start attention
        n_mids: Number of transformer blocks in middle
    """
    def __init__(self, in_channels=1, out_channels=1, nfs=(32, 64, 128, 256, 512),
                 num_layers=1, attn_chans=0, attn_start=1, n_mids=6):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)
        self.n_temb = nf = nfs[0]
        n_emb = nf * 4
        
        # Time embedding MLP
        self.emb_mlp = nn.Sequential(
            lin(self.n_temb, n_emb, norm=nn.BatchNorm1d),
            lin(n_emb, n_emb)
        )
        
        # Downsampling blocks
        self.downs = nn.ModuleList()
        n = len(nfs)
        for i in range(n):
            ni = nf
            nf = nfs[i]
            self.downs.append(DownBlock(
                ni, nf, n_emb,
                add_down=i != n - 1,
                num_layers=num_layers,
                attn_chans=0 if i < attn_start else attn_chans
            ))
        
        # Middle transformer blocks
        self.le = LearnEmbSS(64, nf)
        self.mids = nn.ModuleList([
            EmbTransformerBlk(n_emb, nf) for _ in range(n_mids)
        ])
        
        # Upsampling blocks
        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(n):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i + 1, len(nfs) - 1)]
            self.ups.append(UpBlock(
                ni, prev_nf, nf, n_emb,
                add_up=i != n - 1,
                num_layers=num_layers + 1,
                attn_chans=0 if i >= n - attn_start else attn_chans
            ))
        
        # Output convolution
        self.conv_out = pre_conv(nfs[0], out_channels, act=nn.SiLU, 
                                 norm=nn.BatchNorm2d, bias=False)
    
    def forward(self, x, t):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            t: Timesteps (B,)
        
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        temb = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(temb)
        
        # Initial convolution
        x = self.conv_in(x)
        saved = [x]
        
        # Downsampling
        for block in self.downs:
            x = block(x, emb)
        saved += [p for o in self.downs for p in o.saved]
        
        # Middle transformer blocks
        n, c, h, w = x.shape
        x = self.le(x.reshape(n, c, -1).transpose(1, 2))
        for block in self.mids:
            x = block(x, emb)
        x = x.transpose(1, 2).reshape(n, c, h, w)
        
        # Upsampling
        for block in self.ups:
            x = block(x, emb, saved)
        
        # Output
        return self.conv_out(x)


def init_ddpm(model):
    """Initialize DDPM model weights"""
    for o in model.downs:
        for p in o.resnets:
            # conv2 is Sequential[Conv2d, SiLU], get the Conv2d
            if hasattr(p, 'conv2') and len(p.conv2) > 0:
                p.conv2[0].weight.data.zero_()
    
    for o in model.ups:
        for p in o.resnets:
            # conv2 is Sequential[Conv2d, SiLU], get the Conv2d
            if hasattr(p, 'conv2') and len(p.conv2) > 0:
                p.conv2[0].weight.data.zero_()


if __name__ == "__main__":
    # Test the model
    model = SimpleDiffusion(
        in_channels=1,
        out_channels=1,
        nfs=(32, 64, 128, 256, 512),
        num_layers=1,
        attn_chans=0,
        n_mids=6
    )
    
    init_ddpm(model)
    
    # Test forward pass
    x = torch.randn(2, 1, 128, 128)
    t = torch.randint(0, 1000, (2,))
    
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("SimpleDiffusion model test passed!")
