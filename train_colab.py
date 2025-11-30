"""
Bird Song Generator - Exact Colab Approach
Uses miniminiai library to match working Colab implementation
"""

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import optim
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import torchaudio.transforms as AT
from torch.utils.data import default_collate
from torchvision.transforms import functional as TF
from PIL import Image
import soundfile as sf
from pathlib import Path

# Import miniminiai
from miniminiai import *

# Import our standalone Mel class
from utils.mel_processor import Mel

# Settings
sample_rate_mel = 16000
x_res = 128
y_res = 128
sample_rate_dataset = 22050  # Our dataset sample rate

# Create Mel processor
mel = Mel(x_res, y_res, sample_rate_mel)

# Resampler
resampler = AT.Resample(sample_rate_dataset, sample_rate_mel, dtype=torch.float32)


# ============================================================================
# SimpleDiffusion Model Definition (from Colab)
# ============================================================================

class LearnEmbSS(nn.Module):
    def __init__(self, sz, ni):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(sz, ni))
        self.shift = nn.Parameter(torch.zeros(sz, ni))

    def forward(self, x):
        return x*self.scale + self.shift


def _mlp(ni, nh):
    return nn.Sequential(nn.Linear(ni, nh), nn.GELU(), nn.LayerNorm(nh), nn.Linear(nh, ni))


class EmbTransformerBlk(nn.Module):
    def __init__(self, n_emb, ni, attn_chans=8):
        super().__init__()
        self.attn = SelfAttention(ni, attn_chans=attn_chans, transpose=False)
        self.mlp = _mlp(ni, ni*4)
        self.nrm1 = nn.LayerNorm(ni)
        self.nrm2 = nn.LayerNorm(ni)
        self.emb_proj = nn.Linear(n_emb, ni*2)

    def forward(self, x, t):
        emb = self.emb_proj(F.silu(t))[:, None]
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = x + self.attn(self.nrm1(x))
        x = x*(1+scale) + shift
        return x + self.mlp(self.nrm2(x))


class SimpleDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nfs=(224,448,672,896), num_layers=1,
                 attn_chans=8, attn_start=1, n_mids=8):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, nfs[0], kernel_size=3, padding=1)
        self.n_temb = nf = nfs[0]
        n_emb = nf*4
        self.emb_mlp = nn.Sequential(lin(self.n_temb, n_emb, norm=nn.BatchNorm1d),
                                     lin(n_emb, n_emb))
        self.downs = nn.ModuleList()
        n = len(nfs)
        for i in range(n):
            ni = nf
            nf = nfs[i]
            self.downs.append(DownBlock(n_emb, ni, nf, add_down=i!=n-1, num_layers=num_layers,
                                        attn_chans=0 if i<attn_start else attn_chans))

        self.le = LearnEmbSS(64, nf)
        self.mids = nn.ModuleList([EmbTransformerBlk(n_emb, nf) for _ in range(n_mids)])

        rev_nfs = list(reversed(nfs))
        nf = rev_nfs[0]
        self.ups = nn.ModuleList()
        for i in range(n):
            prev_nf = nf
            nf = rev_nfs[i]
            ni = rev_nfs[min(i+1, len(nfs)-1)]
            self.ups.append(UpBlock(n_emb, ni, prev_nf, nf, add_up=i!=n-1, num_layers=num_layers+1,
                                    attn_chans=0 if i>=n-attn_start else attn_chans))
        self.conv_out = pre_conv(nfs[0], out_channels, act=nn.SiLU, norm=nn.BatchNorm2d, bias=False)

    def forward(self, inp):
        x, t = inp
        temb = timestep_embedding(t, self.n_temb)
        emb = self.emb_mlp(temb)
        x = self.conv_in(x)
        saved = [x]
        for block in self.downs:
            x = block(x, emb)
        saved += [p for o in self.downs for p in o.saved]
        n, c, h, w = x.shape
        x = self.le(x.reshape(n, c, -1).transpose(1, 2))
        for block in self.mids:
            x = block(x, emb)
        x = x.transpose(1, 2).reshape(n, c, h, w)
        for block in self.ups:
            x = block(x, emb, saved)
        return self.conv_out(x)


def init_ddpm(model):
    for o in model.downs:
        for p in o.resnets:
            p.conv2[-1].weight.data.zero_()
    
    for o in model.ups:
        for p in o.resnets:
            p.conv2[-1].weight.data.zero_()

# ============================================================================


# Transform to turn audio array into PIL image
def to_image(audio_array):
    audio_tensor = torch.tensor(audio_array).to(torch.float32)
    audio_tensor = resampler(audio_tensor)
    mel.load_audio(raw_audio=np.array(audio_tensor))
    num_slices = mel.get_number_of_slices()
    if num_slices > 0:
        slice_idx = random.randint(0, max(0, num_slices-1))
        im = mel.audio_slice_to_image(slice_idx)
        return im
    else:
        # Return empty image if no slices
        return Image.new('L', (x_res, y_res))


# Dataset class
class ImagesDS:
    def __init__(self, audio_files):
        self.audio_files = audio_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, i):
        # Load audio
        audio_array, sr = sf.read(self.audio_files[i])
        im = to_image(audio_array)
        im = TF.to_tensor(im) - 0.5
        im = TF.resize(im, (x_res, y_res))
        return (im,)


def collate_ddpm(b):
    """Collate function that adds noise"""
    return noisify(default_collate(b)[0])


def train_colab_style(
    data_dir='data/bird_songs',
    epochs=15,
    batch_size=16,
    lr=1e-4,
    device='cuda',
    save_dir='experiments_colab'
):
    """Train using exact Colab approach with miniminiai"""
    
    print("Loading audio files...")
    import glob
    audio_files = glob.glob(str(Path(data_dir) / "*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print(f"ERROR: No audio files found in {data_dir}")
        return
    
    # Split into train/val
    random.shuffle(audio_files)
    split_idx = int(len(audio_files) * 0.9)
    train_files = audio_files[:split_idx]
    val_files = audio_files[split_idx:]
    
    # Create datasets
    tds = ImagesDS(train_files)
    vds = ImagesDS(val_files)
    
    # Create dataloaders using miniminiai
    dls = DataLoaders(
        *get_dls(tds, vds, bs=batch_size, num_workers=0, collate_fn=collate_ddpm)
    )
    
    print("Creating model...")
    # Model - using miniminiai's SimpleDiffusion (same as Colab)
    model = SimpleDiffusion(
        in_channels=1,
        out_channels=1,
        nfs=(16, 32, 256, 384, 512),
        num_layers=1,
        attn_chans=0,
        n_mids=6
    )
    
    # Initialize DDPM
    def init_ddpm(model):
        for o in model.downs:
            for p in o.resnets:
                p.conv2[-1].weight.data.zero_()
        
        for o in model.ups:
            for p in o.resnets:
                p.conv2[-1].weight.data.zero_()
    
    init_ddpm(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup (matching Colab)
    opt_func = partial(optim.AdamW, eps=1e-5, weight_decay=1e-5)
    tmax = epochs * len(dls.train)
    sched = partial(lr_scheduler.OneCycleLR, max_lr=lr, total_steps=tmax)
    
    # Callbacks (removed ProgressCB due to formatting issues on Windows)
    cbs = [
        DeviceCB(),
        # ProgressCB(plot=False),  # Disabled - causes formatting error
        MetricsCB(),
        BatchSchedCB(sched),
        TrainCB()
    ]
    
    # Create learner
    print("Creating learner...")
    learn = Learner(model, dls, nn.MSELoss(), lr=lr, cbs=cbs, opt_func=opt_func)
    
    # Train
    print(f"Training for {epochs} epochs...")
    learn.fit(epochs)
    
    # Save model
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path / 'model_final.pt')
    print(f"\nModel saved to {save_path / 'model_final.pt'}")
    
    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    
    sz = (4, 1, x_res, y_res)
    
    # DDIM sampling function
    def ddim_step(x_t, noise, abar_t, abar_t1, bbar_t, bbar_t1, eta, sig, clamp=1.):
        sig = ((bbar_t1/bbar_t).sqrt() * (1-abar_t/abar_t1).sqrt()) * eta
        x_0_hat = (x_t-(1-abar_t).sqrt()*noise) / abar_t.sqrt()
        if clamp:
            x_0_hat = x_0_hat.clamp(-clamp, clamp)
        if bbar_t1 <= sig**2 + 0.01:
            sig = 0.
        x_t = abar_t1.sqrt()*x_0_hat + (bbar_t1-sig**2).sqrt()*noise
        x_t += sig * torch.randn(x_t.shape).to(x_t)
        return x_0_hat, x_t
    
    with torch.no_grad():
        preds = sample(ddim_step, model, sz, steps=100, eta=1., clamp=1.)
        s = (preds[-1] + 0.5)
    
    # Save samples
    for i in range(len(s)):
        im_array = np.array((s[i][0]).clip(0, 1)*255)
        im = Image.fromarray(im_array.astype(np.uint8), mode='L')
        
        # Save image
        im.save(save_path / f'sample_{i}_spec.png')
        
        # Convert to audio and save
        audio = mel.image_to_audio(im)
        sf.write(save_path / f'sample_{i}.wav', audio, sample_rate_mel)
        print(f"Saved sample {i}")
    
    print(f"\n✅ Training complete! Samples saved to {save_path}")
    return learn, model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_dir', type=str, default='data/bird_songs')
    parser.add_argument('--save_dir', type=str, default='experiments_colab')
    
    args = parser.parse_args()
    
    train_colab_style(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )
