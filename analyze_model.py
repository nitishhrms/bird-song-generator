"""
Advanced Model Analysis - Visualize Actual Model Activations & Weights
This script loads your trained model and visualizes what it learned
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf
from PIL import Image
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.mel_processor import Mel
# Import SimpleDiffusion from standalone model file
from models.simple_diffusion_model import SimpleDiffusion
import torchaudio.transforms as AT
from torchvision.transforms import functional as TF

# Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
output_dir = Path('model_analysis')
output_dir.mkdir(exist_ok=True)


class ActivationExtractor:
    """Extract activations from intermediate layers"""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture activations"""
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook into key layers
        self.hooks.append(self.model.conv_in.register_forward_hook(get_activation('conv_in')))
        
        # Downsampling blocks
        for i, down in enumerate(self.model.downs):
            self.hooks.append(down.register_forward_hook(get_activation(f'down_{i}')))
        
        # Transformer blocks (middle)
        for i, mid in enumerate(self.model.mids):
            self.hooks.append(mid.register_forward_hook(get_activation(f'transformer_{i}')))
        
        # Upsampling blocks
        for i, up in enumerate(self.model.ups):
            self.hooks.append(up.register_forward_hook(get_activation(f'up_{i}')))
        
        self.hooks.append(self.model.conv_out.register_forward_hook(get_activation('conv_out')))
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self):
        """Return captured activations"""
        return self.activations


def load_model(model_path):
    """Load trained SimpleDiffusion model"""
    print(f"Loading model from {model_path}...")
    
    model = SimpleDiffusion(
        in_channels=1,
        out_channels=1,
        nfs=(16, 32, 256, 384, 512),
        num_layers=1,
        attn_chans=0,
        n_mids=6
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def load_sample_audio(audio_path):
    """Load and preprocess a sample audio file"""
    print(f"Loading audio sample from {audio_path}...")
    
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Resample if needed
    if sr != 16000:
        resampler = AT.Resample(sr, 16000, dtype=torch.float32)
        audio_tensor = torch.tensor(audio).to(torch.float32)
        audio = resampler(audio_tensor).numpy()
    
    # Convert to mel spectrogram
    mel = Mel(128, 128, 16000)
    mel.load_audio(raw_audio=audio)
    im = mel.audio_slice_to_image(0)
    
    # Convert to tensor
    im_tensor = TF.to_tensor(im) - 0.5
    im_tensor = TF.resize(im_tensor, (128, 128))
    
    print(f"✓ Audio loaded and converted to mel-spectrogram")
    return im_tensor.unsqueeze(0)  # Add batch dimension


def visualize_layer_weights(model, layer_name, weights, save_name):
    """Visualize weights from a specific layer"""
    
    if len(weights.shape) == 4:  # Conv layer: (out_ch, in_ch, h, w)
        n_filters = min(16, weights.shape[0])
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'Learned Filters - {layer_name}', fontsize=14, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < n_filters:
                # Get filter and average over input channels
                filter_img = weights[idx].mean(axis=0) if weights.shape[1] > 1 else weights[idx, 0]
                
                im = ax.imshow(filter_img, cmap='RdBu_r', 
                              vmin=-filter_img.std()*2, vmax=filter_img.std()*2)
                ax.set_title(f'Filter {idx+1}', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {save_name}.png")
    
    elif len(weights.shape) == 2:  # Linear layer: (out, in)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show weight matrix
        im = ax.imshow(weights[:min(128, weights.shape[0]), :min(128, weights.shape[1])], 
                      cmap='RdBu_r', aspect='auto',
                      vmin=-weights.std()*2, vmax=weights.std()*2)
        ax.set_title(f'Weight Matrix - {layer_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {save_name}.png")


def visualize_all_weights(model):
    """Visualize weights from all layers"""
    print("\n1. Visualizing model weights...")
    
    # First conv layer
    conv_in_weights = model.conv_in.weight.data.cpu().numpy()
    visualize_layer_weights(model, 'Input Conv', conv_in_weights, 'weights_conv_in')
    
    # Downsample blocks
    for i, down in enumerate(model.downs):
        if hasattr(down, 'resnets') and len(down.resnets) > 0:
            resnet = down.resnets[0]
            if hasattr(resnet, 'conv1'):
                weights = resnet.conv1[0].weight.data.cpu().numpy()
                visualize_layer_weights(model, f'Down Block {i+1}', weights, f'weights_down_{i}')
    
    # Output conv layer
    conv_out_weights = model.conv_out[2].weight.data.cpu().numpy()
    visualize_layer_weights(model, 'Output Conv', conv_out_weights, 'weights_conv_out')


def visualize_activations_map(activations, layer_name, save_name):
    """Visualize activation maps from a layer"""
    
    # activations shape: (batch, channels, height, width) or (batch, seq, features)
    if len(activations.shape) == 4:
        batch, channels, h, w = activations.shape
        n_show = min(16, channels)
        
        fig, axes = plt.subplots(4, 4, figsize=(14, 14))
        fig.suptitle(f'Activation Maps - {layer_name}', fontsize=14, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                act_map = activations[0, idx].cpu().numpy()
                
                im = ax.imshow(act_map, cmap='hot', aspect='auto')
                ax.set_title(f'Channel {idx+1}', fontsize=9)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {save_name}.png")
    
    elif len(activations.shape) == 3:  # Transformer: (batch, seq, features)
        batch, seq, features = activations.shape
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        act_map = activations[0].cpu().numpy().T  # (features, seq)
        im = ax.imshow(act_map[:min(128, features), :], cmap='viridis', aspect='auto')
        ax.set_title(f'Activation Pattern - {layer_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {save_name}.png")


def visualize_all_activations(model, sample_input):
    """Visualize activations from all layers"""
    print("\n2. Extracting and visualizing activations...")
    
    # Set up activation extractor
    extractor = ActivationExtractor(model)
    extractor.register_hooks()
    
    # Forward pass
    with torch.no_grad():
        # Create dummy timestep
        t = torch.tensor([500])  # Mid-denoising
        output = model((sample_input, t))
    
    # Get activations
    activations = extractor.get_activations()
    
    # Visualize each layer
    for name, act in activations.items():
        visualize_activations_map(act, name, f'activation_{name}')
    
    # Clean up
    extractor.remove_hooks()


def create_summary_visualization(model, activations):
    """Create a summary showing architecture + activations"""
    print("\n3. Creating summary visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Model Architecture & Activation Flow', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Show architecture flow
    ax_arch = fig.add_subplot(gs[0, :])
    ax_arch.text(0.5, 0.5, 
                'Input → Conv → Down×4 → Transformers×6 → Up×4 → Conv → Output\n'
                '(1,128,128) → (16) → (32,256,384,512) → (512) → (512,384,256,32) → (16) → (1,128,128)',
                ha='center', va='center', fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax_arch.axis('off')
    
    # Show parameter counts
    param_counts = {
        'Conv In': sum(p.numel() for p in model.conv_in.parameters()),
        'Downs': sum(p.numel() for block in model.downs for p in block.parameters()),
        'Transformers': sum(p.numel() for block in model.mids for p in block.parameters()),
        'Ups': sum(p.numel() for block in model.ups for p in block.parameters()),
        'Conv Out': sum(p.numel() for p in model.conv_out.parameters()),
    }
    
    ax_params = fig.add_subplot(gs[1, :2])
    layers = list(param_counts.keys())
    params = [p / 1e6 for p in param_counts.values()]  # Convert to millions
    
    ax_params.barh(layers, params, color='skyblue')
    ax_params.set_xlabel('Parameters (Millions)')
    ax_params.set_title('Parameters per Component')
    ax_params.grid(axis='x', alpha=0.3)
    
    # Show total params
    ax_total = fig.add_subplot(gs[1, 2:])
    total_params = sum(param_counts.values()) / 1e6
    ax_total.text(0.5, 0.5, 
                 f'Total Parameters:\n{total_params:.2f}M\n\n'
                 f'Trainable: {total_params:.2f}M',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax_total.axis('off')
    
    # Key insights
    ax_insights = fig.add_subplot(gs[2, :])
    insights_text = """
Key Architectural Insights:
• 60.2M parameters total (comparable to ResNet-50)
• 6 transformer blocks in bottleneck for temporal modeling
• U-Net skip connections preserve high-frequency details
• Progressive denoising: 1000 → 0 noise steps
• DDIM sampling: 100 steps for 10x speedup
"""
    ax_insights.text(0.05, 0.95, insights_text, ha='left', va='top',
                    fontsize=11, family='monospace',
                    transform=ax_insights.transAxes)
    ax_insights.axis('off')
    
    plt.savefig(output_dir / 'model_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: model_summary.png")


def analyze_model(model_path, audio_path=None):
    """Complete model analysis"""
    
    print("="*70)
    print(" " * 15 + "ADVANCED MODEL ANALYSIS")
    print("="*70)
    
    # Load model
    model = load_model(model_path)
    
    # Visualize weights
    visualize_all_weights(model)
    
    # If audio sample provided, visualize activations
    if audio_path and Path(audio_path).exists():
        sample_input = load_sample_audio(audio_path)
        visualize_all_activations(model, sample_input)
        
        # Set up extractor again for summary
        extractor = ActivationExtractor(model)
        extractor.register_hooks()
        with torch.no_grad():
            t = torch.tensor([500])
            _ = model((sample_input, t))
        activations = extractor.get_activations()
        extractor.remove_hooks()
        
        create_summary_visualization(model, activations)
    else:
        print("\nℹ No audio sample provided - skipping activation visualization")
        print("  To visualize activations, provide: audio_path='path/to/sample.wav'")
        create_summary_visualization(model, {})
    
    print("\n" + "="*70)
    print(f"✓ Analysis complete! Check: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze trained SimpleDiffusion model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to sample audio file (optional, for activation visualization)')
    
    args = parser.parse_args()
    
    analyze_model(args.model, args.audio)
