"""
Visualization Tools for Bird Song Generator

Provides comprehensive visualization capabilities:
- Activation visualization
- Weight visualization  
- Spectrogram comparison
- Loss curves
- Latent space visualization
- Audio waveform plots
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def visualize_activations(model, input_data, layer_names=None, save_dir=None):
    """
    Visualize intermediate layer activations
    
    Args:
        model: PyTorch model
        input_data: Input tensor
        layer_names: List of layer names to visualize (None for all Conv layers)
        save_dir: Directory to save visualizations
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    activations = {}
    hooks = []
    
    # Register hooks to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize activations
    for name, activation in activations.items():
        act = activation[0].cpu().numpy()  # First sample in batch
        
        # Handle different dimensions
        if act.ndim == 2:  # 1D conv output (channels, length)
            n_channels = min(act.shape[0], 16)  # Show max 16 channels
            fig, axes = plt.subplots(n_channels, 1, figsize=(14, n_channels * 2))
            if n_channels == 1:
                axes = [axes]
            
            for i in range(n_channels):
                axes[i].plot(act[i])
                axes[i].set_title(f'Channel {i}')
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(f'Activations: {name}')
            plt.tight_layout()
            
        elif act.ndim == 3:  # 2D conv output (channels, height, width)
            n_channels = min(act.shape[0], 16)
            n_cols = 4
            n_rows = (n_channels + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
            axes = axes.flatten() if n_channels > 1 else [axes]
            
            for i in range(n_channels):
                im = axes[i].imshow(act[i], cmap='viridis', aspect='auto')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i])
            
            # Hide unused subplots
            for i in range(n_channels, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Activations: {name}')
            plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f'activation_{name.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
            print(f"Saved activation visualization: {name}")
        else:
            plt.show()
        
        plt.close()
    
    return activations


def visualize_weights(model, layer_names=None, save_dir=None):
    """
    Visualize model weights
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to visualize
        save_dir: Directory to save visualizations
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if layer_names is None or name in layer_names:
                weights = module.weight.data.cpu().numpy()
                
                # Handle different weight shapes
                if weights.ndim == 3:  # Conv1d: (out_channels, in_channels, kernel_size)
                    n_filters = min(weights.shape[0], 16)
                    fig, axes = plt.subplots(n_filters, 1, figsize=(12, n_filters * 2))
                    if n_filters == 1:
                        axes = [axes]
                    
                    for i in range(n_filters):
                        for j in range(weights.shape[1]):
                            axes[i].plot(weights[i, j], alpha=0.7, label=f'In {j}')
                        axes[i].set_title(f'Filter {i}')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.suptitle(f'Weights: {name}')
                    plt.tight_layout()
                    
                elif weights.ndim == 4:  # Conv2d: (out_channels, in_channels, h, w)
                    n_filters = min(weights.shape[0], 16)
                    n_cols = 4
                    n_rows = (n_filters + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
                    axes = axes.flatten() if n_filters > 1 else [axes]
                    
                    for i in range(n_filters):
                        # Show first input channel
                        w = weights[i, 0]
                        im = axes[i].imshow(w, cmap='coolwarm', aspect='auto')
                        axes[i].set_title(f'Filter {i}')
                        axes[i].axis('off')
                        plt.colorbar(im, ax=axes[i])
                    
                    # Hide unused subplots
                    for i in range(n_filters, len(axes)):
                        axes[i].axis('off')
                    
                    plt.suptitle(f'Weights: {name}')
                    plt.tight_layout()
                
                if save_dir:
                    plt.savefig(save_dir / f'weights_{name.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
                    print(f"Saved weight visualization: {name}")
                else:
                    plt.show()
                
                plt.close()


def plot_loss_curves(experiment_dir, save_path=None):
    """
    Plot training loss curves from experiment logs
    
    Args:
        experiment_dir: Path to experiment directory
        save_path: Path to save plot
    """
    experiment_dir = Path(experiment_dir)
    log_file = experiment_dir / 'log.json'
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    metrics = log_data['metrics']
    epochs = [m['epoch'] for m in metrics]
    
    # Determine which metrics are available
    metric_keys = set()
    for m in metrics:
        metric_keys.update(m.keys())
    metric_keys.discard('epoch')
    metric_keys.discard('timestamp')
    
    # Plot each metric
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric_name in enumerate(sorted(metric_keys)):
        values = [m.get(metric_name, None) for m in metrics]
        values = [v for v in values if v is not None]
        
        axes[idx].plot(epochs[:len(values)], values, linewidth=2)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
        axes[idx].set_title(f'{metric_name.replace("_", " ").title()} over Training')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Metrics: {log_data["experiment_name"]}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_latent_space(model, dataloader, device='cpu', method='tsne', save_path=None):
    """
    Visualize VAE latent space using t-SNE or UMAP
    
    Args:
        model: VAE model
        dataloader: DataLoader with data
        device: Device
        method: 'tsne' or 'umap'
        save_path: Path to save plot
    """
    from sklearn.manifold import TSNE
    
    model.eval()
    latents = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encoder(batch)
            latents.append(mu.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            latents_2d = reducer.fit_transform(latents)
        except ImportError:
            print("UMAP not installed. Using t-SNE instead.")
            reducer = TSNE(n_components=2, random_state=42)
            latents_2d = reducer.fit_transform(latents)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6, s=20)
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent space visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_spectrograms(real_specs, fake_specs, n_samples=4, save_path=None):
    """
    Side-by-side comparison of real and generated spectrograms
    
    Args:
        real_specs: Real spectrograms tensor
        fake_specs: Generated spectrograms tensor
        n_samples: Number of samples to compare
        save_path: Path to save plot
    """
    n_samples = min(n_samples, real_specs.shape[0], fake_specs.shape[0])
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Real spectrogram
        real = real_specs[i].cpu().numpy().squeeze()
        im1 = axes[i, 0].imshow(real, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Real Sample {i+1}')
        axes[i, 0].set_ylabel('Frequency')
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('Time')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Generated spectrogram
        fake = fake_specs[i].cpu().numpy().squeeze()
        im2 = axes[i, 1].imshow(fake, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Generated Sample {i+1}')
        if i == n_samples - 1:
            axes[i, 1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.suptitle('Real vs Generated Spectrograms')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization Tools")
    print("=" * 50)
    print("\nAvailable visualization functions:")
    print("  - visualize_activations(): Show intermediate layer activations")
    print("  - visualize_weights(): Display learned filters and kernels")
    print("  - plot_loss_curves(): Training metrics over time")
    print("  - visualize_latent_space(): t-SNE/UMAP for VAE latent space")
    print("  - compare_spectrograms(): Side-by-side real vs generated")
