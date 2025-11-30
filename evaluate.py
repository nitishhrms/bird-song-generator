"""
Evaluation Metrics for Bird Song Generation

Implements quantitative metrics for assessing audio quality:
- Mel Cepstral Distortion (MCD)
- Fréchet Audio Distance (FAD) - simplified version
- Inception Score (IS) - adapted for audio
- Perceptual metrics
"""

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import librosa
from pathlib import Path


def compute_mcd(real_audio, fake_audio, sr=22050, n_mfcc=13):
    """
    Compute Mel Cepstral Distortion (MCD)
    
    Lower MCD indicates better quality (more similar to real audio)
    
    Args:
        real_audio: Real audio waveform
        fake_audio: Generated audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCCs to compute
    
    Returns:
        MCD value (dB)
    """
    # Ensure same length
    min_len = min(len(real_audio), len(fake_audio))
    real_audio = real_audio[:min_len]
    fake_audio = fake_audio[:min_len]
    
    # Compute MFCCs
    real_mfcc = librosa.feature.mfcc(y=real_audio, sr=sr, n_mfcc=n_mfcc)
    fake_mfcc = librosa.feature.mfcc(y=fake_audio, sr=sr, n_mfcc=n_mfcc)
    
    # Ensure same number of frames
    min_frames = min(real_mfcc.shape[1], fake_mfcc.shape[1])
    real_mfcc = real_mfcc[:, :min_frames]
    fake_mfcc = fake_mfcc[:, :min_frames]
    
    # Compute MCD
    # MCD = (10 / ln(10)) * sqrt(2 * sum((real_mfcc - fake_mfcc)^2))
    diff = real_mfcc - fake_mfcc
    mcd = (10.0 / np.log(10)) * np.sqrt(2 * np.mean(diff ** 2))
    
    return mcd


def compute_spectral_convergence(real_spec, fake_spec):
    """
    Compute spectral convergence between spectrograms
    
    Args:
        real_spec: Real spectrogram
        fake_spec: Generated spectrogram
    
    Returns:
        Spectral convergence value
    """
    numerator = np.linalg.norm(real_spec - fake_spec, ord='fro')
    denominator = np.linalg.norm(real_spec, ord='fro')
    return numerator / (denominator + 1e-8)


def compute_log_spectral_distance(real_spec, fake_spec):
    """
    Compute log spectral distance
    
    Args:
        real_spec: Real spectrogram (in dB)
        fake_spec: Generated spectrogram (in dB)
    
    Returns:
        Log spectral distance
    """
    # Convert to power scale
    real_power = librosa.db_to_power(real_spec)
    fake_power = librosa.db_to_power(fake_spec)
    
    # Compute log spectral distance
    lsd = np.sqrt(np.mean((np.log10(real_power + 1e-8) - np.log10(fake_power + 1e-8)) ** 2))
    
    return lsd


def compute_frechet_distance(real_features, fake_features):
    """
    Compute Fréchet Distance between feature distributions
    
    Simplified version of FAD (Fréchet Audio Distance)
    
    Args:
        real_features: Features from real audio (N x D)
        fake_features: Features from generated audio (M x D)
    
    Returns:
        Fréchet distance
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute Fréchet distance
    # FD = ||mu_real - mu_fake||^2 + Tr(sigma_real + sigma_fake - 2*sqrt(sigma_real*sigma_fake))
    diff = mu_real - mu_fake
    mean_dist = np.dot(diff, diff)
    
    # Simplified: just use trace of covariance difference
    cov_dist = np.trace(sigma_real + sigma_fake - 2 * np.sqrt(sigma_real @ sigma_fake + 1e-8))
    
    fd = mean_dist + cov_dist
    
    return fd


def extract_audio_features(audio, sr=22050):
    """
    Extract features from audio for FAD computation
    
    Args:
        audio: Audio waveform
        sr: Sample rate
    
    Returns:
        Feature vector
    """
    features = []
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append([np.mean(spectral_centroid), np.std(spectral_centroid)])
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append([np.mean(zcr), np.std(zcr)])
    
    # Flatten and concatenate
    features = np.concatenate([np.array(f).flatten() for f in features])
    
    return features


def compute_fad(real_audios, fake_audios, sr=22050):
    """
    Compute Fréchet Audio Distance (FAD)
    
    Args:
        real_audios: List of real audio waveforms
        fake_audios: List of generated audio waveforms
        sr: Sample rate
    
    Returns:
        FAD value
    """
    # Extract features
    real_features = np.array([extract_audio_features(audio, sr) for audio in real_audios])
    fake_features = np.array([extract_audio_features(audio, sr) for audio in fake_audios])
    
    # Compute Fréchet distance
    fad = compute_frechet_distance(real_features, fake_features)
    
    return fad


def compute_inception_score(generated_audios, sr=22050, n_splits=10):
    """
    Compute Inception Score (IS) adapted for audio
    
    Higher IS indicates better quality and diversity
    
    Args:
        generated_audios: List of generated audio waveforms
        sr: Sample rate
        n_splits: Number of splits for computing IS
    
    Returns:
        Mean and std of IS
    """
    # Extract features
    features = np.array([extract_audio_features(audio, sr) for audio in generated_audios])
    
    # Simple classifier: use k-means to create pseudo-classes
    from sklearn.cluster import KMeans
    n_classes = 10
    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    
    # Get predictions (pseudo-probabilities)
    kmeans.fit(features)
    distances = kmeans.transform(features)
    
    # Convert distances to probabilities (softmax)
    probs = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)
    
    # Compute IS
    scores = []
    split_size = len(probs) // n_splits
    
    for i in range(n_splits):
        part = probs[i * split_size:(i + 1) * split_size]
        
        # p(y|x)
        py_x = part
        
        # p(y) = mean over samples
        py = np.mean(part, axis=0)
        
        # KL divergence
        kl = py_x * (np.log(py_x + 1e-8) - np.log(py + 1e-8))
        kl = np.sum(kl, axis=1)
        
        # IS = exp(E[KL(p(y|x) || p(y))])
        scores.append(np.exp(np.mean(kl)))
    
    return np.mean(scores), np.std(scores)


def evaluate_model(model, dataloader, device='cpu', model_type='gan', n_samples=100):
    """
    Comprehensive evaluation of a generative model
    
    Args:
        model: Generative model
        dataloader: DataLoader with real data
        device: Device
        model_type: 'gan', 'diffusion', or 'vae'
        n_samples: Number of samples to generate for evaluation
    
    Returns:
        Dictionary of metrics
    """
    from utils.audio import spectrogram_to_audio
    
    model.eval()
    
    # Collect real samples
    real_audios = []
    real_specs = []
    
    for batch in dataloader:
        if len(real_audios) >= n_samples:
            break
        
        batch = batch.to(device)
        
        # Store spectrograms
        for spec in batch:
            if len(real_specs) >= n_samples:
                break
            real_specs.append(spec.cpu())
        
        # Convert to audio if needed
        for spec in batch:
            if len(real_audios) >= n_samples:
                break
            
            spec_np = spec.cpu().numpy().squeeze()
            # Denormalize
            spec_np = (spec_np + 1) / 2 * 80 - 80
            audio = spectrogram_to_audio(spec_np, sr=22050)
            real_audios.append(audio)
    
    # Generate fake samples
    fake_audios = []
    fake_specs = []
    
    with torch.no_grad():
        if model_type == 'gan':
            # WaveGAN generates audio directly
            for _ in range(n_samples // 16):
                z = torch.randn(16, 100, device=device)
                fake_audio = model(z)
                
                for audio in fake_audio:
                    if len(fake_audios) >= n_samples:
                        break
                    fake_audios.append(audio.cpu().numpy().squeeze())
        
        elif model_type in ['diffusion', 'vae']:
            # Generate spectrograms
            for _ in range(n_samples // 16):
                if model_type == 'diffusion':
                    samples = model.sample(batch_size=16, device=device)
                else:  # vae
                    samples = model.sample(num_samples=16, device=device)
                
                for spec in samples:
                    if len(fake_specs) >= n_samples:
                        break
                    
                    fake_specs.append(spec.cpu())
                    
                    # Convert to audio
                    spec_np = spec.cpu().numpy().squeeze()
                    spec_np = (spec_np + 1) / 2 * 80 - 80
                    audio = spectrogram_to_audio(spec_np, sr=22050)
                    fake_audios.append(audio)
    
    # Compute metrics
    metrics = {}
    
    # MCD (average over pairs)
    mcd_values = []
    for i in range(min(50, len(real_audios), len(fake_audios))):
        mcd = compute_mcd(real_audios[i], fake_audios[i])
        mcd_values.append(mcd)
    metrics['mcd_mean'] = np.mean(mcd_values)
    metrics['mcd_std'] = np.std(mcd_values)
    
    # FAD
    metrics['fad'] = compute_fad(real_audios[:50], fake_audios[:50])
    
    # Inception Score
    is_mean, is_std = compute_inception_score(fake_audios)
    metrics['inception_score_mean'] = is_mean
    metrics['inception_score_std'] = is_std
    
    print("\nEvaluation Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 50)
    print("\nAvailable metrics:")
    print("  - compute_mcd(): Mel Cepstral Distortion")
    print("  - compute_fad(): Fréchet Audio Distance")
    print("  - compute_inception_score(): Inception Score for audio")
    print("  - compute_spectral_convergence(): Spectral similarity")
    print("  - compute_log_spectral_distance(): Log spectral distance")
    print("  - evaluate_model(): Comprehensive model evaluation")
