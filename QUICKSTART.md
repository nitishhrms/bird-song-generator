# Bird Song Generator - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Navigate to Project
```bash
cd C:\Users\Anush\.gemini\antigravity\scratch\bird-song-generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset (Easiest Method)
```bash
# Download the Hugging Face birdcall dataset (5-second samples, 20 species)
python download_dataset.py --split train

# This will save audio files to data/bird_songs/
# Takes ~2-5 minutes depending on your internet speed
```

**Alternative**: Manually add your own bird song audio files to `data\bird_songs\`

### 4. Train a Model (Choose One)

**Option A: VAE (Fastest, ~6 hours)**
```bash
python train_vae.py --data_dir data/bird_songs --epochs 50 --batch_size 32
```

**Option B: WaveGAN (Best Speed/Quality Trade-off, ~8 hours)**
```bash
python train_gan.py --data_dir data/bird_songs --epochs 50 --batch_size 64
```

**Option C: Diffusion (Best Quality, ~12 hours)**
```bash
python train_diffusion.py --data_dir data/bird_songs --epochs 50 --batch_size 32
```

### 5. Generate Bird Songs
```bash
# After training completes, generate samples
python generate.py --model vae --checkpoint experiments/vae_*/checkpoints/vae_epoch_50.pt --num_samples 10
```

## 📊 Project Structure

```
bird-song-generator/
├── models/          # 3 generative models (GAN, Diffusion, VAE)
├── utils/           # Audio processing, datasets, training
├── train_*.py       # Training scripts
├── generate.py      # Generation script
├── visualize.py     # Visualization tools
├── evaluate.py      # Evaluation metrics
├── README.md        # Full documentation
├── RESULTS.md       # Detailed analysis (2-3 pages)
└── PRESENTATION.md  # 15-min presentation
```

## 🎯 What Each Model Does

| Model | Input | Output | Best For |
|-------|-------|--------|----------|
| **WaveGAN** | Random noise | Raw audio | Speed & quality |
| **Diffusion** | Random noise | Spectrogram → Audio | Best quality |
| **VAE** | Random latent | Spectrogram → Audio | Control & interpolation |

## 📚 Documentation

- **README.md**: Complete installation and usage guide
- **RESULTS.md**: Experimental results, hypothesis, analysis, visualizations
- **PRESENTATION.md**: 15-minute presentation summary
- **walkthrough.md**: Detailed project walkthrough

## 🔧 Key Features

✅ Three state-of-the-art generative models  
✅ Dual audio processing (waveforms + spectrograms)  
✅ Comprehensive visualization (activations, weights, latent space)  
✅ Evaluation metrics (MCD, FAD, Inception Score)  
✅ Experiment tracking and TensorBoard logging  
✅ Complete documentation for presentations  

## 💡 Next Steps

1. **Read README.md** for detailed instructions
2. **Check RESULTS.md** for experimental analysis
3. **Use PRESENTATION.md** for your 15-minute presentation
4. **Train models** on your bird song dataset
5. **Visualize results** using `visualize.py`
6. **Evaluate quality** using `evaluate.py`

---

**Questions?** Check the documentation files or examine the code - everything is well-commented!
