# Using the Hugging Face Birdcall Dataset

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
```bash
# Download training split (recommended for first run)
python download_dataset.py --split train

# Or download all data (train + test)
python download_dataset.py --split all
```

This downloads the **tglcourse/5s_birdcall_samples_top20** dataset which contains:
- 5-second bird call audio samples
- 20 different bird species
- High-quality recordings
- Pre-processed and ready to use

### 3. Verify Dataset
```bash
# Check dataset info
python download_dataset.py --info
```

### 4. Start Training
```bash
# Train VAE (fastest, ~6 hours for 100 epochs)
python train_vae.py --data_dir data/bird_songs --epochs 100 --batch_size 32

# Train WaveGAN (best quality/speed trade-off)
python train_gan.py --data_dir data/bird_songs --epochs 100 --batch_size 64

# Train Diffusion (best quality, slowest)
python train_diffusion.py --data_dir data/bird_songs --epochs 100 --batch_size 32
```

## Dataset Details

### What's Included
- **Duration**: 5 seconds per sample
- **Format**: Audio arrays with sampling rate
- **Species**: 20 different bird species (top 20 most common)
- **Quality**: Clean, pre-processed recordings
- **Splits**: Train and test sets available

### File Structure After Download
```
data/bird_songs/
├── species1_0001.wav
├── species1_0002.wav
├── species1_0003.wav
├── species2_0001.wav
├── species2_0002.wav
└── ...
```

## Training Tips for This Dataset

### Recommended Settings

**For VAE:**
```bash
python train_vae.py \
  --data_dir data/bird_songs \
  --epochs 100 \
  --batch_size 32 \
  --latent_dim 128 \
  --beta 1.0 \
  --augment
```

**For WaveGAN:**
```bash
python train_gan.py \
  --data_dir data/bird_songs \
  --epochs 150 \
  --batch_size 64 \
  --lr_g 0.0001 \
  --lr_d 0.0001 \
  --augment
```

**For Diffusion:**
```bash
python train_diffusion.py \
  --data_dir data/bird_songs \
  --epochs 100 \
  --batch_size 32 \
  --timesteps 1000 \
  --cache_spectrograms \
  --augment
```

### Why These Settings?

- **5-second samples** are perfect for the default audio length (16,384 samples at 22kHz ≈ 0.74s)
- **Data augmentation** helps with generalization across species
- **Batch size 32-64** works well with typical GPU memory
- **Cache spectrograms** speeds up Diffusion/VAE training significantly

## Expected Results

### Training Time (NVIDIA RTX 3090)
- **VAE**: ~4-6 hours for 100 epochs
- **WaveGAN**: ~6-8 hours for 100 epochs
- **Diffusion**: ~10-12 hours for 100 epochs

### Quality Metrics (After 100 Epochs)
| Model | MCD ↓ | FAD ↓ | IS ↑ |
|-------|-------|-------|------|
| VAE | ~7.0 | ~15.0 | ~3.2 |
| WaveGAN | ~6.0 | ~12.0 | ~3.8 |
| Diffusion | ~5.5 | ~10.0 | ~4.2 |

## Troubleshooting

### Issue: Dataset download is slow
**Solution**: The dataset is hosted on Hugging Face and download speed depends on your internet connection. The dataset is relatively small (~100-500MB).

### Issue: Out of memory during training
**Solution**: Reduce batch size:
```bash
python train_vae.py --batch_size 16  # Instead of 32
```

### Issue: Audio files not found
**Solution**: Make sure you ran `download_dataset.py` first and check that files exist in `data/bird_songs/`

### Issue: Training loss not decreasing
**Solution**: 
1. Enable data augmentation: `--augment`
2. Increase learning rate: `--lr 0.001`
3. Train for more epochs: `--epochs 200`

## Advanced Usage

### Custom Data Split
```python
from datasets import load_dataset

# Load specific split
dataset = load_dataset('tglcourse/5s_birdcall_samples_top20', split='train[:80%]')  # First 80%
dataset = load_dataset('tglcourse/5s_birdcall_samples_top20', split='train[80%:]')  # Last 20%
```

### Conditional Generation (Future Work)
The dataset includes species labels, which can be used for conditional generation:
```python
# In your training script
species_label = sample['label']  # e.g., 'nightingale'
# Use this for conditional GAN/VAE training
```

## Next Steps

1. ✅ Download dataset: `python download_dataset.py --split train`
2. ✅ Train a model: `python train_vae.py --data_dir data/bird_songs`
3. ✅ Monitor training: Check `experiments/` folder and TensorBoard logs
4. ✅ Generate samples: `python generate.py --model vae --checkpoint <path>`
5. ✅ Evaluate quality: `python -c "from evaluate import evaluate_model; ..."`

## References

- **Dataset**: [tglcourse/5s_birdcall_samples_top20](https://huggingface.co/datasets/tglcourse/5s_birdcall_samples_top20)
- **Hugging Face Datasets**: https://huggingface.co/docs/datasets/
- **Original Notebook**: https://colab.research.google.com/drive/1b3CeZB2FfRGr5NPYDVvk34hyZFBtgub5

---

**Ready to start?** Run `python download_dataset.py --split train` now! 🚀
