"""
Download and prepare the tglcourse/5s_birdcall_samples_top20 dataset from Hugging Face

This script downloads the dataset and saves it to the data/bird_songs directory
in a format compatible with the BirdSongDataset class.
"""

from datasets import load_dataset
import soundfile as sf
from pathlib import Path
import numpy as np
from tqdm import tqdm


def download_birdcall_dataset(output_dir='data/bird_songs', split='train'):
    """
    Download the tglcourse/5s_birdcall_samples_top20 dataset from Hugging Face
    
    Args:
        output_dir: Directory to save audio files
        split: Dataset split ('train', 'test', or 'all')
    """
    
    print("Loading dataset from Hugging Face...")
    print("Dataset: tglcourse/5s_birdcall_samples_top20")
    print("=" * 60)
    
    # Load dataset WITHOUT decoding audio (to avoid torchcodec dependency)
    from datasets import load_dataset, Audio
    
    if split == 'all':
        dataset = load_dataset('tglcourse/5s_birdcall_samples_top20')
        # Combine all splits
        all_data = []
        for split_name in dataset.keys():
            # Cast audio column to disable automatic decoding
            ds = dataset[split_name].cast_column("audio", Audio(decode=False))
            all_data.extend(ds)
        data = all_data
    else:
        dataset = load_dataset('tglcourse/5s_birdcall_samples_top20', split=split)
        # Cast audio column to disable automatic decoding
        data = dataset.cast_column("audio", Audio(decode=False))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDataset loaded! Total samples: {len(data)}")
    print(f"Saving audio files to: {output_path}")
    print()
    
    # Save audio files
    saved_count = 0
    species_count = {}
    
    for idx, sample in enumerate(tqdm(data, desc="Saving audio files")):
        try:
            # Get audio data (raw bytes)
            audio = sample['audio']
            
            # Get label/species name
            label = sample.get('label', sample.get('species', sample.get('primary_label', f'species_{idx}')))
            
            # Track species
            if label not in species_count:
                species_count[label] = 0
            species_count[label] += 1
            
            # Create filename
            filename = f"{label}_{species_count[label]:04d}.wav"
            filepath = output_path / filename
            
            # Save audio file from bytes
            if 'bytes' in audio and audio['bytes'] is not None:
                # Write raw bytes directly
                with open(filepath, 'wb') as f:
                    f.write(audio['bytes'])
            elif 'path' in audio and audio['path'] is not None:
                # Copy from path
                import shutil
                shutil.copy(audio['path'], filepath)
            else:
                print(f"Warning: Could not save sample {idx}, skipping...")
                continue
                
            saved_count += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"\n[SUCCESS] Successfully saved {saved_count} audio files!")
    print(f"\nSpecies distribution:")
    print("=" * 60)
    for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {species}: {count} samples")
    
    print(f"\n[FILES] Audio files saved to: {output_path.absolute()}")
    print("\n[NEXT STEPS] You can now train models using:")
    print(f"   python train_vae.py --data_dir {output_dir}")
    print(f"   python train_gan.py --data_dir {output_dir}")
    print(f"   python train_diffusion.py --data_dir {output_dir}")


def get_dataset_info():
    """Get information about the dataset without downloading"""
    
    print("Fetching dataset information...")
    dataset = load_dataset('tglcourse/5s_birdcall_samples_top20', split='train')
    
    print("\n" + "=" * 60)
    print("Dataset: tglcourse/5s_birdcall_samples_top20")
    print("=" * 60)
    print(f"Number of samples: {len(dataset)}")
    print(f"Features: {dataset.features}")
    
    # Sample one item
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, value in sample.items():
        if key == 'audio':
            print(f"  {key}:")
            print(f"    - array shape: {value['array'].shape}")
            print(f"    - sampling_rate: {value['sampling_rate']}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download birdcall dataset from Hugging Face')
    parser.add_argument('--output_dir', type=str, default='data/bird_songs',
                        help='Output directory for audio files')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'all'],
                        help='Dataset split to download')
    parser.add_argument('--info', action='store_true',
                        help='Just show dataset info without downloading')
    
    args = parser.parse_args()
    
    if args.info:
        get_dataset_info()
    else:
        download_birdcall_dataset(args.output_dir, args.split)
