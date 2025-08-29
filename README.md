# Self-Supervised Diffusion Probabilistic Model (ssDDPM)

PyTorch implementation of ssDDPM for diffusion-weighted imaging (DWI) data analysis, combining diffusion models with self-supervised learning through ADC estimation.

## Quick Start

### Setup
```bash
conda env create -f environment.yml
conda activate ssddpm
```

### Training
```bash
python train.py
```

### Inference
```bash
python inference.py --checkpoint checkpoints/ssddpm-epoch=01-val_loss=1.0012.ckpt
```

## Project Structure
```
ssDDPM/
├── src/
│   ├── model/          # SSDDPM and ADC models
│   ├── data/           # Dataset, preprocessing, utilities
│   └── config/         # Configuration
├── train.py            # Training script
├── inference.py        # Inference script
├── environment.yml     # Conda environment
└── checkpoints/        # Trained models
```

## Model

- **SSDDPM**: Main diffusion model with UNet2D noise prediction
- **ADC Model**: Self-supervised component for S₀ and D estimation
- **Loss**: Noise prediction + λ × self-supervised regularization

## Data

Expected format: PyTorch files with DWI images of shape `(108, 134, 25, 25)` (width, height, slices, b-values).

## Citation

```
Vasylechko SD, et al. Self-supervised denoising diffusion probabilistic models for abdominal DW-MRI. 
Magn Reson Med. 2025; 94: 1284-1300. doi: 10.1002/mrm.30536
```

## License

MIT License
