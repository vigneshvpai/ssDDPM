# Self-Supervised Diffusion Probabilistic Model (ssDDPM)

A PyTorch implementation of a self-supervised diffusion probabilistic model for diffusion-weighted imaging (DWI) data analysis. This project implements the ssDDPM algorithm from the research paper by Vasylechko et al., combining traditional diffusion models with self-supervised learning through ADC (Apparent Diffusion Coefficient) estimation.

## Overview

The ssDDPM model extends traditional diffusion probabilistic models by incorporating self-supervised regularization through ADC estimation. The model learns to predict noise in diffusion-weighted images while simultaneously learning meaningful representations through ADC parameter estimation.

## Project Structure

```
ssDDPM/
├── src/
│   ├── config/
│   │   └── config.py          # Configuration parameters
│   ├── data/
│   │   ├── DWIDataset.py      # DWI dataset implementation
│   │   ├── DWIDataLoader.py   # PyTorch Lightning data module
│   │   ├── Preprocess.py      # Data preprocessing utilities
│   │   ├── dataset_split/     # Dataset split JSON files
│   │   └── utils/             # Data utilities
│   └── model/
│       ├── SSDDPM.py          # Main ssDDPM model implementation
│       └── ADC.py             # ADC estimation model
├── train.py                   # Training script
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Key Components

### SSDDPM Model (`src/model/SSDDPM.py`)

The main model implementing the ssDDPM algorithm with the following key features:

- **UNet2DModel**: Noise prediction network `f_θ`
- **ADC Model**: Self-supervised component for S₀ and D estimation
- **DDPMScheduler**: Manages diffusion timesteps and noise schedules
- **Loss Function**: Combines noise prediction loss with self-supervised regularization

### ADC Model (`src/model/ADC.py`)

Implements the ADC estimation component that:
- Takes denoised images as input
- Estimates S₀ (signal at b=0) and D (apparent diffusion coefficient)
- Uses SVD-based pseudoinverse for robust parameter estimation
- Supports both average and directional ADC estimation (directional not yet implemented)

### Data Pipeline (`src/data/`)

- **DWIDataset**: Handles loading of DWI data from PyTorch files
- **DWIDataLoader**: PyTorch Lightning data module for efficient data loading
- **Preprocess**: Data preprocessing including normalization, reshaping, and padding

## Algorithm Implementation

The implementation follows the ssDDPM algorithm exactly:

1. **Sample batch y₀ ~ Y**: Load DWI data batch
2. **Sample t ~ Uniform({1, ..., T})**: Random timestep selection
3. **Sample ε ~ N(0, I)**: Random noise generation
4. **y_t = √ā_t y₀ + √1 - ā_t ε**: Add noise to images
5. **ê_t = f_θ(y_t, t)**: Predict noise using UNet
6. **ε₀ ~ N(0, I)**: Additional random noise
7. **y'_{t-1}**: Denoise using predicted noise
8. **Ŝ₀, D̂ ← f_ADC(y'_{t-1})**: Estimate ADC parameters
9. **ŷ_{t-1} ← Ŝ₀ e^(-b D̂)**: Reconstruct signal using ADC model
10. **Loss**: Minimize noise prediction + λ × self-supervised regularization

## Configuration

Key configuration parameters in `src/config/config.py`:

- **Model Parameters**: `in_channels=625`, `out_channels=625`, `lambda_reg=1`
- **Diffusion**: `num_train_timesteps=250`, linear beta schedule
- **Training**: `max_epochs=10`, `batch_size=2`
- **Data**: Expected shape `(108, 134, 25, 25)`, UNet compatible shape `(144, 128)`

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Diffusers library

### Data Requirements

The model expects DWI data in the following format:
- **Data Structure**: PyTorch files containing DWI images
- **Image Shape**: `(width, height, slices, b_values)` where b_values=25
- **Expected Dimensions**: `(108, 134, 25, 25)` before preprocessing
- **Data Split**: JSON files defining train/val/test splits

### Environment Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install torch torchvision lightning diffusers
   ```
3. Configure data paths in `src/config/config.py`
4. Prepare dataset split JSON files

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script will:
- Load the DWI dataset using the configured data paths
- Initialize the ssDDPM model with specified parameters
- Train for the configured number of epochs
- Use PyTorch Lightning for training orchestration

### Model Architecture

- **Noise Prediction**: UNet2D with attention blocks
- **ADC Estimation**: SVD-based linear regression
- **Loss Function**: MSE noise loss + λ × MSE self-supervised loss
- **Optimizer**: Adam with cosine annealing learning rate

## Data Preprocessing

The preprocessing pipeline includes:

1. **Padding**: Zero-padding to UNet-compatible dimensions
2. **Normalization**: Global normalization to [0,1] range
3. **Reshaping**: Flattening slices and b-values for UNet input
4. **B-value Integration**: Proper handling of diffusion gradient directions

## Model Outputs

The trained model provides:
- **Noise Prediction**: Accurate noise estimation for diffusion denoising
- **ADC Parameters**: S₀ and D maps for tissue characterization
- **Self-Supervised Features**: Learned representations through ADC estimation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite the original paper:

```
Vasylechko SD, Tsai A, Afacan O, Kurugol S. Self-supervised denoising diffusion probabilistic models for abdominal DW-MRI. Magn Reson Med. 2025; 94: 1284-1300. doi: 10.1002/mrm.30536
```

And include a reference to this repository.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

This implementation is based on the research paper by Vasylechko et al. describing the ssDDPM algorithm. The project uses PyTorch Lightning for training orchestration and the Diffusers library for diffusion model components.
