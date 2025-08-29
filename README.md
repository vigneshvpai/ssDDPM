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
│   │   ├── Postprocess.py     # Data postprocessing utilities
│   │   ├── dataset_split/     # Dataset split JSON files
│   │   └── utils/             # Data utilities
│   │       ├── convert_to_pt.py    # Convert NIfTI to PyTorch format
│   │       ├── split_dataset.py    # Dataset splitting utilities
│   │       └── get_data_summary.py # Data analysis utilities
│   └── model/
│       ├── SSDDPM.py          # Main ssDDPM model implementation
│       └── ADC.py             # ADC estimation model
├── train.py                   # Training script
├── inference.py               # Inference script
├── environment.yml            # Conda environment configuration
├── checkpoints/               # Trained model checkpoints
├── inference_results/         # Generated inference results
├── lightning_logs/            # Training logs
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
- **Inference Pipeline**: Complete denoising pipeline with progress tracking

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
- **Postprocess**: Data postprocessing for inference results
- **Utils**: Data conversion, splitting, and analysis utilities

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
- **ADC**: Average ADC estimation with 25 b-values

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Diffusers library
- Nibabel (for NIfTI file handling)
- NumPy
- Matplotlib
- tqdm

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ssddpm
```

Or install dependencies manually:
```bash
conda create -n ssddpm python>=3.8
conda activate ssddpm
conda install pytorch torchvision pytorch-cuda -c pytorch
conda install lightning diffusers nibabel numpy matplotlib tqdm -c conda-forge
```

### Data Requirements

The model expects DWI data in the following format:
- **Data Structure**: PyTorch files containing DWI images
- **Image Shape**: `(width, height, slices, b_values)` where b_values=25
- **Expected Dimensions**: `(108, 134, 25, 25)` before preprocessing
- **Data Split**: JSON files defining train/val/test splits

### Environment Setup

1. Clone the repository
2. Create and activate the conda environment using `environment.yml`
3. Configure data paths in `src/config/config.py`
4. Prepare dataset split JSON files using the utilities in `src/data/utils/`

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
- Save checkpoints in the `checkpoints/` directory
- Log training progress to `lightning_logs/`

### Inference

To run inference on trained models:

```bash
python inference.py --checkpoint path/to/checkpoint.ckpt --save_dir inference_results
```

The inference script will:
- Load a trained model from checkpoint
- Process test data through the complete denoising pipeline
- Generate denoised DWI images
- Save results as NIfTI files in the specified directory
- Apply postprocessing to restore original data format

### Model Architecture

- **Noise Prediction**: UNet2D with attention blocks
- **ADC Estimation**: SVD-based linear regression
- **Loss Function**: MSE noise loss + λ × MSE self-supervised loss
- **Optimizer**: Adam with cosine annealing learning rate
- **Inference**: 250-step denoising process with progress tracking

## Data Preprocessing

The preprocessing pipeline includes:

1. **Padding**: Zero-padding to UNet-compatible dimensions
2. **Normalization**: Global normalization to [0,1] range
3. **Reshaping**: Flattening slices and b-values for UNet input
4. **B-value Integration**: Proper handling of diffusion gradient directions

## Data Postprocessing

The postprocessing pipeline for inference results:

1. **Unflattening**: Restore original slice and b-value dimensions
2. **Unpadding**: Remove padding to restore original image dimensions
3. **Denormalization**: Restore original intensity scale using saved min/max values
4. **NIfTI Export**: Save results in standard medical imaging format

## Model Outputs

The trained model provides:
- **Noise Prediction**: Accurate noise estimation for diffusion denoising
- **ADC Parameters**: S₀ and D maps for tissue characterization
- **Self-Supervised Features**: Learned representations through ADC estimation
- **Denoised Images**: High-quality DWI reconstructions

## Current Status

- ✅ **Training Pipeline**: Complete with PyTorch Lightning integration
- ✅ **Inference Pipeline**: Complete with NIfTI export capabilities
- ✅ **Model Implementation**: Full ssDDPM algorithm implementation
- ✅ **Data Processing**: Complete preprocessing and postprocessing
- ✅ **Checkpoints**: Trained models available in `checkpoints/`
- ✅ **Results**: Inference results available in `inference_results/`
- 🔄 **Directional ADC**: Not yet implemented (marked as TODO)

## Performance

The model has been trained and tested with:
- **Training**: 10 epochs completed with validation loss monitoring
- **Inference**: 250-step denoising process with progress tracking
- **Memory**: Efficient batch processing with configurable batch sizes
- **Output**: NIfTI format compatible with medical imaging software

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
