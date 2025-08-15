# Noise Schedule Optimization for Medical Imaging Diffusion Models

This implementation provides a comprehensive solution for optimizing noise schedules in diffusion models specifically for medical imaging applications, particularly DWI (Diffusion Weighted Imaging) lesion assessment.

## Overview

The key innovation in this work is optimizing the noise schedule based on SNR (Signal-to-Noise Ratio) similarity between high-SNR images (NEX=6) and low-SNR images (NEX=1) at the maximum noise level (t=T). This approach ensures that the diffusion model learns to denoise images to a clinically relevant noise level rather than completely destroying all structure.

## Key Features

- **SNR-based noise schedule optimization**: Optimizes noise schedule to maximize SNR similarity between NEX=6 and NEX=1 images
- **Multiple schedule types**: Support for linear, cosine, and sigmoid noise schedules
- **Adaptive optimization**: Can adapt the schedule during training based on data characteristics
- **Medical imaging specific**: Designed for DWI lesion assessment protocols
- **Comprehensive evaluation**: Tools for comparing different noise schedules
- **Real data support**: Utilities for loading and processing clinical DWI data

## Implementation Components

### 1. Noise Schedule Optimizer (`src/utils/noise_schedule.py`)

The core optimization module that implements the SNR-based approach:

```python
from src.utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig

# Configure optimization
config = NoiseScheduleConfig(
    num_timesteps=1000,
    target_snr_ratio=1.0,
    optimization_steps=200,
    learning_rate=1e-3
)

# Create optimizer
optimizer = NoiseScheduleOptimizer(config)

# Optimize schedule
optimized_schedule, history = optimizer.optimize_noise_schedule(nex6_images, nex1_images)
```

### 2. Custom Scheduler (`src/models/custom_scheduler.py`)

Extends the standard DDPM scheduler to support custom noise schedules:

```python
from src.models.custom_scheduler import CustomDDPMScheduler, AdaptiveNoiseScheduler

# Use custom scheduler with optimized schedule
scheduler = CustomDDPMScheduler(
    num_train_timesteps=1000,
    custom_betas=optimized_schedule
)

# Or use adaptive scheduler that updates during training
adaptive_scheduler = AdaptiveNoiseScheduler(
    num_train_timesteps=1000,
    adaptation_frequency=100
)
```

### 3. Enhanced Diffusion Model (`src/models/ssddpm.py`)

Updated diffusion model that integrates the optimized noise schedule:

```python
from src.models.ssddpm import DiffusionModel
from src.utils.noise_schedule import NoiseScheduleConfig

# Create model with optimized schedule
config = NoiseScheduleConfig()
model = DiffusionModel(
    use_optimized_schedule=True,
    schedule_config=config,
    adaptation_frequency=100
)

# Optimize schedule offline
results = model.optimize_noise_schedule_offline(nex1_data, nex6_data)
```

### 4. Medical Data Loader (`src/utils/medical_data_loader.py`)

Utilities for loading and processing clinical DWI data:

```python
from src.utils.medical_data_loader import MedicalDataLoader

# Load clinical data
loader = MedicalDataLoader()
dataset = loader.load_clinical_dataset("path/to/dwi/data")

# Create NEX=6 from NEX=1 repetitions
nex6_data = loader.create_nex6_from_nex1_repetitions(nex1_files)
```

## Usage Examples

### Basic Noise Schedule Optimization

```python
import torch
from src.utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig

# Create synthetic data (replace with real clinical data)
nex1_images = torch.randn(100, 1, 64, 64)  # NEX=1 images
nex6_images = torch.randn(100, 1, 64, 64)  # NEX=6 images

# Configure optimization
config = NoiseScheduleConfig(
    num_timesteps=1000,
    target_snr_ratio=1.0,
    optimization_steps=200
)

# Optimize schedule
optimizer = NoiseScheduleOptimizer(config)
optimized_schedule, history = optimizer.optimize_noise_schedule(nex6_images, nex1_images)

# Visualize results
optimizer.visualize_noise_schedule(optimized_schedule, "optimized_schedule.png")
```

### Training with Optimized Schedule

```python
from src.models.ssddpm import DiffusionModel
from src.utils.noise_schedule import NoiseScheduleConfig

# Create model with optimized schedule
config = NoiseScheduleConfig()
model = DiffusionModel(
    use_optimized_schedule=True,
    schedule_config=config
)

# Optimize schedule offline first
optimization_results = model.optimize_noise_schedule_offline(nex1_data, nex6_data)

# Train the model (schedule will be used automatically)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, data_module)
```

### Working with Real Clinical Data

```python
from src.utils.medical_data_loader import MedicalDataLoader, create_synthetic_clinical_data

# Option 1: Use synthetic data for testing
synthetic_data = create_synthetic_clinical_data(
    num_subjects=10,
    slices_per_subject=20,
    image_size=(128, 128)
)

nex1_data = synthetic_data["nex1_data"]
nex6_data = synthetic_data["nex6_data"]

# Option 2: Load real clinical data
loader = MedicalDataLoader()
clinical_data = loader.load_clinical_dataset("path/to/clinical/data")

nex1_data = clinical_data["nex1_data"]
nex6_data = clinical_data["nex6_data"]

# Preprocess for training
nex1_processed, nex6_processed = loader.preprocess_for_training(
    nex1_data, nex6_data, 
    target_size=(64, 64), 
    augment=True
)
```

### Complete Workflow Example

See `examples/noise_schedule_optimization.py` for a complete demonstration that includes:

1. Creating synthetic medical imaging data
2. Comparing different noise schedules
3. Optimizing the noise schedule
4. Training a diffusion model with the optimized schedule
5. Evaluating and visualizing results

## Mathematical Background

The optimization objective is to maximize SNR similarity between noise-corrupted NEX=6 images and NEX=1 images at t=T:

```
SNR_ratio = SNR(noisy_NEX6) / SNR(NEX1)
similarity = 1 / (1 + |SNR_ratio - target_ratio|)
```

Where:
- `noisy_NEX6` is the NEX=6 image corrupted with maximum noise at t=T
- `NEX1` is the original NEX=1 image
- `target_ratio` is the desired SNR ratio (typically 1.0)

The noise schedule is optimized to maximize this similarity metric.

## Configuration Options

### NoiseScheduleConfig Parameters

- `num_timesteps`: Number of diffusion timesteps (default: 1000)
- `beta_start`: Starting beta value (default: 1e-4)
- `beta_end`: Ending beta value (default: 0.02)
- `schedule_type`: Type of initial schedule ("linear", "cosine", "sigmoid")
- `target_snr_ratio`: Target SNR ratio between NEX=6 and NEX=1 (default: 1.0)
- `optimization_steps`: Number of optimization iterations (default: 100)
- `learning_rate`: Learning rate for schedule optimization (default: 1e-3)

### Model Configuration

- `use_optimized_schedule`: Enable optimized noise schedule (default: True)
- `schedule_config`: Configuration for noise schedule optimization
- `adaptation_frequency`: How often to adapt schedule during training (default: 100)

## Performance Considerations

1. **Memory Usage**: The optimization process can be memory-intensive for large datasets. Consider using smaller batches or reducing the number of timesteps.

2. **Computation Time**: Schedule optimization adds computational overhead. For production use, consider pre-optimizing the schedule offline.

3. **Data Requirements**: The optimization works best with paired NEX=1 and NEX=6 data. If only NEX=1 data is available, the system can create synthetic NEX=6 data by averaging multiple acquisitions.

## Dependencies

Required packages:
```
torch>=1.9.0
diffusers>=0.10.0
nibabel>=3.2.0
matplotlib>=3.3.0
numpy>=1.19.0
scipy>=1.7.0
lightning>=1.5.0
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the example: `python examples/noise_schedule_optimization.py`

## Citation

If you use this implementation in your research, please cite the original paper that describes the SNR-based noise schedule optimization approach for medical imaging diffusion models.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the implementation.

## License

This implementation is provided under the same license as the original project.
