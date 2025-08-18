import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L

# Enable Tensor Core optimization for A100 GPU
torch.set_float32_matmul_precision("high")


# Define the diffusion model using PyTorch Lightning
class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Initialize a simple UNet model for 2D images with sample size 32x32
        self.model = diffusers.models.UNet2DModel(sample_size=32)
        # Enable gradient checkpointing to save memory during backpropagation
        self.model.enable_gradient_checkpointing()
        # Use the default DDPM scheduler for noise scheduling
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        # Get the images from the batch
        images = batch["images"]
        # Generate random Gaussian noise with the same shape as images
        noise = torch.randn_like(images)
        # Randomly select a timestep for each image in the batch
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,  # total number of timesteps
            (images.size(0),),  # one timestep per image
            device=self.device,
        )
        # Add noise to the images according to the selected timesteps
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        # Predict the noise residual using the model
        residual = self.model(noisy_images, steps).sample
        # Compute mean squared error loss between predicted and true noise
        loss = torch.nn.functional.mse_loss(residual, noise)
        # Log the training loss for monitoring
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use AdamW optimizer for training
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # Use a step learning rate scheduler that decays every epoch
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]


# Define the data module for loading and augmenting data
class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        # Define a set of augmentations to apply to each image
        self.augment = transforms.Compose(
            [
                transforms.Resize(32),  # Resize image to 32x32
                transforms.CenterCrop(32),  # Center crop to 32x32
                transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
                transforms.ToTensor(),  # Convert PIL image to tensor
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )

    def prepare_data(self):
        # Download the CIFAR-10 dataset if not already present
        load_dataset("cifar10")

    def train_dataloader(self):
        # Load the CIFAR-10 dataset
        dataset = load_dataset("cifar10")
        # Apply the augmentations to each image in the dataset
        dataset.set_transform(
            lambda sample: {"images": [self.augment(image) for image in sample["img"]]}
        )
        # Return a PyTorch DataLoader for the training set
        return torch.utils.data.DataLoader(
            dataset["train"], batch_size=32, shuffle=True, num_workers=4
        )


# Main training loop
if __name__ == "__main__":
    model = DiffusionModel()  # Instantiate the model
    data = DiffusionData()  # Instantiate the data module
    # Create a PyTorch Lightning trainer with 150 epochs and bfloat16 mixed precision
    trainer = L.Trainer(max_epochs=150, precision="bf16-mixed")
    # Start training
    trainer.fit(model, data)
