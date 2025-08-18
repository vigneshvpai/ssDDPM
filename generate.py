import torch
import diffusers
from diffusers import DDPMPipeline
import lightning as L
from ssddpm import DiffusionModel
import os


def generate_samples(checkpoint_path, num_samples=16, output_dir="generated_samples"):
    """
    Generate samples using a saved model checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint file
        num_samples: Number of samples to generate
        output_dir: Directory to save generated samples
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model from checkpoint
    model = DiffusionModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create a pipeline for generation
    pipeline = DDPMPipeline(unet=model.model, scheduler=model.scheduler)

    # Generate samples
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        samples = pipeline(
            batch_size=num_samples,
            generator=torch.Generator(device=device).manual_seed(42),
        ).images

    # Save samples
    for i, sample in enumerate(samples):
        sample_path = os.path.join(output_dir, f"sample_{i:03d}.png")
        sample.save(sample_path)
        print(f"Saved {sample_path}")

    print(f"Generated {num_samples} samples in {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate samples from trained diffusion model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_samples", help="Output directory"
    )

    args = parser.parse_args()

    generate_samples(args.checkpoint, args.num_samples, args.output_dir)
