import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.diffusion_model import DiffusionMRIDataset, DiffusionModel


def test_dataset_initialization():
    """Test that the dataset can be initialized and finds files"""
    data_dir = "/home/vault/mfdp/mfdp118h/data"

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"SKIP: Data directory {data_dir} not found")
        return False

    dataset = DiffusionMRIDataset(data_dir=data_dir, target_size=(64, 64))

    # Check that files were found
    if len(dataset) == 0:
        print("FAIL: No NIfTI files found in dataset")
        return False
    if len(dataset.nii_files) == 0:
        print("FAIL: No NIfTI files in nii_files list")
        return False

    print("PASS: Dataset initialization")
    return True


def test_dataset_getitem():
    """Test that dataset returns properly formatted data"""
    data_dir = "/home/vault/mfdp/mfdp118h/data"

    if not os.path.exists(data_dir):
        print(f"SKIP: Data directory {data_dir} not found")
        return False

    dataset = DiffusionMRIDataset(data_dir=data_dir, target_size=(64, 64))

    if len(dataset) == 0:
        print("SKIP: No files found in dataset")
        return False

    # Get first item
    sample = dataset[0]

    # Check structure
    if "images" not in sample:
        print("FAIL: Sample should contain 'images' key")
        return False

    # Check tensor properties
    images = sample["images"]
    if not isinstance(images, torch.Tensor):
        print("FAIL: Images should be a torch.Tensor")
        return False
    if images.dtype != torch.float32:
        print("FAIL: Images should be float32")
        return False
    if images.shape != (1, 64, 64):
        print(f"FAIL: Expected shape (1, 64, 64), got {images.shape}")
        return False
    if images.min() < 0.0:
        print("FAIL: Images should be normalized to [0, 1]")
        return False
    if images.max() > 1.0:
        print("FAIL: Images should be normalized to [0, 1]")
        return False

    print("PASS: Dataset getitem")
    return True


def test_model_initialization():
    """Test that the diffusion model can be initialized"""
    model = DiffusionModel(
        data_dir="/home/vault/mfdp/mfdp118h/data", target_size=(64, 64)
    )

    # Check model components
    if not hasattr(model, "model"):
        print("FAIL: Model should have UNet component")
        return False
    if not hasattr(model, "scheduler"):
        print("FAIL: Model should have scheduler component")
        return False
    if not hasattr(model, "data_dir"):
        print("FAIL: Model should have data_dir attribute")
        return False
    if not hasattr(model, "target_size"):
        print("FAIL: Model should have target_size attribute")
        return False

    print("PASS: Model initialization")
    return True


def test_model_forward_pass():
    """Test that the model can perform a forward pass"""
    model = DiffusionModel(
        data_dir="/home/vault/mfdp/mfdp118h/data", target_size=(64, 64)
    )

    # Create dummy input
    batch_size = 2
    images = torch.randn(batch_size, 1, 64, 64)  # (batch, channel, height, width)

    # Test training step
    loss = model.training_step({"images": images}, batch_idx=0)

    # Check loss is a tensor
    if not isinstance(loss, torch.Tensor):
        print("FAIL: Loss should be a torch.Tensor")
        return False
    if not loss.requires_grad:
        print("FAIL: Loss should require gradients")
        return False

    print("PASS: Model forward pass")
    return True


def test_dataloader_creation():
    """Test that the model can create a dataloader"""
    model = DiffusionModel(
        data_dir="/home/vault/mfdp/mfdp118h/data", target_size=(64, 64)
    )

    dataloader = model.train_dataloader()

    # Check dataloader properties
    if not hasattr(dataloader, "__iter__"):
        print("FAIL: Dataloader should be iterable")
        return False
    if dataloader.batch_size != 8:
        print("FAIL: Expected batch size 8")
        return False
    if dataloader.num_workers != 4:
        print("FAIL: Expected 4 workers")
        return False

    print("PASS: Dataloader creation")
    return True


def test_dataloader_iteration():
    """Test that the dataloader can iterate and return batches"""
    model = DiffusionModel(
        data_dir="/home/vault/mfdp/mfdp118h/data", target_size=(64, 64)
    )

    dataloader = model.train_dataloader()

    # Try to get first batch
    try:
        batch = next(iter(dataloader))

        # Check batch structure
        if "images" not in batch:
            print("FAIL: Batch should contain 'images' key")
            return False

        images = batch["images"]

        # Check batch shape
        if len(images.shape) != 4:
            print(f"FAIL: Expected 4D tensor, got {len(images.shape)}D")
            return False
        if images.shape[1] != 1:
            print(f"FAIL: Expected 1 channel, got {images.shape[1]}")
            return False
        if images.shape[2:] != (64, 64):
            print(f"FAIL: Expected spatial size (64, 64), got {images.shape[2:]}")
            return False

        print("PASS: Dataloader iteration")
        return True

    except StopIteration:
        print("SKIP: No data available in dataloader")
        return False


if __name__ == "__main__":
    # Run tests
    tests = [
        test_dataset_initialization,
        test_dataset_getitem,
        test_model_initialization,
        test_model_forward_pass,
        test_dataloader_creation,
        test_dataloader_iteration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\nTest Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed!")
