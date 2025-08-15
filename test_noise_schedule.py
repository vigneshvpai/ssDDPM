#!/usr/bin/env python3
"""
Simple test script to verify the noise schedule optimization implementation.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_noise_schedule_optimization():
    """Test the noise schedule optimization functionality."""
    print("Testing noise schedule optimization...")
    
    try:
        from utils.noise_schedule import NoiseScheduleOptimizer, NoiseScheduleConfig
        
        # Create test data
        nex1_images = torch.randn(10, 1, 32, 32)
        nex6_images = torch.randn(10, 1, 32, 32)
        
        # Create optimizer
        config = NoiseScheduleConfig(
            num_timesteps=100,
            optimization_steps=10,  # Short for testing
            learning_rate=1e-3
        )
        optimizer = NoiseScheduleOptimizer(config)
        
        # Test schedule creation
        linear_schedule = optimizer.create_noise_schedule("linear")
        cosine_schedule = optimizer.create_noise_schedule("cosine")
        sigmoid_schedule = optimizer.create_noise_schedule("sigmoid")
        
        print(f"✓ Created schedules: linear({linear_schedule.shape}), cosine({cosine_schedule.shape}), sigmoid({sigmoid_schedule.shape})")
        
        # Test evaluation
        metrics = optimizer.evaluate_noise_schedule(nex6_images, nex1_images, linear_schedule)
        print(f"✓ Evaluation metrics: {metrics}")
        
        # Test optimization (short version)
        optimized_schedule, history = optimizer.optimize_noise_schedule(nex6_images, nex1_images)
        print(f"✓ Optimization completed: {optimized_schedule.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_custom_scheduler():
    """Test the custom scheduler functionality."""
    print("Testing custom scheduler...")
    
    try:
        from models.custom_scheduler import CustomDDPMScheduler
        
        # Create scheduler
        scheduler = CustomDDPMScheduler(num_train_timesteps=100)
        
        # Test basic functionality
        test_images = torch.randn(2, 1, 32, 32)
        test_noise = torch.randn(2, 1, 32, 32)
        test_timesteps = torch.randint(0, 100, (2,))
        
        noisy_images = scheduler.add_noise_with_schedule(test_images, test_noise, test_timesteps)
        print(f"✓ Custom scheduler works: {noisy_images.shape}")
        
        # Test schedule info
        info = scheduler.get_noise_schedule_info()
        print(f"✓ Schedule info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_medical_data_loader():
    """Test the medical data loader functionality."""
    print("Testing medical data loader...")
    
    try:
        from utils.medical_data_loader import create_synthetic_clinical_data
        
        # Create synthetic data
        data = create_synthetic_clinical_data(
            num_subjects=2,
            slices_per_subject=5,
            image_size=(64, 64)
        )
        
        print(f"✓ Created synthetic data: {data['nex1_data'].shape}, {data['nex6_data'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running noise schedule optimization tests...")
    print("=" * 50)
    
    tests = [
        test_noise_schedule_optimization,
        test_custom_scheduler,
        test_medical_data_loader
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
