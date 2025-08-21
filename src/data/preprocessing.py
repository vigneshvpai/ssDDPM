"""
Preprocessing script to parse DWI dataset and create data lists.
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import nibabel as nib

from src.config import Config


def find_dwi_files(data_root: str) -> List[Dict]:
    """
    Scan the dataset directory and find all DWI files with their corresponding bval/bvec files.
    
    Args:
        data_root: Root directory containing subject folders
        
    Returns:
        List of dictionaries containing file paths and metadata
    """
    data_list = []
    
    for subject_dir in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject_dir)
        
        if not os.path.isdir(subject_path):
            continue
            
        # Find all .nii.gz files in the subject directory
        for file in os.listdir(subject_path):
            if file.endswith('.nii.gz'):
                # Extract base name without extension
                base_name = file.replace('.nii.gz', '')
                
                # Construct paths for corresponding files
                nii_path = os.path.join(subject_path, file)
                bval_path = os.path.join(subject_path, f"{base_name}.bval")
                bvec_path = os.path.join(subject_path, f"{base_name}.bvec")
                
                # Check if corresponding files exist
                if os.path.exists(bval_path) and os.path.exists(bvec_path):
                    # Load basic metadata
                    try:
                        img = nib.load(nii_path)
                        shape = img.shape
                        bvals = np.loadtxt(bval_path)
                        bvecs = np.loadtxt(bvec_path)
                        
                        data_item = {
                            'subject_id': subject_dir,
                            'acquisition_id': base_name,
                            'nii_path': nii_path,
                            'bval_path': bval_path,
                            'bvec_path': bvec_path,
                            'shape': shape,
                            'num_b_values': len(bvals),
                            'b_values': bvals.tolist(),
                            'b_vectors': bvecs.tolist() if len(bvecs.shape) == 2 else bvecs.reshape(-1, 3).tolist()
                        }
                        data_list.append(data_item)
                        
                    except Exception as e:
                        print(f"Error processing {nii_path}: {e}")
                        continue
                else:
                    print(f"Missing bval/bvec files for {nii_path}")
    
    return data_list


def split_dataset(data_list: List[Dict], train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_list: List of data items
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Group by subject to ensure no data leakage
    subjects = list(set(item['subject_id'] for item in data_list))
    random.shuffle(subjects)
    
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    train_data = [item for item in data_list if item['subject_id'] in train_subjects]
    val_data = [item for item in data_list if item['subject_id'] in val_subjects]
    test_data = [item for item in data_list if item['subject_id'] in test_subjects]
    
    return train_data, val_data, test_data


def save_data_list(data_list: List[Dict], output_path: str):
    """
    Save data list to JSON file.
    
    Args:
        data_list: List of data items
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved {len(data_list)} items to {output_path}")


def create_dataset_lists():
    """
    Main function to create dataset lists.
    """
    print("Scanning dataset directory...")
    data_list = find_dwi_files(Config.DATA_ROOT)
    
    if not data_list:
        print("No valid DWI files found!")
        return
    
    print(f"Found {len(data_list)} DWI acquisitions from {len(set(item['subject_id'] for item in data_list))} subjects")
    
    # Create output directory
    Config.create_dirs()
    
    # Split dataset
    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(
        data_list, 
        Config.TRAIN_RATIO, 
        Config.VAL_RATIO, 
        Config.TEST_RATIO
    )
    
    # Save data lists
    save_data_list(train_data, Config.TRAIN_DATA_LIST)
    save_data_list(val_data, Config.VAL_DATA_LIST)
    save_data_list(test_data, Config.TEST_DATA_LIST)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Training: {len(train_data)} acquisitions from {len(set(item['subject_id'] for item in train_data))} subjects")
    print(f"Validation: {len(val_data)} acquisitions from {len(set(item['subject_id'] for item in val_data))} subjects")
    print(f"Test: {len(test_data)} acquisitions from {len(set(item['subject_id'] for item in test_data))} subjects")
    
    # Print shape statistics
    shapes = [item['shape'] for item in data_list]
    print(f"\nImage shapes: {set(shapes)}")
    print(f"B-values: {set(item['num_b_values'] for item in data_list)}")


if __name__ == "__main__":
    create_dataset_lists()
