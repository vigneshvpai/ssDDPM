#!/usr/bin/env python3
"""
DWI Dataset Filter Script

This script filters DWI datasets based on specific bval values.
It looks for cases where the bval file contains the exact sequence:
0 10 10 10 50 50 50 80 80 80 200 200 200 400 400 400 600 600 600 800 800 800 1000 1000 1000

Usage:
    python filter_dwi_dataset.py --data_dir $HPCVAULT/data --output_file filtered_cases.txt
"""

import os
import argparse
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Target bval sequence to match
TARGET_BVAL_SEQUENCE = [
    0,
    10,
    10,
    10,
    50,
    50,
    50,
    80,
    80,
    80,
    200,
    200,
    200,
    400,
    400,
    400,
    600,
    600,
    600,
    800,
    800,
    800,
    1000,
    1000,
    1000,
]


def read_bval_file(bval_path):
    """
    Read bval file and return the values as a list of integers

    Args:
        bval_path: Path to the .bval file

    Returns:
        List of bval values as integers
    """
    try:
        with open(bval_path, "r") as f:
            content = f.read().strip()

        # Split by whitespace and convert to integers
        bval_values = [int(x) for x in content.split()]
        return bval_values
    except Exception as e:
        logger.error(f"Error reading bval file {bval_path}: {e}")
        return None


def read_bvec_file(bvec_path):
    """
    Read bvec file and return the values as a numpy array

    Args:
        bvec_path: Path to the .bvec file

    Returns:
        Numpy array of bvec values
    """
    try:
        with open(bvec_path, "r") as f:
            lines = f.readlines()

        # Parse bvec file (3 lines: x, y, z components)
        bvec_values = []
        for line in lines:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                bvec_values.append(values)

        # Transpose to get (n_directions, 3) format
        bvec_array = np.array(bvec_values).T
        return bvec_array
    except Exception as e:
        logger.error(f"Error reading bvec file {bvec_path}: {e}")
        return None


def check_bval_match(bval_values):
    """
    Check if bval values match the target sequence

    Args:
        bval_values: List of bval values

    Returns:
        Boolean indicating if the sequence matches
    """
    if len(bval_values) != len(TARGET_BVAL_SEQUENCE):
        return False

    return bval_values == TARGET_BVAL_SEQUENCE


def find_dwi_files(case_dir):
    """
    Find all DWI files (nii.gz, bval, bvec) in a case directory

    Args:
        case_dir: Path to case directory

    Returns:
        Dictionary with lists of nii.gz, bval, and bvec files
    """
    case_path = Path(case_dir)

    nii_files = list(case_path.glob("*.nii.gz"))
    bval_files = list(case_path.glob("*.bval"))
    bvec_files = list(case_path.glob("*.bvec"))

    return {"nii": nii_files, "bval": bval_files, "bvec": bvec_files}


def process_case(case_dir):
    """
    Process a single case directory

    Args:
        case_dir: Path to case directory

    Returns:
        Dictionary with case information and matching status
    """
    case_name = os.path.basename(case_dir)
    case_path = Path(case_dir)

    # Find all DWI files
    dwi_files = find_dwi_files(case_dir)

    case_info = {
        "case_name": case_name,
        "case_path": str(case_path),
        "nii_files": [str(f) for f in dwi_files["nii"]],
        "bval_files": [str(f) for f in dwi_files["bval"]],
        "bvec_files": [str(f) for f in dwi_files["bvec"]],
        "matching_cases": [],
    }

    # Check each bval file for matches
    for bval_file in dwi_files["bval"]:
        bval_values = read_bval_file(bval_file)
        if bval_values is None:
            continue

        if check_bval_match(bval_values):
            # Find corresponding bvec and nii files
            base_name = bval_file.stem.replace(".bval", "")

            # Find corresponding bvec file
            bvec_file = None
            for bvec in dwi_files["bvec"]:
                if base_name in str(bvec):
                    bvec_file = bvec
                    break

            # Find corresponding nii.gz file
            nii_file = None
            for nii in dwi_files["nii"]:
                if base_name in str(nii):
                    nii_file = nii
                    break

            if bvec_file and nii_file:
                case_info["matching_cases"].append(
                    {
                        "bval_file": str(bval_file),
                        "bvec_file": str(bvec_file),
                        "nii_file": str(nii_file),
                        "bval_values": bval_values,
                    }
                )

    return case_info


def main():
    parser = argparse.ArgumentParser(
        description="Filter DWI dataset based on bval values"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory containing case folders",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="filtered_cases.txt",
        help="Output file to save filtered cases",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return

    logger.info(f"Scanning directory: {data_dir}")
    logger.info(f"Looking for bval sequence: {TARGET_BVAL_SEQUENCE}")

    # Get all case directories
    case_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(case_dirs)} case directories")

    matching_cases = []
    total_matches = 0

    # Process each case
    for case_dir in case_dirs:
        if args.verbose:
            logger.info(f"Processing case: {case_dir.name}")

        case_info = process_case(case_dir)

        if case_info["matching_cases"]:
            matching_cases.append(case_info)
            total_matches += len(case_info["matching_cases"])

            if args.verbose:
                logger.info(
                    f"  ✓ Found {len(case_info['matching_cases'])} matching cases"
                )
                for match in case_info["matching_cases"]:
                    logger.info(f"    - {os.path.basename(match['nii_file'])}")
        else:
            if args.verbose:
                logger.info(f"  ✗ No matches found")

    # Save results
    with open(args.output_file, "w") as f:
        f.write(f"# DWI Dataset Filter Results\n")
        f.write(f"# Target bval sequence: {TARGET_BVAL_SEQUENCE}\n")
        f.write(f"# Total matching cases: {total_matches}\n")
        f.write(f"# Total case directories: {len(case_dirs)}\n")
        f.write(f"# Matching case directories: {len(matching_cases)}\n\n")

        for case_info in matching_cases:
            f.write(f"Case: {case_info['case_name']}\n")
            f.write(f"Path: {case_info['case_path']}\n")
            f.write(f"Matching files:\n")

            for match in case_info["matching_cases"]:
                f.write(f"  - NII: {os.path.basename(match['nii_file'])}\n")
                f.write(f"  - BVAL: {os.path.basename(match['bval_file'])}\n")
                f.write(f"  - BVEC: {os.path.basename(match['bvec_file'])}\n")
                f.write(f"  - BVAL values: {match['bval_values']}\n")
                f.write(f"\n")

    # Print summary
    logger.info(f"Filtering complete!")
    logger.info(f"Total case directories processed: {len(case_dirs)}")
    logger.info(f"Case directories with matches: {len(matching_cases)}")
    logger.info(f"Total matching cases: {total_matches}")
    logger.info(f"Results saved to: {args.output_file}")

    # Print matching cases
    if matching_cases:
        logger.info("\nMatching cases:")
        for case_info in matching_cases:
            logger.info(
                f"  {case_info['case_name']}: {len(case_info['matching_cases'])} matches"
            )
            for match in case_info["matching_cases"]:
                logger.info(f"    - {os.path.basename(match['nii_file'])}")


if __name__ == "__main__":
    main()
