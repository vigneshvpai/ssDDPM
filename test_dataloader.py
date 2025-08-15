#!/usr/bin/env python3
"""
Simple test script for DWI dataloader parsing logic
"""

import os
import re
from pathlib import Path


def parse_filtered_cases(filtered_cases_file: str):
    """
    Parse the filtered_cases.txt file to extract case information
    """
    cases = []

    if not os.path.exists(filtered_cases_file):
        print(f"Filtered cases file not found: {filtered_cases_file}")
        return cases

    with open(filtered_cases_file, "r") as f:
        content = f.read()

    # Split content into case blocks
    case_blocks = content.split("\nCase: ")[1:]  # Skip the header

    for block in case_blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        case_name = lines[0]
        case_info = {"case_name": case_name, "files": []}

        # Extract file information
        current_files = {}
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Path:"):
                case_info["case_path"] = line.split("Path: ")[1]
            elif line.startswith("- NII:"):
                current_files["nii"] = line.split("- NII: ")[1]
            elif line.startswith("- BVAL:"):
                current_files["bval"] = line.split("- BVAL: ")[1]
            elif line.startswith("- BVEC:"):
                current_files["bvec"] = line.split("- BVEC: ")[1]
            elif line.startswith("- BVAL values:"):
                bval_str = line.split("- BVAL values: ")[1]
                current_files["bval_values"] = eval(bval_str)
            elif line == "" and current_files:  # Empty line indicates end of file group
                case_info["files"].append(current_files.copy())
                current_files = {}

        # Don't forget the last file group
        if current_files:
            case_info["files"].append(current_files)

        cases.append(case_info)

    return cases


def main():
    """Test the parsing logic"""
    filtered_file = "filtered_cases.txt"

    if os.path.exists(filtered_file):
        print("Testing DWI DataLoader parsing...")

        # Parse the filtered cases
        cases = parse_filtered_cases(filtered_file)

        print(f"Successfully parsed {len(cases)} cases")

        # Show first few cases
        for i, case in enumerate(cases[:3]):
            print(f"\nCase {i+1}: {case['case_name']}")
            print(f"  Path: {case['case_path']}")
            print(f"  Files: {len(case['files'])}")

            for j, file_info in enumerate(case["files"]):
                print(f"    File {j+1}:")
                print(f"      NII: {file_info['nii']}")
                print(f"      BVAL: {file_info['bval']}")
                print(f"      BVEC: {file_info['bvec']}")
                print(
                    f"      BVAL values: {file_info['bval_values'][:5]}..."
                )  # Show first 5 values

        print(f"\nTotal cases: {len(cases)}")
        print("Parsing test completed successfully!")

    else:
        print(f"Filtered cases file not found: {filtered_file}")
        print("Please run the filtering script first.")


if __name__ == "__main__":
    main()
