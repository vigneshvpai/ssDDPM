#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

unset SLURM_EXPORT_ENV

module load python
conda activate thesis

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment_name> [direction]"
    echo "  experiment_name: Name of the experiment"
    echo "  direction: Optional direction (x, y, z). If not provided, uses parent folder files"
    exit 1
fi

EXP_NAME="$1"
DIRECTION="$2"

# Determine JSON file source directory
if [ -n "$DIRECTION" ]; then
    JSON_SOURCE_DIR="src/data/dataset_split_slicewise/$DIRECTION"
    echo "Using direction-specific JSON files from: $JSON_SOURCE_DIR"
else
    JSON_SOURCE_DIR="src/data/dataset_split_slicewise"
    echo "Using parent directory JSON files from: $JSON_SOURCE_DIR"
fi

# Validate that the JSON source directory exists
if [ ! -d "$JSON_SOURCE_DIR" ]; then
    echo "Error: JSON source directory does not exist: $JSON_SOURCE_DIR"
    exit 1
fi

# Validate that required JSON files exist
for json_file in train.json val.json test.json; do
    if [ ! -f "$JSON_SOURCE_DIR/$json_file" ]; then
        echo "Error: Required JSON file not found: $JSON_SOURCE_DIR/$json_file"
        exit 1
    fi
done

# Use the HPC-provided TMPDIR (which has SSD storage)
echo "HPC TMPDIR: $TMPDIR"
echo "PT_DATA_ROOT_SLICEWISE: /home/vault/mfdp/mfdp118h/pt_data_slicewise"

# Set HPC_DATA_ROOT to use the HPC-provided TMPDIR directly
export HPC_DATA_ROOT="$TMPDIR/pt_data"

# Create the pt_data directory in TMPDIR if it doesn't exist
mkdir -p "$HPC_DATA_ROOT"

# Check if data already exists in HPC location
if [ -d "$HPC_DATA_ROOT" ] && [ "$(ls -A $HPC_DATA_ROOT)" ]; then
    echo "Data already exists in HPC location: $HPC_DATA_ROOT"
else
    echo "Moving data from PT_DATA_ROOT_SLICEWISE to HPC location..."
    echo "Source: /home/vault/mfdp/mfdp118h/pt_data_slicewise"
    echo "Destination: $HPC_DATA_ROOT"
    
    # Copy data from vault to HPC SSD (using rsync for efficiency)
    rsync -av --inplace /home/vault/mfdp/mfdp118h/pt_data_slicewise/ "$HPC_DATA_ROOT/"
    
    echo "Data transfer completed!"
fi

# Copy JSON split files to TMPDIR
echo "Copying JSON split files to TMPDIR..."
cp "$JSON_SOURCE_DIR/train.json" "$TMPDIR/"
cp "$JSON_SOURCE_DIR/val.json" "$TMPDIR/"
cp "$JSON_SOURCE_DIR/test.json" "$TMPDIR/"

echo "JSON files copied to: $TMPDIR"

# Print the data paths being used
echo "Using HPC_DATA_ROOT: $HPC_DATA_ROOT"
echo "Using JSON files from: $TMPDIR"
echo "Experiment name: $EXP_NAME"
if [ -n "$DIRECTION" ]; then
    echo "Direction: $DIRECTION"
fi

# Run the training script with experiment name
echo "Python script is starting execution..."
srun python train.py --hpc --exp-name "$EXP_NAME"

echo "Training completed!"