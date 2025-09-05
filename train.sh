#!/bin/bash -l
#
#SBATCH --gres=gpu:rtx2080ti:4
#SBATCH --time=24:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate thesis

# Use the HPC-provided TMPDIR (which has SSD storage)
echo "HPC TMPDIR: $TMPDIR"
echo "PT_DATA_ROOT: /home/vault/mfdp/mfdp118h/pt_data"

# Set HPC_DATA_ROOT to use the HPC-provided TMPDIR directly
export HPC_DATA_ROOT="$TMPDIR/pt_data"

# Create the pt_data directory in TMPDIR if it doesn't exist
mkdir -p "$HPC_DATA_ROOT"

# Check if data already exists in HPC location
if [ -d "$HPC_DATA_ROOT" ] && [ "$(ls -A $HPC_DATA_ROOT)" ]; then
    echo "Data already exists in HPC location: $HPC_DATA_ROOT"
else
    echo "Moving data from PT_DATA_ROOT to HPC location..."
    echo "Source: /home/vault/mfdp/mfdp118h/pt_data"
    echo "Destination: $HPC_DATA_ROOT"
    
    # Copy data from vault to HPC SSD (using rsync for efficiency)
    rsync -a --progress /home/vault/mfdp/mfdp118h/pt_data/ "$HPC_DATA_ROOT/"
    
    echo "Data transfer completed!"
fi

# Copy JSON split files to TMPDIR
echo "Copying JSON split files to TMPDIR..."
cp src/data/dataset_split/train.json "$TMPDIR/"
cp src/data/dataset_split/val.json "$TMPDIR/"
cp src/data/dataset_split/test.json "$TMPDIR/"

echo "JSON files copied to: $TMPDIR"

# Print the data paths being used
echo "Using HPC_DATA_ROOT: $HPC_DATA_ROOT"
echo "Using JSON files from: $TMPDIR"

# Run the training script
echo "Starting training..."
srun python train.py --hpc 

echo "Training completed!"