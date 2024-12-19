#!/usr/bin/env bash

# Datasets and their respective augmentation sets
DATASETS=("BACH" "BRACS" "BreakHis")

# Augmentations per dataset
declare -A AUGS
AUGS["BACH"]="5 10 20"
AUGS["BRACS"]="5 10 20 40"
AUGS["BreakHis"]="5 10 20 40"

# Pooling methods
POOLINGS=("mean" "GatedAttention")

# Classifier layers: 1 for linear, 2 for MLP
LAYERS=("1" "2")

# Dropout ratios
DROPOUTS=("0" "0.2" "0.4")

# Get the number of physical cores available
MAX_JOBS=$(lscpu | awk '/^Core\(s\) per socket:/ {print $4}')
CURRENT_JOBS=0

# Loop over each dataset
for d in "${DATASETS[@]}"; do
    # Select augmentation list based on dataset
    AUGMENTATIONS=(${AUGS[$d]})

    # Loop over augmentations
    for aug in "${AUGMENTATIONS[@]}"; do
        # Loop over pooling methods
        for p in "${POOLINGS[@]}"; do
            # Loop over layers
            for l in "${LAYERS[@]}"; do
                # Loop over dropout ratios
                for dr in "${DROPOUTS[@]}"; do
                    echo "Running experiment with dataset=$d, augmentation=$aug, pooling=$p, nlayers=$l, dropout=$dr"
                    
                    # Launch the process in background
                    python downstream/main_newmodels.py \
                        --dataset "$d" \
                        --augmentation "$aug" \
                        --pooling "$p" \
                        --nlayers_classifier "$l" \
                        --dropout_ratio "$dr" &

                    # Increment the job counter
                    ((CURRENT_JOBS++))

                    # If max parallel jobs reached, wait for some to finish
                    if ((CURRENT_JOBS >= MAX_JOBS)); then
                        wait -n  # Wait for any job to finish
                        ((CURRENT_JOBS--))
                    fi
                done
            done
        done
    done

done

# Wait for all remaining background processes to finish
wait
