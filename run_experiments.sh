#!/usr/bin/env bash

# Datasets and their respective augmentation sets
DATASETS=("BreakHis")

# Augmentations per dataset
# For BACH: 5, 10, 20
# For BRACS: 5, 10, 20, 40
AUGS_BACH=("40")
# AUGS_BRACS=("20" "40")
# AUGS_BreakHis=("20" "40")

# Pooling methods
POOLINGS=("mean" "GatedAttention")

# Classifier layers: 1 for linear, 2 for MLP
LAYERS=("1" "2")

# Dropout ratios
DROPOUTS=("0" "0.2" "0.4")

# Loop over each dataset
for d in "${DATASETS[@]}"; do
    
    # Select augmentation list based on dataset
    if [ "$d" == "BRACS" ]; then
        AUGS=("${AUGS_BRACS[@]}")
    else
        AUGS=("${AUGS_BACH[@]}")
    fi
    
    # Loop over augmentations
    for aug in "${AUGS[@]}"; do
        
        # Loop over pooling methods
        for p in "${POOLINGS[@]}"; do
            
            # Loop over layers
            for l in "${LAYERS[@]}"; do
                
                # Loop over dropout ratios
                for dr in "${DROPOUTS[@]}"; do
                    echo "Running experiment with dataset=$d, augmentation=$aug, pooling=$p, nlayers=$l, dropout=$dr"
                    python downstream/main_newmodels.py \
                        --dataset "$d" \
                        --augmentation "$aug" \
                        --pooling "$p" \
                        --nlayers_classifier "$l" \
                        --dropout_ratio "$dr"
                done
            done
        done
    done
done
