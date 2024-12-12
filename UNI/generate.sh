#!/bin/bash
# SBATCH --job-name=generate_uni_embeddings_panda
# SBATCH --output=/home/carlos/ml-project-2-ml-ts-4science/outputs/%j_%a.log
# #SBATCH --reservation=bristen
# SBATCH --cpus-per-task=32
# SBATCH --array=0-15
# SBATCH --nodes=1
# SBATCH --time=4:00:00
# #SBATCH --environment=/home/carlos/ml-project-2-ml-ts-4science/pyproject.toml
# #SBATCH --account=a02
# SBATCH -q cs433
# SBATCH --gres=gpu:2  # Solicita 2 GPUs por nodo


DATASET="BRACS"     # TCGA, GTEx
NUM_SUBJOBS=71     # TCGA: 184, GTEx: 397, BACH: 6, BRACS: 71
MAX_JOB_NUMBER=1135 #TCGA: 2942 #GTEx: 6338, BACH: 100, BRACS: 1134

# adapt to 2 gpus
# BACH: 400/4 = 100, 4 jobs per node, 100/16=6, split over 16 array jobs
# BRACS: ceil(4539/4) = 1135, 4 jobs per node, 1135/16=71, split over 16 array jobs

DATA_DIR="/mnt/lts4-pathofm/scratch/data/ml4science/$DATASET"

MAGNIFICATION="_10x" # _20x, _40x
CSV_FILE="/mnt/lts4-pathofm/scratch/data/ml4science/$DATASET/images_metadata_slurm$MAGNIFICATION.csv"

TOTAL_ROWS=$(awk 'NR>1' ${CSV_FILE} | wc -l)
HEADER=$(head -1 ${CSV_FILE})
COLUMN_INDEX=$(echo "${HEADER}" | tr ',' '\n' | awk '/^metadata_path_224$/{print NR}')

JOB_START_IDX=$((SLURM_ARRAY_TASK_ID * NUM_SUBJOBS))
JOB_END_IDX=$((JOB_START_IDX + $((NUM_SUBJOBS - 1))))
echo $JOB_START_IDX
echo $JOB_END_IDX

NUM_GPUS=1

TOTAL_JOB_COUNT=$MAX_JOB_NUMBER

for JOB_IDX in $(seq 0 $((TOTAL_JOB_COUNT - 1))); do
    START_IDX=$((JOB_IDX * 4))
    END_IDX=$((START_IDX + 3))

    if [ ${JOB_IDX} -gt ${MAX_JOB_NUMBER} ]; then
        exit 1 # si el último trabajo ha terminado, detén el script
    fi

    if [ ${END_IDX} -gt ${TOTAL_ROWS} ]; then
        END_IDX=${TOTAL_ROWS}
    fi

    # Procesa 4 filas consecutivas
    for IDX in $(seq ${START_IDX} ${END_IDX}); do
        JSON_FILE=$(awk -v idx=${IDX} -v col=${COLUMN_INDEX} -F',' 'NR==idx+2 {print $col}' ${CSV_FILE})
        CUDA_ID=$(( (IDX - START_IDX) % NUM_GPUS ))

        echo "$IDX, $JSON_FILE"
        export CUDA_VISIBLE_DEVICES=${CUDA_ID}

        python /mnt/lts4-pathofm/scratch/students/carlos/ml-project-2-ml-ts-4science/UNI/infer_uni_regions.py --data_dir $DATA_DIR --metadata_path $JSON_FILE &
    done

    wait
done
wait
