#!/bin/bash
#SBATCH --job-name=generate_uni_embeddings_panda
#SBATCH --output=/capstor/scratch/cscs/vsubrama/slurm/logs/%x/GTEx/%j_%a.log
# #SBATCH --reservation=bristen
#SBATCH --cpus-per-task=32
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --environment=/capstor/scratch/cscs/vsubrama/edf/uni_torch_env_todi.toml
#SBATCH --account=a02

DATASET="PANDA"     # TCGA, GTEx
NUM_SUBJOBS=166     # TCGA: 184, GTEx: 397
MAX_JOB_NUMBER=2654 #TCGA: 2942 #GTEx: 6338

# PANDA: 10615/4 = 2654, 4 jobs per node, 2654/16=166, split over 16 array jobs
# TCGA: 11765/4 = 2942, 4 jobs per node, 2942/16=184, split over 16 array jobs
# GTEx: 25355/4 = 6338, 4 jobs per node, 6338/16=397, split over 16 array jobs

DATA_DIR="/capstor/scratch/cscs/vsubrama/data/$DATASET"
CSV_FILE="/capstor/scratch/cscs/vsubrama/data/$DATASET/images_metadata_slurm.csv"

TOTAL_ROWS=$(awk 'NR>1' ${CSV_FILE} | wc -l)
HEADER=$(head -1 ${CSV_FILE})
COLUMN_INDEX=$(echo "${HEADER}" | tr ',' '\n' | awk '/^metadata_path_224$/{print NR}')

JOB_START_IDX=$((SLURM_ARRAY_TASK_ID * NUM_SUBJOBS))
JOB_END_IDX=$((JOB_START_IDX + $((NUM_SUBJOBS - 1))))
echo $JOB_START_IDX
echo $JOB_END_IDX

for JOB_IDX in $(seq ${JOB_START_IDX} ${JOB_END_IDX}); do
    START_IDX=$((JOB_IDX * 4))
    END_IDX=$((START_IDX + 3))

    if [ ${JOB_IDX} -gt ${MAX_JOB_NUMBER} ]; then
        exit 1 # if the last job has finished, stop the script
    fi

    if [ ${END_IDX} -gt ${TOTAL_ROWS} ]; then
        END_IDX=${TOTAL_ROWS}
    fi

    # Extract and process 4 consecutive rows
    for IDX in $(seq ${START_IDX} ${END_IDX}); do
        JSON_FILE=$(awk -v idx=${IDX} -v col=${COLUMN_INDEX} -F',' 'NR==idx+2 {print $col}' ${CSV_FILE})
        CUDA_ID=$((IDX - START_IDX))

        echo "$IDX, $JSON_FILE"
        python /capstor/scratch/cscs/vsubrama/code/UNI/infer_uni_regions.py --data_dir $DATA_DIR --metadata_path $JSON_FILE --gpu_node ${CUDA_ID} &
    done

    wait
done
wait
