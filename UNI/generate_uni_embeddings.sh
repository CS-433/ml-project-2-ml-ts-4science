#!/bin/bash
# # SBATCH --job-name=generate_uni_embeddings_panda
# # SBATCH --output=/capstor/scratch/cscs/vsubrama/slurm/logs/%x/GTEx/%j_%a.log
# # #SBATCH --reservation=bristen
# # SBATCH --cpus-per-task=32
# # SBATCH --array=0-15
# # SBATCH --nodes=1
# # SBATCH --time=4:00:00
# # SBATCH --environment=/capstor/scratch/cscs/vsubrama/edf/uni_torch_env_todi.toml
# # SBATCH --account=a02
# # SBATCH --gres=gpu:2  # Solicita 2 GPUs por nodo


#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20000
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2



DATASET="BRACS"     # TCGA, GTEx
NUM_SUBJOBS=166     # TCGA: 184, GTEx: 397
MAX_JOB_NUMBER=2654 #TCGA: 2942 #GTEx: 6338

DATA_DIR="/scratch/izar/dlopez/ml4science/data/$DATASET"
CSV_FILE="/scratch/izar/dlopez/ml4science/data/$DATASET/images_metadata_slurm.csv"

TOTAL_ROWS=$(awk 'NR>1' ${CSV_FILE} | wc -l)
HEADER=$(head -1 ${CSV_FILE})
COLUMN_INDEX=$(echo "${HEADER}" | tr ',' '\n' | awk '/^metadata_path_224$/{print NR}')

JOB_START_IDX=$((SLURM_ARRAY_TASK_ID * NUM_SUBJOBS))
JOB_END_IDX=$((JOB_START_IDX + $((NUM_SUBJOBS - 1))))
echo $JOB_START_IDX
echo $JOB_END_IDX

NUM_GPUS=2  # Número de GPUs solicitadas


### Check values of some environment variables
echo INFO: SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE
echo INFO: SLURM_JOB_GPUS=$SLURM_JOB_GPUS
echo INFO: SLURM_STEP_GPUS=$SLURM_STEP_GPUS
echo INFO: ALPHAFOLD_DIR=$ALPHAFOLD_DIR
echo INFO: ALPHAFOLD_DATADIR=$ALPHAFOLD_DATADIR
echo INFO: TMP=$TMP


for JOB_IDX in $(seq ${JOB_START_IDX} ${JOB_END_IDX}); do
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

        python /home/ricoiban/ml-project-2-ml-ts-4science/UNI/infer_uni_regions.py --data_dir $DATA_DIR --metadata_path $JSON_FILE &
    done

    wait
done
wait
