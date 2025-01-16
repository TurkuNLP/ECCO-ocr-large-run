#!/bin/bash
#SBATCH --account=project_462000615
#SBATCH --partition=standard-g
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gpus-per-node=4
#SBATCH --exclude=nid007955,nid005022,nid007957

export TRITON_HOME=/scratch/project_462000615/ecco_ocr/triton_cache/triton_cache
export TRITON_CACHE_DIR=/scratch/project_462000615/ecco_ocr/triton_cache/triton_cache

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# activate venv to use sentence_transformers, since it's not part of the pytorch module.
# If you don't use sentence_transformers, all you need is in the pytorch module.
#source ../venv/bin/activate

export TRITON_HOME=/scratch/project_462000615/ecco_ocr/triton_cache/triton_cache
export TRITON_CACHE_DIR=/scratch/project_462000615/ecco_ocr/triton_cache/triton_cache


gpu-energy --save
echo "VLLM JOB START $(date)"
srun python3 vllm_test.py --jobid $SLURM_JOB_ID --total-workers=200 --out-dir ECCO-BIG-RUN-OUT --model-name=meta-llama/Llama-3.3-70B-Instruct --max-time=71100 $*
echo "VLLM JOB END $(date)"
gpu-energy --diff
