#!/usr/bin/env bash
# Slurm job configuration
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --output=s5_paral-%A_%a.out 
#SBATCH --error=s5_paral-%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --job-name=s5_paral
#SBATCH --array=0-1
#SBATCH --partition=dp-cn

module purge
module use $OTHERSTAGES
module load Stages/2023
module load GCC/11.3.0 GCCcore/.11.3.0 ParaStationMPI/5.8.0-1-mt
module load GDAL/3.5.0

source /p/project/sdlrs/tian1/jupyter/kernels/tian1_kernel_ff/bin/activate

#echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#export CUDA_VISIBLE_DEVICES=0

#export PSP_CUDA=1
#export PSP_UCP=1

##### Number of total processes
echo " "
echo " Nodelist       := " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " Ntasks per node:= " $SLURM_NTASKS_PER_NODE
echo " Ntasks         := " $SLURM_NTASKS
echo " "

echo "Running on multiple nodes"
echo ""
echo "Run started at:- "
date
srun -N 1 --ntasks-per-node=1 python -u step5_RF_liang.py --id_acquisition ${SLURM_ARRAY_TASK_ID}
echo "Run finished at:- "
date