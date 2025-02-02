#!/bin/bash
#SBATCH --job-name=INSTALL     # job name
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name
set -x
cd $WORK/Deformable-ProtoPNet/Deformable-Convolution-V2-PyTorch/

module purge
module load pytorch-gpu/py3/1.8.1

srun python setup.py build install
