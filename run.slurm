#!/bin/bash
#SBATCH --job-name=INSTALL     # job name
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:2                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=05:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=logs/%A_%a.out # output file name
#SBATCH --error=logs/%A_%a.err  # error file name

set -x
cd $WORK/Deformable-ProtoPNet/

module purge
module load pytorch-gpu/py3/1.8.1


python3 main.py \
-gpuid='0, 1' \
-m=0.1 \
-last_layer_fixed=True \
-subtractive_margin=True \
-using_deform=True \
-topk_k=1 \
-num_prototypes=1200 \
-incorrect_class_connection=-0.5 \
-deformable_conv_hidden_channels=128 \
-rand_seed=1
