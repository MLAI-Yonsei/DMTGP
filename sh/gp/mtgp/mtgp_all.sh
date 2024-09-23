#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=cpu1
#SBATCH --nodelist=n030
##
#SBATCH --job-name=lswag-sgd-constant
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
##
##

# hostname
# date

##
# for init_lr in 0.005 0.01 0.05 0.1
for tolerance in 500
do
for kernel_name in 'SML+P' 'SM+P' 'SM+L' 'SML' 'SMPL' 'SMP'
do
for init_lr in 0.05
do
for num_mixture in 2 3 4
do
for max_epoch in 5000
do 
for rank in 1 2
do
for start_date in '2020-02-27'
do
CUDA_VISIBLE_DEVICES=1 /mlainas/teang1995/anaconda3/envs/mogp/bin/python mtgp.py \
--max_epoch ${max_epoch} \
--tolerance ${tolerance} \
--init_lr ${init_lr} \
--kernel_name ${kernel_name} \
--num_mixture ${num_mixture} \
--model_type 'MTGP' \
--rank ${rank} \
--start_date ${start_date} \
--obs_end_date '2023-02-07' \
--pred_end_date '2023-12-31' \
--device 'cuda' \
--nation 'denmark' \
--ignore_wandb
done
done
done
done
done
done
done
