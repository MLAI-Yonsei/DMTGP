for tolerance in 500
do
for num_mixture in 2 3 4
do
for kernel_name in 'SpectralMixture' 'SML+P' 'SM+P' 'SM+L' 'SML' 'SMP'
do
for init_lr in 0.01
do
for max_epoch in 5000
do 
for rank in 1 2
do
for start_date in '2021-01-01'
do
CUDA_VISIBLE_DEVICES=6 /mlainas/teang1995/anaconda3/envs/mogp/bin/python mtgp.py \
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
--nation 'france' \
# --ignore_wandb
done
done
done
done
done
done
done