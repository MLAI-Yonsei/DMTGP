for tolerance in 500
do
for init_lr in 0.005
do
for kernel_name in 'SML+P'
do
for max_epoch in 5000
do 
for num_mixture in 3
do
for rank in 2
do
<<<<<<< HEAD
for start_date in '2020-02-27'
do
CUDA_VISIBLE_DEVICES=6 /mlainas/teang1995/anaconda3/envs/mogp/bin/python mtgp.py \
--max_epoch ${max_epoch} \
--tolerance ${tolerance} \
--init_lr ${init_lr} \
--kernel_name ${kernel_name} \
--num_mixture ${num_mixture} \
--model_type 'MTGP' \
--start_date ${start_date} \
--rank ${rank} \
--obs_end_date '2023-02-07' \
--pred_end_date '2023-12-31' \
--device 'cuda' \
--nation 'us' \
<<<<<<< HEAD
=======
--dkl \
>>>>>>> 7046e07c533b08d26ccb976a24b9631d4774359b
--ignore_wandb
done
done
done
done
done
done
done