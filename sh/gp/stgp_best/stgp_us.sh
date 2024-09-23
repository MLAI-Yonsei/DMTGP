for tolerance in 1000
do
for init_lr in 0.01
do
for kernel_name in 'SM+P'
do
for max_epoch in 5000
do 
for num_mixture in 4
do
for start_date in '2020-02-27'
do
CUDA_VISIBLE_DEVICES=7 /mlainas/teang1995/anaconda3/envs/mogp/bin/python stgp.py \
--max_epoch ${max_epoch} \
--tolerance ${tolerance} \
--init_lr ${init_lr} \
--kernel_name ${kernel_name} \
--num_mixture ${num_mixture} \
--model_type 'STGP' \
--start_date ${start_date} \
--obs_end_date '2023-02-07' \
--pred_end_date '2023-12-31' \
--device 'cuda' \
--nation 'us'
done
done
done
done
done
done

for tolerance in 1000
do
for init_lr in 0.01
do
for kernel_name in 'SMPL'
do
for max_epoch in 5000
do 
for num_mixture in 3
do
for start_date in '2020-02-27'
do
CUDA_VISIBLE_DEVICES=7 /mlainas/teang1995/anaconda3/envs/mogp/bin/python stgp.py \
--max_epoch ${max_epoch} \
--tolerance ${tolerance} \
--init_lr ${init_lr} \
--kernel_name ${kernel_name} \
--num_mixture ${num_mixture} \
--model_type 'STGP' \
--start_date ${start_date} \
--obs_end_date '2023-02-07' \
--pred_end_date '2023-12-31' \
--device 'cuda' \
--nation 'us'
done
done
done
done
done
done