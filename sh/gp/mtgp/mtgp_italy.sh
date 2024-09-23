for tolerance in 1000
do
for kernel_name in 'SML+P'
do
for init_lr in 0.005
do
for num_mixture in 4
do
for max_epoch in 5000
do 
for rank in 1
do
for start_date in '2020-02-27'
do
python mtgp.py \
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
--nation 'italy' \
--ignore_wandb
done
done
done
done
done
done
done