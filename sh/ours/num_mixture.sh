# 추가 스윕 범위 : fe_lr / MTL method
for num_mixture in 4 3 2
do
for rank in 2
do
for emb_dim in 16
do
for dkl_layers in 0
do
for init_lr in 0.01
do
for fe_lr in 0.01
do
for cnn in '3 7 14' #'7 14 30'
do
for MTL_method in 'equal' # 'dwa' 'uncert' 'gradnorm' 'pcgrad'
do
CUDA_VISIBLE_DEVICES=1 /mlainas/teang1995/anaconda3/envs/mogp/bin/python dmtgp.py \
--init_lr ${init_lr} \
--fe_lr ${fe_lr} \
--kernel_kr 'SM+L' \
--kernel_jp 'SM+P' \
--kernel_tai 'SM+P' \
--num_mixture ${num_mixture} \
--emb_dim ${emb_dim} \
--dkl_layers ${dkl_layers} \
--MTL_method ${MTL_method} \
--case_num 'sensitivity_fe_lr' \
--dkl \
--rank ${rank} \
--tolerance 1000 \
--max_epoch 3000 \
--wd 0 \
--model_type 'ours' \
--start_date '2020-02-27' \
--obs_end_date '2023-02-07' \
--pred_end_date '2024-12-31' \
--device 'cuda' \
--nation 'sub' \
--nation_case 12 \
--freq 100 \
--eval_criterion 'mae' \
--scaled_metric \
--temporal_conv_length_list ${cnn} \
# --ignore_wandb
done
done
done
done
done
done
done
done
