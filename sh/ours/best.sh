CUDA_VISIBLE_DEVICES=0 /mlainas/teang1995/anaconda3/envs/mogp/bin/python dmtgp.py --init_lr 0.01 --fe_lr 0.01 --kernel_kr SM+L --kernel_jp SM+P --kernel_tai SM+P --num_mixture 5 --emb_dim 16 --dkl_layers 0 --MTL_method equal --case_num equal-SM+P-SM+L-SM+P --dkl --rank 2 --tolerance 1000 --max_epoch 3000 --wd 0 --model_type ours --start_date 2020-02-27 --obs_end_date 2023-02-07 --pred_end_date 2024-12-31 --device cuda --nation sub --nation_case 12 --freq 100 --eval_criterion mae --scaled_metric --temporal_conv_length_list 3 7 14