import os
import json
import wandb
import gpytorch
import argparse
import warnings
import torch
import logging

import numpy as np
import pandas as pd

from torch import nn as nn
from matplotlib import pyplot as plt

# custom modules
import data
import metrics
import gp_models
import utils
import paths

from mtl.pcgrad import PCGrad
from mtl.weight_methods import WeightMethods

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# IGNORE WARNINGS
warnings.filterwarnings(action='ignore')

## COMMON ------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--ignore_wandb", action='store_true',
                    help = "Stop using wandb (Default : False)")
parser.add_argument("--nation", type=str, default='sub')
parser.add_argument("--plot_attn", action='store_true', default=False,
                    help = "Visualize Attention Map (Default : False)")
# --------------------------------------------------------------------------------------------------

## Data --------------------------------------------------------------------------------------------
parser.add_argument("--preprocess_data", action='store_true',
                    help = "Preprocessing data (Default : False)")
parser.add_argument("--start_date", type=str, default='2020-02-27')
parser.add_argument("--obs_end_date", type=str, default='2023-02-08')
parser.add_argument("--pred_end_date", type=str, default='2024-12-31')

parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--valid_ratio", type=float, default=0.1)

parser.add_argument("--num_variants", type=int, default=28)
parser.add_argument("--scaled_metric", action='store_true',
                    help = "scaled metric (Default : False)")
parser.add_argument("--nation_case", type=int, default=12)
# --------------------------------------------------------------------------------------------------

## Model -------------------------------------------------------------------------------------------
parser.add_argument("--model_type", type=str, default='ours', choices=['ours', 'debug'])
parser.add_argument("--num_tasks", type=int, default=2)
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--num_mixture", type=int, default=2)
parser.add_argument("--dkl_layers", type=int, default=0)
# --------------------------------------------------------------------------------------------------

## GP  ---------------------------------------------------------------------------------------------
parser.add_argument("--kernel_name", type=str, default='SM+P')
parser.add_argument("--kernel_kr", type=str, default='SM+P')
parser.add_argument("--kernel_jp", type=str, default='SM+P')
parser.add_argument("--kernel_tai", type=str, default='SM+P')
parser.add_argument("--dkl", action="store_true", default=False, help="Deep Kernel Learning")
parser.add_argument("--jitter", type=float, default=1e-6)
parser.add_argument("--prior_scale", type=float, default=1)
parser.add_argument("--rank", type=int, default=1)
# --------------------------------------------------------------------------------------------------

## Ours  -------------------------------------------------------------------------------------------
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--num_tran_layer", type=int, default=1)
parser.add_argument("--temporal_conv_length_list", nargs='+', metavar='N', 
                    type=int, default=[3], help='List type')
parser.add_argument("--ignore_cnn", action='store_true', default=False,
                    help = "ignore_cnn (Default : False)")
parser.add_argument("--ignore_transformer", action='store_true', default=False,
                    help = "ignore_transformer (Default : False)")
parser.add_argument("--layernorm", action='store_true', default=False,
                    help = "add LayerNorm module (Default : False)")
parser.add_argument("--case_num", type=str, default='none')
parser.add_argument("--mix_concat", action='store_true',
                    help = "mix concat (Default : False)")
parser.add_argument("--MTL_method",
                    default='uncert', type=str, 
                    choices=['equal', 'uncert', 'dwa', 'gradnorm', 'pcgrad'],
                    help='multi-task weighting: equal, uncert, dwa, gradnorm')
parser.add_argument("--grad_norm_alpha", type=float, default=1.5)
# --------------------------------------------------------------------------------------------------

## Training ----------------------------------------------------------------------------------------
parser.add_argument("--max_epoch", type=int, default=3000)
parser.add_argument("--init_lr", type=float, default=0.01)
parser.add_argument("--fe_lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=0.0)
parser.add_argument("--optim", type=str, default='adam')
parser.add_argument("--tolerance", type=int, default=1000)
parser.add_argument("--tol_start", type=int, default=1000)
parser.add_argument("--tol_nation", type=int, default=2)
parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)
parser.add_argument("--freq", type=int, default=-1)
parser.add_argument("--eval_criterion", type=str, default="mae",
                    help="Validation criterion (Default : MAE)")
parser.add_argument("--weights_lr", type=float, default=None)
parser.add_argument("--logsigma", type=float, default=0.5)
parser.add_argument("--temperature", type=float, default=2.0)
# --------------------------------------------------------------------------------------------------

args = parser.parse_args()
model_type = args.model_type

if args.device == 'cuda' and torch.cuda.is_available():
    device_name = 'cuda:0'
else:
    device_name = 'cpu'

device = torch.device(device_name)

print(f"Device : {device_name}")

## DATA --------------------------------------------------------------------------------------------
raw_root = paths.RAW_DATA_ROOT
prp_root = paths.DATA_ROOT
if args.preprocess_data:
    data.preprocess_data(raw_root=raw_root,
                         prp_root=prp_root,
                         start_date=args.start_date,
                         pred_end_date=args.pred_end_date,)
data_root = paths.DATA_ROOT
if args.nation == 'sub':
    nation_list = ['denmark', 'france', 'south_korea', 'uk', 'germany']
    if args.nation_case == 10:
        nation_list = ['south_korea', 'japan', 'denmark']
    elif args.nation_case == 11:
        nation_list = ['south_korea', 'japan', 'italy']
    elif args.nation_case == 12:
        nation_list = ['south_korea', 'japan', 'taiwan']
    elif args.nation_case == 13:
        nation_list = ['south_korea', 'japan', 'us']
    elif args.nation_case == 14:
        nation_list = ['south_korea', 'us', 'uk']
    else:
        """
        List 1: ['denmark', 'france', 'south_korea']
        List 2: ['denmark', 'france', 'uk']
        List 3: ['denmark', 'france', 'germany']
        List 4: ['denmark', 'south_korea', 'uk']
        List 5: ['denmark', 'south_korea', 'germany']
        List 6: ['denmark', 'uk', 'germany']
        List 7: ['france', 'south_korea', 'uk']
        List 8: ['france', 'south_korea', 'germany']
        List 9: ['france', 'uk', 'germany']
        List 10: ['south_korea', 'uk', 'germany']
        """
        from itertools import combinations
        result_lists = list(combinations(nation_list, 3))
        nation_list = result_lists[args.nation_case]
    print(f'nations = {nation_list}')
elif args.nation == 'all':
    nation_list = ['denmark', 'france', 'germany', 'italy', 
                   'japan', 'south_korea', 'taiwan', 'uk', 'us']
else:
    nation_list = [args.nation]


'''
data_dict: data of every nation
dates: train/valid/test/pred_only (necessary for plotting)
'''
data_dict, dates = utils.get_data(nation_list, data_root, args.train_ratio, args.valid_ratio, args.start_date, args.obs_end_date, args.pred_end_date, device, args)
# ---------------------------------------------------------
mae = nn.L1Loss()
rmse = metrics.RMSELoss()
mape = metrics.MAPE()
nme = metrics.NME()
# wandb setting
if not args.ignore_wandb:
    wandb.init(project='gp-regression',
               entity='mlai_medical_ai',
               group=f'{args.case_num}'
               )

    wandb.config.update(args)
    if model_type == "debug":
        run_name = f'{model_type}-{args.nation}-kernel: {args.kernel_name}-num_mixture: \
            {args.num_mixture}-wd: {args.wd}-fe_lr: {args.fe_lr}'
    else:
        run_name = f'{args.MTL_method}-JP: {args.kernel_jp}-KR: {args.kernel_kr}-TAI: {args.kernel_tai} \
        -{args.fe_lr}-{args.init_lr}-{args.num_mixture}-{args.rank}-{args.emb_dim}-{args.dkl_layers}'

    wandb.run.name = run_name

# plot root
plots_root = paths.PLOTS_ROOT
model_root = os.path.join(plots_root, args.model_type)
if args.case_num != 'none':
    model_root = os.path.join(model_root, args.case_num)
kernel_date_lr = f'{args.fe_lr}-{args.init_lr}-{args.num_mixture}-{args.rank}-{args.emb_dim}-{args.dkl_layers}'
kernel_plot_root = os.path.join(model_root, kernel_date_lr)

conv = '_'.join(str(e) for e in args.temporal_conv_length_list)
if args.MTL_method == "uncert":
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}')
elif args.MTL_method == "dwa":
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}')
else:
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}')

logsigma = None
weights_lr = args.weights_lr if args.weights_lr is not None else 1e-4
if args.MTL_method == 'gradnorm':
    weights_lr = 0.025
    logging.info("For GradNorm the default lr for weights is 0.025, like in the GradNorm paper")

if args.MTL_method in ["gradnorm", "uncert", "dwa"]:
    weighting_method = WeightMethods(
        method=args.MTL_method,
        n_tasks=len(nation_list),
        alpha=args.grad_norm_alpha,
        temp=args.temperature,
        n_train_batch=1,
        n_epochs=args.max_epoch,
        device=device
    )

# weight root
weight_root = paths.WEIGHTS_ROOT
model_root = os.path.join(weight_root, args.model_type)
if args.case_num != 'none':
    model_root = os.path.join(model_root, args.case_num)
kernel_weight_root = os.path.join(model_root, kernel_date_lr)
if args.MTL_method == "uncert":
    kernel_weight_root = os.path.join(kernel_weight_root, f'{conv}-{args.logsigma}')
elif args.MTL_method == "dwa":
    kernel_weight_root = os.path.join(kernel_weight_root, f'{conv}-{args.temperature}')
else:
    kernel_weight_root = os.path.join(kernel_weight_root, f'{conv}')

print("plot saved at :", kernel_plot_root)
print("weight saved at :", kernel_weight_root)

# test set performance list
conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []
type_list = ['conf', 'dead']

utils.set_seed(args.seed)

if not args.ignore_transformer:
    shared_feature_extractor = gp_models.TSFeatureExtractor(args, device).to(device)

# last shared layer for GradNorm
def get_last_shared_layer():
    last_shared_layer = shared_feature_extractor.multiheadattention.out_proj.parameters()
    return list(last_shared_layer)

if not args.ignore_transformer:
    names = [n for n, p in shared_feature_extractor.named_parameters()]
    if args.MTL_method == 'gradnorm':
        # add weights to optimizer
        param_groups = [
            {'params': shared_feature_extractor.parameters()},
            {'params': weighting_method.method.weights, 'lr': weights_lr}
        ]
        fe_optimizer = torch.optim.Adam(param_groups, lr=weights_lr, weight_decay=args.wd)

    elif args.MTL_method == 'uncert':
        # add weights to optimizer
        non_logsigma_params = [p for n, p in shared_feature_extractor.named_parameters() if n != 'logsigma']
        param_groups = [
                {'params': non_logsigma_params},
                {'params': shared_feature_extractor.logsigma, 'lr': weights_lr}
        ]
        fe_optimizer = torch.optim.Adam(param_groups, lr=args.fe_lr, weight_decay=args.wd)
    else:
        param_groups = [
                {'params': shared_feature_extractor.parameters()}
        ]
        fe_optimizer = torch.optim.Adam(param_groups, lr=args.fe_lr, weight_decay=args.wd)
        if args.MTL_method == 'pcgrad':
            fe_optimizer = PCGrad(fe_optimizer)

# Data Preparation
train_x_list = []; val_x_list = []; test_x_list = []
train_y_list = []; val_y_list = []; test_y_list = []

pred_x_list = []
minmax_list = []

for nation in nation_list:
    # meta data for min-max scaling
    meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
    meta_df = pd.read_csv(meta_df_path)
    min_confirmation = meta_df['min_confirmation'][0]
    max_confirmation = meta_df['max_confirmation'][0]
    min_dead = meta_df['min_dead'][0]
    max_dead = meta_df['max_dead'][0]

    min_value_list = [min_confirmation, min_dead]
    max_value_list = [max_confirmation, max_dead]
    minmax_list.append((min_value_list, max_value_list))

    # plot root
    nation_plot_root = os.path.join(kernel_plot_root, nation)
    nation_weight_root = os.path.join(kernel_weight_root, nation)
    os.makedirs(nation_weight_root, exist_ok=True)

    train_x = data_dict[nation]['train_x']
    train_y = data_dict[nation]['train_y']
    valid_x = data_dict[nation]['valid_x']
    valid_y = data_dict[nation]['valid_y']
    test_x = data_dict[nation]['test_x']
    test_y = data_dict[nation]['test_y']
    pred_x = data_dict[nation]['pred_x']

    train_x_list.append(train_x)
    train_y_list.append(train_y)
    val_x_list.append(valid_x)
    val_y_list.append(valid_y)
    test_x_list.append(test_x)
    test_y_list.append(test_y)
    pred_x_list.append(pred_x)

# total : 1769
total_train_x = torch.stack(train_x_list)   # 860 : 2020-02-27 ~ 2022-07-05
total_valid_x = torch.stack(val_x_list)     # 107 : 2020-07-06 ~ 2022-10-20
total_test_x = torch.stack(test_x_list)     # 109 : 2022-10-21 ~ 2023-02-06
# total 1076
total_pred_x = torch.stack(pred_x_list)     # 693 : 2023-02-07 ~ 2024-12-30

# Train Preparation
model_list = []
likelihood_list = []
optimizer_list = []
scheduler_list = []
mll_list = []
for i, nation in enumerate(nation_list):
    train_x = total_train_x[i]
    train_y = train_y_list[i]
    valid_x = total_valid_x
    valid_y = val_y_list[i]
    test_x = total_test_x
    test_y = test_y_list[i]
    
    if nation=='japan':
        kernel = args.kernel_jp
    elif nation=='taiwan':
        kernel = args.kernel_tai
    else:
        kernel = args.kernel_kr

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=args.num_tasks)
    model = gp_models.ExactGPModel(train_x=train_x.to(device), 
                                    train_y=train_y.to(device), 
                                    likelihood=likelihood, 
                                    args=args, kernel=kernel)
    model.to(device)
    likelihood.to(device)
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch) 

    performance_dict = {
        'conf-mae': 0,
        'conf-rmse': 0,
        'conf-mape': 0,
        'conf-nme': 0,
        'dead-mae': 0,
        'dead-rmse': 0,
        'dead-mape': 0,
        'dead-nme': 0,
    }
    all_nation_performance_dict = {}
    model_list.append(model)
    likelihood_list.append(likelihood)
    optimizer_list.append(optimizer)
    scheduler_list.append(scheduler)
    mll_list.append(mll)
tolerance_cnt = 0
best_crit = [999999999999999] * len(nation_list)
best_loss = 999999999999999; best_var = 999999999999999
best_mae = float('inf')

# training
for epoch in range(args.max_epoch):
    if not args.ignore_transformer:
        fe_optimizer.zero_grad()
    best_flag = False
    mll_val = 0; mae_val = 0
    total_train_loss = 0; total_valid_loss = 0; total_valid_mae = 0
    total_conf_std = 0; total_dead_std = 0
    val_conf_mae = []; val_conf_rmse = []
    val_dead_mae = []; val_dead_rmse = []
    test_conf_mae = []; test_conf_rmse = []
    test_dead_mae = []; test_dead_rmse = []

    nation_task_dict = {nation: {} for nation in nation_list}
    mll_value = 0 # sum of valid_mae of all nation
    criterions = [0] * len(nation_list)
    if not args.ignore_transformer:
        if args.MTL_method == 'uncert':
            shared_train_feature, logsigma, attn_tr = shared_feature_extractor(total_train_x)
            shared_valid_feature, _ = shared_feature_extractor(total_valid_x)
            shared_test_feature, _ = shared_feature_extractor(total_test_x)
            shared_pred_feature, _ = shared_feature_extractor(total_pred_x)
        else:
            shared_train_feature, attn_tr = shared_feature_extractor(total_train_x)
            shared_valid_feature = shared_feature_extractor(total_valid_x)
            shared_test_feature = shared_feature_extractor(total_test_x)
            shared_pred_feature = shared_feature_extractor(total_pred_x)
    loss_list = []
    valid_loss_list = []
    cur_var = 0

    for i, nation in enumerate(nation_list):
        model = model_list[i]
        likelihood = likelihood_list[i]
        mll = mll_list[i]
        train_y = train_y_list[i]
        valid_y = val_y_list[i]
        test_y = test_y_list[i]
        
        # TRAIN
        model.train()
        likelihood.train()
        optimizer_list[i].zero_grad()
        train_x = total_train_x[i]
        if not args.ignore_transformer:
            model.shared_emb = shared_train_feature[i]
        
        pred_train = model(train_x) 
        train_loss = -mll(pred_train, train_y)
        loss_list.append(train_loss)

        train_loss.backward(retain_graph=True)
        total_train_loss += train_loss.item()
        
        # VALIDATION / TEST / PRED
        train_x = total_train_x[i]
        valid_x = total_valid_x[i]
        test_x = total_test_x[i]
        pred_x = total_pred_x[i]
        
        model.eval()
        likelihood.eval()
        
        # save model prediction for train/valid/test
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
            if not args.ignore_transformer:
                model.shared_emb = shared_train_feature[i]
            model_train = likelihood(model(train_x))
            # model_train = model(train_x)
            if not args.ignore_transformer:
                model.shared_emb = torch.cat([shared_train_feature[i], shared_valid_feature[i]], axis=0)
                model_valid = likelihood(model(valid_x))
                model.shared_emb = torch.cat([shared_train_feature[i], shared_test_feature[i]], axis=0)
                model_test = likelihood(model(test_x))
                model.shared_emb = torch.cat([shared_train_feature[i], shared_pred_feature[i]], axis=0)
                model_pred = likelihood(model(pred_x))
            else:
                model_valid = likelihood(model(valid_x))
                model_test = likelihood(model(test_x))
                model_pred = likelihood(model(pred_x))
        
        train_mean = model_train.mean
        valid_mean = model_valid.mean
        test_mean = model_test.mean
        pred_mean = model_pred.mean    
        task_dict = {'conf': {}, 'dead': {}}
        min_value_list = minmax_list[i][0]
        max_value_list = minmax_list[i][1]
        
        # CR criterion
        cur_var += (model_train.variance[:, 0].detach().cpu().numpy() + model_train.variance[:, 1].detach().cpu().numpy())
        
        for j, type_ in enumerate(type_list):
            min_value = min_value_list[j]
            max_value = max_value_list[j]
            
            ori_train_mean = train_mean[:, j] * (max_value - min_value) + min_value
            ori_valid_mean = valid_mean[:, j] * (max_value - min_value) + min_value
            ori_test_mean = test_mean[:, j] * (max_value - min_value) + min_value
            ori_pred_mean = pred_mean[:, j] * (max_value - min_value) + min_value
            
            task_train_mean = model_train.mean[:, j]
            task_valid_mean = model_valid.mean[:, j]
            task_test_mean = model_test.mean[:, j]
            task_pred_mean = model_pred.mean[:, j]
            
            task_ori_train_y = train_y[:, j] * (max_value - min_value) + min_value
            task_ori_valid_y = valid_y[:, j] * (max_value - min_value) + min_value
            task_ori_test_y = test_y[:, j] * (max_value - min_value) + min_value
            
            task_dict[type_]['train_variance'] = model_train.variance[:, j].detach().cpu().numpy()
            task_dict[type_]['valid_variance'] = model_valid.variance[:, j].detach().cpu().numpy()
            task_dict[type_]['test_variance'] = model_test.variance[:, j].detach().cpu().numpy()
            task_dict[type_]['pred_variance'] = model_pred.variance[:, j].detach().cpu().numpy()
            
            task_dict[type_]['min_value'] = min_value
            task_dict[type_]['max_value'] = max_value
            
            task_dict[type_]['model_train'] = model_train
            task_dict[type_]['model_valid'] = model_valid
            task_dict[type_]['model_test'] = model_test
            task_dict[type_]['model_pred'] = model_pred
            
            task_dict[type_]['train_mean'] = train_mean[:, j]
            task_dict[type_]['valid_mean'] = valid_mean[:, j]
            task_dict[type_]['test_mean'] = test_mean[:, j]
            task_dict[type_]['pred_mean'] = pred_mean[:, j]
            
            task_dict[type_]['ori_train_mean'] = ori_train_mean
            task_dict[type_]['ori_valid_mean'] = ori_valid_mean
            task_dict[type_]['ori_test_mean'] = ori_test_mean
            task_dict[type_]['ori_pred_mean'] = ori_pred_mean
            
            task_dict[type_]['ori_train_y'] = task_ori_train_y
            task_dict[type_]['ori_valid_y'] = task_ori_valid_y
            task_dict[type_]['ori_test_y'] = task_ori_test_y
            
            task_dict[type_]['train_y'] = train_y[:, j]
            task_dict[type_]['valid_y'] = valid_y[:, j]
            task_dict[type_]['test_y'] = test_y[:, j]
        
        # VALIDATION
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-2):
            if not args.ignore_transformer:
                model.shared_emb = torch.cat([shared_train_feature[i], shared_valid_feature[i]], axis=0)
            mll_value += -mll(model(valid_x), valid_y)
            
            mae_value = 0
            for j, type_ in enumerate(type_list):
                if args.scaled_metric:
                    task_valid_y = task_dict[type_]['valid_y']
                    task_valid_mean = task_dict[type_]['valid_mean']
                else:
                    task_valid_y = task_dict[type_]['ori_valid_y']
                    task_valid_mean = task_dict[type_]['ori_valid_mean']
                # clipping
                task_valid_mean = torch.clamp(task_valid_mean, min=0)
                mae_value += mae(task_valid_y, task_valid_mean)
            valid_loss_list.append(mae_value.item())
        
        total_valid_mae += mae_value 
        total_conf_std += torch.std_mean(valid_mean[valid_mean.shape[0]//2:,0])[0]
        total_dead_std += torch.std_mean(valid_mean[valid_mean.shape[0]//2:,1])[0]
        nation_task_dict[nation] = task_dict
        
        if args.eval_criterion == 'mae':
            criterions[i] = mae_value
        else:
            criterions[i] = -mll(model(valid_x), valid_y).item()
        
        # TEST
        performance_dict = {}
        
        for j, type_ in enumerate(type_list):
            task_test_y = task_dict[type_]['test_y']
            task_test_mean = task_dict[type_]['test_mean']
            task_ori_test_y = task_dict[type_]['ori_test_y']
            task_ori_test_mean = task_dict[type_]['ori_test_mean']
            # clipping
            task_test_mean = torch.clamp(task_test_mean, min=0)
            task_ori_test_mean = torch.clamp(task_ori_test_mean, min=0)
            mae_loss, rmse_loss, mape_loss, nme_loss = \
                metrics.get_all_metrics([mae, rmse, mape, nme], task_test_y, task_test_mean)
            ori_mae_loss, ori_rmse_loss, ori_mape_loss, ori_nme_loss = \
                metrics.get_all_metrics([mae, rmse, mape, nme], task_ori_test_y, task_ori_test_mean)
            
            performance_dict[f'{type_}-mae'] = mae_loss.item()
            performance_dict[f'{type_}-rmse'] = rmse_loss.item()
            performance_dict[f'{type_}-mape'] = mape_loss.item()
            performance_dict[f'{type_}-nme'] = nme_loss.item()
            
            performance_dict[f'ori_{type_}-mae'] = ori_mae_loss.item()
            performance_dict[f'ori_{type_}-rmse'] = ori_rmse_loss.item()
            performance_dict[f'ori_{type_}-mape'] = ori_mape_loss.item()
            performance_dict[f'ori_{type_}-nme'] = ori_nme_loss.item()
            
            if j == 0:
                test_conf_mae.append(ori_mae_loss)
            else:
                test_dead_mae.append(ori_mae_loss)
        
        if not args.ignore_wandb:
            wandb.log({f'{nation}-mae' : performance_dict['conf-mae']+performance_dict['dead-mae']})
        
        if nation not in all_nation_performance_dict:
            all_nation_performance_dict[nation] = {}
            
        all_nation_performance_dict[nation].update(performance_dict)
        
        # logging
        conf_mae = all_nation_performance_dict[nation]['conf-mae']
        conf_rmse = all_nation_performance_dict[nation]['conf-rmse']
        dead_mae = all_nation_performance_dict[nation]['dead-mae']
        dead_rmse = all_nation_performance_dict[nation]['dead-rmse']
        val_conf_mae.append(conf_mae); val_conf_rmse.append(conf_rmse)
        val_dead_mae.append(dead_mae); val_dead_rmse.append(dead_rmse)
    
    if not args.ignore_wandb:
        wandb.log({'total_train_loss' : total_train_loss})
        wandb.log({'total_valid_mae' : total_valid_mae.item()})
        wandb.log({'total_valid_loss' : mll_value})
    
    # optim step
    if not args.ignore_transformer:
        if args.MTL_method == 'pcgrad':
            fe_optimizer.pc_backward(loss_list)
        if args.MTL_method in ['uncert', 'gradnorm', 'DWA']:
            last_shared_layer = get_last_shared_layer() if args.MTL_method == 'gradnorm' else None
            mtl_loss = weighting_method.backwards(
                loss_list,
                epoch=epoch,
                logsigmas=logsigma,
                last_shared_params=last_shared_layer,
                returns=True
            )
        
        fe_optimizer.step()
    
    for i, _ in enumerate(nation_list):
        optimizer_list[i].step()
        scheduler_list[i].step()     
    
    # best 확인 및 갱신
    if args.eval_criterion == 'mae':
        cur_loss = total_valid_mae
    else:
        cur_loss = mll_value
    
    conf_mask = total_conf_std < 1e-3
    dead_mask = total_dead_std < 1e-3
    
    if conf_mask or dead_mask:
        args.tolerance += 5
    
    if (not cur_loss.isnan()) and (cur_loss < best_loss) and (not conf_mask) and (not dead_mask):
        best_loss = cur_loss
        best_var = cur_var
        best_crit = criterions
        best_flag = True
        tolerance_cnt = 0
    
    else:
        if epoch >= args.tol_start:
            tolerance_cnt +=1 
    logging = f' Iter {epoch+1}/{args.max_epoch} - Train Loss: {total_train_loss:.3f} Valid MAE: {total_valid_mae:.3f} Valid mll: {mll_value:.3f} Test conf MAE: {sum(test_conf_mae):.3f} Test dead MAE: {sum(test_dead_mae):.3f} Tol: {tolerance_cnt}'
    
    if best_flag: # update best loss
        logging += ' *'
    print(logging)

    if tolerance_cnt >= args.tolerance:
        break
    
    if args.freq != -1 and epoch % args.freq == 0:
        for i, nation in enumerate(nation_list):
            best_task_dict = nation_task_dict[nation]
            # nation_performance_dict = all_nation_performance_dict[nation]
            min_value_list = minmax_list[i][0]
            max_value_list = minmax_list[i][1]
            # visualize
            for j, type_ in enumerate(type_list):
                min_value = min_value_list[j]
                max_value = max_value_list[j]
                with torch.no_grad():
                    train_y = best_task_dict[type_]['train_y'].detach().cpu()
                    valid_y_cpu = best_task_dict[type_]['valid_y'].detach().cpu()
                    test_y = best_task_dict[type_]['test_y'].detach().cpu()
                    
                    ori_train_y = best_task_dict[type_]['ori_train_y'].detach().cpu()
                    ori_valid_y_cpu = best_task_dict[type_]['ori_valid_y'].detach().cpu()
                    ori_test_y = best_task_dict[type_]['ori_test_y'].detach().cpu()
                    
                    train_mean = best_task_dict[type_]['train_mean'].detach().cpu()
                    valid_mean = best_task_dict[type_]['valid_mean'].detach().cpu()
                    test_mean = best_task_dict[type_]['test_mean'].detach().cpu()
                    pred_mean = best_task_dict[type_]['pred_mean'].detach().cpu()
                    
                    ori_train_mean = best_task_dict[type_]['ori_train_mean'].detach().cpu()
                    ori_valid_mean = best_task_dict[type_]['ori_valid_mean'].detach().cpu()
                    ori_test_mean = best_task_dict[type_]['ori_test_mean'].detach().cpu()
                    ori_pred_mean = best_task_dict[type_]['ori_pred_mean'].detach().cpu()
                    
                    train_lower, train_upper = best_task_dict[type_]['model_train'].confidence_region()
                    valid_lower, valid_upper = best_task_dict[type_]['model_valid'].confidence_region()
                    test_lower, test_upper = best_task_dict[type_]['model_test'].confidence_region()
                    pred_lower, pred_upper = best_task_dict[type_]['model_pred'].confidence_region()
                    
                    min_value = min_value_list[j]
                    max_value = max_value_list[j]
                    
                    ori_train_lower = train_lower[:, j] * (max_value - min_value) + min_value
                    ori_train_upper = train_upper[:, j] * (max_value - min_value) + min_value
                    ori_valid_lower = valid_lower[:, j] * (max_value - min_value) + min_value
                    ori_valid_upper = valid_upper[:, j] * (max_value - min_value) + min_value
                    ori_test_lower = test_lower[:, j] * (max_value - min_value) + min_value
                    ori_test_upper = test_upper[:, j] * (max_value - min_value) + min_value
                    ori_pred_lower = pred_lower[:, j] * (max_value - min_value) + min_value
                    ori_pred_upper = pred_upper[:, j] * (max_value - min_value) + min_value
                    
                    train_lower = train_lower[:, j]
                    train_upper = train_upper[:, j]
                    valid_lower = valid_lower[:, j]
                    valid_upper = valid_upper[:, j]
                    test_lower = test_lower[:, j]
                    test_upper = test_upper[:, j]
                    pred_lower = pred_lower[:, j]
                    pred_upper = pred_upper[:, j]
                
                # clipping
                train_mean = torch.clamp(train_mean, min=0)
                valid_mean = torch.clamp(valid_mean, min=0)
                test_mean = torch.clamp(test_mean, min=0)
                pred_mean = torch.clamp(pred_mean, min=0)
                
                ori_train_mean = torch.clamp(ori_train_mean, min=0)
                ori_valid_mean = torch.clamp(ori_valid_mean, min=0)
                ori_test_mean = torch.clamp(ori_test_mean, min=0)
                ori_pred_mean = torch.clamp(ori_pred_mean, min=0)
                
                mae_loss, rmse_loss, mape_loss, nme_loss = \
                    metrics.get_all_metrics([mae, rmse, mape, nme], ori_test_y, ori_test_mean)
                
                lower_list = [ori_train_lower, ori_valid_lower, ori_test_lower, ori_pred_lower]
                lower_list = [x.detach().cpu() for x in lower_list]
                upper_list = [ori_train_upper, ori_valid_upper, ori_test_upper, ori_pred_upper]
                upper_list = [x.detach().cpu() for x in upper_list]
                y_list = [ori_train_y, ori_valid_y_cpu, ori_test_y]
                all_mean = np.concatenate([ori_train_mean, ori_valid_mean, ori_test_mean, ori_pred_mean])
                nation_plot_root = os.path.join(kernel_plot_root, nation)
                utils.plot_gp(all_mean, upper_list, lower_list, y_list, nation, nation_plot_root, max_value, type_, dates, f'{epoch}-', args)
    
    if best_flag and not args.ignore_wandb:
        wandb.run.summary['best epoch'] = epoch
        
        all_conf_mae = 0
        all_dead_mae = 0
        all_ori_conf_mae = 0
        all_ori_dead_mae = 0
        for nation in nation_list:
            for j, type_ in enumerate(type_list):
                best_mae = all_nation_performance_dict[nation][f'{type_}-mae']
                best_rmse = all_nation_performance_dict[nation][f'{type_}-rmse']
                best_mape = all_nation_performance_dict[nation][f'{type_}-mape']
                best_nme = all_nation_performance_dict[nation][f'{type_}-nme']
                
                best_ori_mae = all_nation_performance_dict[nation][f'ori_{type_}-mae']
                best_ori_rmse = all_nation_performance_dict[nation][f'ori_{type_}-rmse']
                best_ori_mape = all_nation_performance_dict[nation][f'ori_{type_}-mape']
                best_ori_nme = all_nation_performance_dict[nation][f'ori_{type_}-nme']
                wandb.run.summary[f'{nation}-{type_}-mae'] = best_mae
                wandb.run.summary[f'{nation}-{type_}-rmse'] = best_rmse
                wandb.run.summary[f'{nation}-{type_}-mape'] = best_mape
                wandb.run.summary[f'{nation}-{type_}-nme'] = best_nme
                
                wandb.run.summary[f'ori_{nation}-{type_}-mae'] = best_ori_mae
                wandb.run.summary[f'ori_{nation}-{type_}-rmse'] = best_ori_rmse
                wandb.run.summary[f'ori_{nation}-{type_}-mape'] = best_ori_mape
                wandb.run.summary[f'ori_{nation}-{type_}-nme'] = best_ori_nme
                if type_ == 'conf':
                    all_conf_mae += best_mae
                    all_ori_conf_mae += best_ori_mae
                else:
                    all_dead_mae += best_mae
                    all_ori_dead_mae += best_ori_mae
        
        wandb.log({
            'best-conf-mae': all_conf_mae,
            'best-dead-mae': all_dead_mae,
            'best-sum-mae': all_conf_mae + all_dead_mae,
            'best-ori-conf-mae': all_ori_conf_mae,
            'best-ori-dead-mae': all_ori_dead_mae,
            'best-ori-sum-mae': all_ori_conf_mae + all_ori_dead_mae
        })
    
    if best_flag:
        import copy
        nation_best_task_dict = copy.deepcopy(nation_task_dict)
        model_weight_path = os.path.join(kernel_weight_root, 'gp_weight.pt')
        model_weight = model.state_dict()
        torch.save(model_weight, model_weight_path)
        if not args.ignore_transformer:
            fe_weight_path = os.path.join(kernel_weight_root, "fe_weight.pt")
            fe_weight = shared_feature_extractor.state_dict()
            torch.save(fe_weight, fe_weight_path)

for i, nation in enumerate(nation_list):
    nation_weight_root = os.path.join(kernel_weight_root, nation)
    best_task_dict = nation_best_task_dict[nation]
    # nation_performance_dict = all_nation_performance_dict[nation]
    min_value_list = minmax_list[i][0]
    max_value_list = minmax_list[i][1]
    # visualize
    for j, type_ in enumerate(type_list):
        min_value = min_value_list[j]
        max_value = max_value_list[j]
        with torch.no_grad():
            train_y = best_task_dict[type_]['train_y'].detach().cpu()
            valid_y_cpu = best_task_dict[type_]['valid_y'].detach().cpu()
            test_y = best_task_dict[type_]['test_y'].detach().cpu()
            
            ori_train_y = best_task_dict[type_]['ori_train_y'].detach().cpu()
            ori_valid_y_cpu = best_task_dict[type_]['ori_valid_y'].detach().cpu()
            ori_test_y = best_task_dict[type_]['ori_test_y'].detach().cpu()
            
            train_mean = best_task_dict[type_]['train_mean'].detach().cpu()
            valid_mean = best_task_dict[type_]['valid_mean'].detach().cpu()
            test_mean = best_task_dict[type_]['test_mean'].detach().cpu()
            pred_mean = best_task_dict[type_]['pred_mean'].detach().cpu()
            
            ori_train_mean = best_task_dict[type_]['ori_train_mean'].detach().cpu()
            ori_valid_mean = best_task_dict[type_]['ori_valid_mean'].detach().cpu()
            ori_test_mean = best_task_dict[type_]['ori_test_mean'].detach().cpu()
            ori_pred_mean = best_task_dict[type_]['ori_pred_mean'].detach().cpu()
            
            train_lower, train_upper = best_task_dict[type_]['model_train'].confidence_region()
            valid_lower, valid_upper = best_task_dict[type_]['model_valid'].confidence_region()
            test_lower, test_upper = best_task_dict[type_]['model_test'].confidence_region()
            pred_lower, pred_upper = best_task_dict[type_]['model_pred'].confidence_region()
            
            min_value = min_value_list[j]
            max_value = max_value_list[j]
            
            ori_train_lower = train_lower[:, j] * (max_value - min_value) + min_value
            ori_train_upper = train_upper[:, j] * (max_value - min_value) + min_value
            ori_valid_lower = valid_lower[:, j] * (max_value - min_value) + min_value
            ori_valid_upper = valid_upper[:, j] * (max_value - min_value) + min_value
            ori_test_lower = test_lower[:, j] * (max_value - min_value) + min_value
            ori_test_upper = test_upper[:, j] * (max_value - min_value) + min_value
            ori_pred_lower = pred_lower[:, j] * (max_value - min_value) + min_value
            ori_pred_upper = pred_upper[:, j] * (max_value - min_value) + min_value
            train_lower = train_lower[:, j]
            train_upper = train_upper[:, j]
            valid_lower = valid_lower[:, j]
            valid_upper = valid_upper[:, j]
            test_lower = test_lower[:, j]
            test_upper = test_upper[:, j]
            pred_lower = pred_lower[:, j]
            pred_upper = pred_upper[:, j]
            
            train_variance = best_task_dict[type_]['train_variance']
            valid_variance = best_task_dict[type_]['valid_variance']
            test_variance = best_task_dict[type_]['test_variance']
            pred_variance = best_task_dict[type_]['pred_variance']
        
        # clipping
        train_mean = torch.clamp(train_mean, min=0)
        valid_mean = torch.clamp(valid_mean, min=0)
        test_mean = torch.clamp(test_mean, min=0)
        pred_mean = torch.clamp(pred_mean, min=0)
        
        ori_train_mean = torch.clamp(ori_train_mean, min=0)
        ori_valid_mean = torch.clamp(ori_valid_mean, min=0)
        ori_test_mean = torch.clamp(ori_test_mean, min=0)
        ori_pred_mean = torch.clamp(ori_pred_mean, min=0)
        
        mae_loss, rmse_loss, mape_loss, nme_loss = \
            metrics.get_all_metrics([mae, rmse, mape, nme], ori_test_y, ori_test_mean)
        lower_list = [ori_train_lower, ori_valid_lower, ori_test_lower, ori_pred_lower]
        lower_list = [x.detach().cpu() for x in lower_list]
        upper_list = [ori_train_upper, ori_valid_upper, ori_test_upper, ori_pred_upper]
        upper_list = [x.detach().cpu() for x in upper_list]
        ori_y_list = [ori_train_y, ori_valid_y_cpu, ori_test_y]
        
        model_pred_path = os.path.join(nation_weight_root, f'{type_}.json')
        variance_path = os.path.join(nation_weight_root, f'{type_}_var.npy')
        
        pred_dict = {
            'nation': nation,
            'dates':  [x.to_dict() for x in dates],
            'min_conf': int(min_confirmation),
            'max_conf': int(max_confirmation),
            'min_dead': int(min_dead),
            'max_dead': int(max_dead),
            'train_mean': train_mean.tolist(),
            'valid_mean': valid_mean.tolist(),
            'test_mean': test_mean.tolist(),
            'pred_mean': pred_mean.tolist(),
            'ori_train_mean': ori_train_mean.tolist(),
            'ori_valid_mean': ori_valid_mean.tolist(),
            'ori_test_mean': ori_test_mean.tolist(),
            'ori_pred_mean': ori_pred_mean.tolist(),
            'lower_list': [x.numpy().tolist() for x in lower_list],
            'upper_list': [x.numpy().tolist() for x in upper_list],
            'ori_y_list': [x.numpy().tolist() for x in ori_y_list], 
            'strat_date': args.start_date,
            'pred_end_date': args.pred_end_date
        }
        
        variance_dict = {
            'train_variance': train_variance,
            'valid_variance': valid_variance,
            'test_variance': test_variance,
            'pred_variance': pred_variance
        }
        with open(model_pred_path, 'w') as fp:
            json.dump(pred_dict, fp)
        
        np.save(variance_path, variance_dict)
        all_mean = np.concatenate([ori_train_mean, ori_valid_mean, ori_test_mean, ori_pred_mean])
        nation_plot_root = os.path.join(kernel_plot_root, nation)
        utils.plot_gp(all_mean, upper_list, lower_list, ori_y_list, nation, nation_plot_root, max_value, type_, dates, 'best-', args)