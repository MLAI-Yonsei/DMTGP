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
parser.add_argument("--nation", type=str, default='all')
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
parser.add_argument("--nation_case", type=int, default=-1)
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

# plot 결과 저장할 경로 지정
plots_root = paths.PLOTS_ROOT
model_root = os.path.join(plots_root, args.model_type)
if args.case_num != 'none':
    model_root = os.path.join(model_root, args.case_num)
kernel_date_lr = f'{args.fe_lr}-{args.init_lr}-{args.num_mixture}-{args.rank}-{args.emb_dim}-{args.dkl_layers}'
kernel_plot_root = os.path.join(model_root, kernel_date_lr)

conv = '_'.join(str(e) for e in args.temporal_conv_length_list)
if args.MTL_method == "uncert":
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}-{args.logsigma}')
elif args.MTL_method == "dwa":
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}-{args.temperature}')
else:
    kernel_plot_root = os.path.join(kernel_plot_root, f'{conv}')

# root to save weight
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
model_weight_path = os.path.join(kernel_weight_root, 'gp_weight.pt')
fe_weight_path = os.path.join(kernel_weight_root, "fe_weight.pt")

shared_feature_extractor = gp_models.TSFeatureExtractor(args, device).to(device)
names = [n for n, p in shared_feature_extractor.named_parameters()]
print(names)

# Data Preparation
train_x_list = []; val_x_list = []; test_x_list = []
train_y_list = []; val_y_list = []; test_y_list = []

pred_x_list = []
minmax_list = []

for nation in nation_list:
    # min-max scaling의 역변환을 위한 국가 별 meta data 호출 (최대 / 최소 확진자 / 사망자)
    meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
    meta_df = pd.read_csv(meta_df_path)
    min_confirmation = meta_df['min_confirmation'][0]
    max_confirmation = meta_df['max_confirmation'][0]
    min_dead = meta_df['min_dead'][0]
    max_dead = meta_df['max_dead'][0]

    min_value_list = [min_confirmation, min_dead]
    max_value_list = [max_confirmation, max_dead]
    minmax_list.append((min_value_list, max_value_list))

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
total_pred_x = torch.stack(pred_x_list)     # 693 : 2023-02-07 ~ 2024-12-30

utils.set_seed(args.seed)

# Visualize attention map
shared_feature_extractor.load_state_dict(torch.load(fe_weight_path))

if args.MTL_method == 'uncert':
    shared_train_feature, logsigma, attn_tr = shared_feature_extractor(total_train_x)
    shared_valid_feature, _, attn_val = shared_feature_extractor(total_valid_x)
    shared_test_feature, _, attn_te = shared_feature_extractor(total_test_x)
    shared_pred_feature, _, attn_pred = shared_feature_extractor(total_pred_x)
else:
    shared_train_feature, attn_tr = shared_feature_extractor(total_train_x)
    shared_valid_feature, attn_val = shared_feature_extractor(total_valid_x)
    shared_test_feature, attn_te = shared_feature_extractor(total_test_x)
    shared_pred_feature, attn_pred = shared_feature_extractor(total_pred_x)

attention_weights = attn_tr[0][1]  # shape: [860, 3, 3]

# Specify the orders for which we want to plot the attention maps
# selected indices
orders = [716, 722, 723, 724, 725, 726, 728]

# Create a figure to hold all subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# Iterate through the orders and plot each attention map
for idx, order in enumerate(orders):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # Extract the attention map for the specific order
    attention_map = attention_weights[order - 1].detach().cpu().numpy()
    
    # Plot the attention map
    cax = ax.matshow(attention_map, cmap='viridis')
    # annotate
    for (i, j), val in np.ndenumerate(attention_map):
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color='white')
    
    ax.set_title(f'{order}th date')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')

    # Adding color bar for each subplot
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

# # Add a single colorbar for all subplots
# cax = ax.matshow(attention_map, cmap='viridis', vmin=vmin, vmax=vmax)
# fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.6)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
attn_plot_root = os.path.join(kernel_plot_root, 'attention_maps_7.png')
# attn_plot_root = os.path.join(kernel_plot_root, 'attention_maps_14.png')
print(f'Attention map saved at {attn_plot_root}')
plt.savefig(attn_plot_root)