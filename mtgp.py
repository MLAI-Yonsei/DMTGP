import os
import json
import wandb
import gpytorch
import argparse
import warnings
import torch

import numpy as np
import pandas as pd

from torch import nn as nn
from matplotlib import pyplot as plt

# custom
import data
import metrics
import models
import utils
import paths

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# IGNORE WARNINGS
warnings.filterwarnings(action='ignore')

# COMMON
parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
parser.add_argument("--nation", type=str, default='all')
## Data -------------------------------------------------
parser.add_argument("--preprocess_data", action='store_true',
        help = "Preprocessing data (Default : False)")
parser.add_argument("--start_date", type=str, default='2020-02-27')
parser.add_argument("--obs_end_date", type=str, default='2023-02-08')
parser.add_argument("--pred_end_date", type=str, default='2023-12-31')

parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--valid_ratio", type=float, default=0.1)

parser.add_argument("--num_variants", type=int, default=28)
# -------------------------------------------------------

## Model -------------------------------------------------
parser.add_argument("--model_type", type=str, default='MTGP', choices=['MTGP', 'STGP', 'ARIMA', 'SVR', 'LINEAR', 'POLYNOMIAL'])
parser.add_argument("--num_tasks", type=int, default=2)
parser.add_argument("--data_dim", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--num_mixture", type=int, default=2)

## GP
parser.add_argument("--kernel_name", type=str, default='RBF')
parser.add_argument("--dkl", action="store_true", default=False, help="Deep Kernel Learning")
parser.add_argument("--jitter", type=float, default=1e-6)
parser.add_argument("--prior_scale", type=float, default=1)
parser.add_argument("--rank", type=int, default=1)

## ARIMA
parser.add_argument("--max_p", type=int, default=3, help="search range for p (Default : 3)")
parser.add_argument("--max_d", type=int, default=1, help="search range for d (Default : 1)")
parser.add_argument("--max_q", type=int, default=3, help="search range for q (Default : 3)")

## SVR
parser.add_argument("--gamma", type=list, default=[1e-4, 1e-3, 0.01, 0.1, 1, 10] ,
            help="Kernel coefficient for rbf kernel")
parser.add_argument("--C", type=list, default=[1e-4, 1e-3, 0.01, 0.1, 1, 10],
            help="Regularization parameter. The strength of the regularization is inversely proportional to C. \
            Must be strictly positive. The penalty is a squared l2 penalty.")
parser.add_argument("--epsilon", type=float, default=[1e-4, 1e-3, 0.01, 0.1, 1, 10],
            help="Epsilon in the epsilon-SVR model. \
                It specifies the epsilon-tube within which no penalty is associated in the training loss function \
                with points predicted wisthin a distance epsilon from the actual value. Must be non-negative.")
# ---------------------------------------------------------

## Training ------------------------------------------------
parser.add_argument("--max_epoch", type=int, default=5000)
parser.add_argument("--init_lr", type=float, default=0.1)
parser.add_argument("--optim", type=str, default='adam')
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--tolerance", type=int, default=500)
parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)
parser.add_argument("--freq", type=int, default=-1)
parser.add_argument("--eval_criterion", type=str, default="mae",
                help="Validation criterion (Default : MAE)")
# ---------------------------------------------------------

args = parser.parse_args()
model_type = args.model_type

if args.device == 'cuda' and torch.cuda.is_available():
    device_name = 'cuda:0'
else:
    device_name = 'cpu'
# device_name = 'cpu'
device = torch.device(device_name)
# args.device = 'cpu'
print(f"Device : {device_name}")

## DATA ----------------------------------------------------
raw_root = paths.RAW_DATA_ROOT
prp_root = paths.DATA_ROOT
if args.preprocess_data:
    data.preprocess_data(raw_root=raw_root,
                         prp_root=prp_root,
                         start_date=args.start_date,
                         pred_end_date=args.pred_end_date,)
data_root = paths.DATA_ROOT
nation_list = ['denmark', 'france', 'germany', 'italy', 
               'japan', 'south_korea', 'taiwan', 'uk', 'us']

'''
data_dict: data of every nation
dates: train/valid/test/pred_only (necessary for plotting)
'''
data_dict, dates = utils.get_data(nation_list, data_root, args.train_ratio, args.valid_ratio, args.start_date, args.obs_end_date, args.pred_end_date, device, args)
# ---------------------------------------------------------

if args.nation != 'all':
     nation_list = [args.nation]

mae = nn.L1Loss()
rmse = metrics.RMSELoss()
mape = metrics.MAPE()
nme = metrics.NME()

"""
wandb.init(project='gp-regression',
            entity='mlai_medical_ai',
        #    group=f'sweep-{args.eval_criterion}'
            group=f'{args.case_num}'
            )
"""
if args.dkl:
    dkl = "dkl"
else:
    dkl = "no-dkl"

if not args.ignore_wandb:
    wandb.init(project='gp-regression',
               entity='mlai_medical_ai',
               group='real-final-MTGP')
    wandb.config.update(args)

    run_name = f'seed: {args.seed}-{args.model_type}-kernel: {args.kernel_name}-num_mixture: \
    {args.num_mixture}-max_epoch: {args.max_epoch}-tolerance: {args.tolerance}-\
    init_lr: {args.init_lr}-start_date: {args.start_date}-nation: {args.nation}'

    if args.model_type in ['MTGP', 'STGP']:
        if dkl == 'dkl':
            run_name = f'seed: {args.seed}-{args.model_type}({dkl})-kernel: {args.kernel_name}-num_mixture: {args.num_mixture}-emb_dim: {args.emb_dim}-max_epoch: {args.max_epoch}-tolerance: {args.tolerance}-init_lr: {args.init_lr}-start_date: {args.start_date}'
            
        elif dkl == 'no-dkl':
            run_name = f'seed: {args.seed}-{args.model_type}({dkl})-kernel: {args.kernel_name}-num_mixture: {args.num_mixture}-max_epoch: {args.max_epoch}-tolerance: {args.tolerance}-init_lr: {args.init_lr}-start_date: {args.start_date}-nation: {args.nation}'

    elif args.model_type == 'ARIMA':
        run_name = f'seed: {args.seed}-{args.model_type}-start_date: {args.start_date}'
    elif args.model_type == "SVR":
        run_name = f'seed: {args.seed}-{args.model_type}-start_date: {args.start_date}-gamma: {args.gamma}-C: {args.C}-epsilon: {args.epsilon}'
    elif args.model_type in ["POLYNOMIAL", "LINEAR"]:
        run_name = f'seed: {args.seed}-{args.model_type}-start_date: {args.start_date}'
    else:
        run_name = f''
    wandb.run.name = run_name

plots_root = paths.PLOTS_ROOT
model_root = os.path.join(plots_root, args.model_type)
kernel_date_lr = f'{args.kernel_name}-{args.rank}-{args.start_date}-{args.init_lr}-{args.num_mixture}'
kernel_plot_root = os.path.join(model_root, kernel_date_lr, dkl)

weight_root = paths.WEIGHTS_ROOT
model_root = os.path.join(weight_root, args.model_type)
kernel_date_lr = f'{args.kernel_name}-{args.rank}-{args.start_date}-{args.init_lr}-{args.num_mixture}'
kernel_weight_root = os.path.join(model_root, kernel_date_lr, dkl)

print("plot saved at :", kernel_plot_root)
print("weight saved at :", kernel_weight_root)

# test set performance list for every country
conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []
type_list = ['conf', 'dead']

train_date = dates[0]
valid_date = dates[1]
test_date = dates[2]
pred_date = dates[3]
all_date = pd.concat([train_date, valid_date, test_date, pred_date]).reset_index(drop=True)

for nation_idx, nation in enumerate(nation_list):
    utils.set_seed(args.seed)

    meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
    meta_df = pd.read_csv(meta_df_path)
    min_confirmation = meta_df['min_confirmation'][0]
    max_confirmation = meta_df['max_confirmation'][0]
    min_dead = meta_df['min_dead'][0]
    max_dead = meta_df['max_dead'][0]

    min_value_list = [min_confirmation, min_dead]
    max_value_list = [max_confirmation, max_dead]

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

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    model = models.ExactGPModel(train_x, train_y, likelihood, args)
    model.to(torch.float64).to(device)
    likelihood.to(torch.float64).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch) 

    best_loss = 999999999999999
    tolerance_cnt = 0
    
    conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
    dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []
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
    if not args.eval_only:
        for epoch in range(args.max_epoch):
            best_flag = False

            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                model_train = likelihood(model(train_x))
                model_valid = likelihood(model(valid_x))
                model_test = likelihood(model(test_x))
                model_pred = likelihood(model(pred_x))

            train_mean = model_train.mean
            valid_mean = model_valid.mean
            test_mean = model_test.mean
            pred_mean = model_pred.mean

            task_dict = {
                'conf': {},
                'dead': {}
                }

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

            with torch.no_grad(), gpytorch.settings.fast_pred_var(),gpytorch.settings.cholesky_jitter(1e-2):
                mll_value = 0
                mll_value += -mll(model_valid, valid_y)

                mae_value = 0
                for j, type_ in enumerate(type_list):
                    task_valid_y = task_dict[type_]['valid_y']
                    task_valid_mean = task_dict[type_]['valid_mean'] 
                    mae_value += mae(task_valid_y, task_valid_mean)

                    task_valid_mean = torch.clamp(task_valid_mean, min=0)

                if not args.ignore_wandb:
                    wandb.log({'valid_loss (mll)' : mll_value})

            
            if args.eval_criterion == 'mae':
                cur_loss = mae_value
            else:
                cur_loss = mll_value

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_flag = True
                tolerance_cnt = 0

            else:
                tolerance_cnt +=1 

            if tolerance_cnt == args.tolerance:
                break

            if best_flag and not args.ignore_wandb:
                wandb.run.summary[f'{nation}-best epoch'] = epoch
                sum_mae = 0
                ori_sum_mae = 0
                for j, type_ in enumerate(type_list):
                    task_ori_test_y = task_dict[type_]['ori_test_y']
                    task_ori_test_mean = task_dict[type_]['ori_test_mean']
                    task_test_y = task_dict[type_]['test_y']
                    task_test_mean = task_dict[type_]['test_mean']

                    # clipping
                    task_test_mean = torch.clamp(task_test_mean, min=0)
                    task_ori_test_mean = torch.clamp(task_ori_test_mean, min=0)

                    ori_mae_loss, ori_rmse_loss, ori_mape_loss, ori_nme_loss = \
                        metrics.get_all_metrics([mae, rmse, mape, nme], task_ori_test_y, task_ori_test_mean)
                    mae_loss, rmse_loss, mape_loss, nme_loss = \
                        metrics.get_all_metrics([mae, rmse, mape, nme], task_test_y, task_test_mean)
                        
                    ori_best_mae = ori_mae_loss
                    ori_best_rmse = ori_rmse_loss
                    ori_best_mape = ori_mape_loss
                    ori_best_nme = ori_nme_loss

                    best_mae = mae_loss
                    best_rmse = rmse_loss
                    best_mape = mape_loss
                    best_nme = nme_loss
                    
                    sum_mae += mae_loss
                    ori_sum_mae += ori_mae_loss
                    if j == 0:
                        conf_mae_loss = ori_mae_loss
                    else:
                        dead_mae_loss = ori_mae_loss
                    wandb.run.summary[f'ori_{nation}-{type_}-mae'] = ori_best_mae
                    wandb.run.summary[f'ori_{nation}-{type_}-rmse'] = ori_best_rmse
                    wandb.run.summary[f'ori_{nation}-{type_}-mape'] = ori_best_mape
                    wandb.run.summary[f'ori_{nation}-{type_}-nme'] = ori_best_nme
                    
                    wandb.run.summary[f'{nation}-{type_}-mae'] = best_mae
                    wandb.run.summary[f'{nation}-{type_}-rmse'] = best_rmse
                    wandb.run.summary[f'{nation}-{type_}-mape'] = best_mape
                    wandb.run.summary[f'{nation}-{type_}-nme'] = best_nme

                wandb.run.summary[f'{nation}-sum-mae'] = sum_mae
                wandb.run.summary[f'{nation}-ori-sum-mae'] = ori_sum_mae

            if best_flag:
                import copy
                best_task_dict = copy.deepcopy(task_dict)

                model_weight_path = os.path.join(nation_weight_root, 'weight.pt')
                model_weight = model.state_dict()
                torch.save(model_weight, model_weight_path)

            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            train_loss = loss.item()
            if not args.ignore_wandb:
                wandb.log({'train_loss' : loss.item(),})
            loss.backward()
            logging = f' Iter {epoch+1}/{args.max_epoch} - Train Loss: {train_loss:.3f} Valid MAE: {mae_value:.3f} Valid mll: {mll_value:.3f} Test conf MAE: {conf_mae_loss:.3f} Test dead MAE: {dead_mae_loss:.3f}'

            if best_flag: 
                logging += ' *'
            print(logging)
            optimizer.step()
            scheduler.step()

            # save visualization for every `freq` epoch
            if args.freq != -1 and (epoch+1) % args.freq == 0:
                for j, type_ in enumerate(type_list):
                    with torch.no_grad():
                        ori_train_y = task_dict[type_]['ori_train_y'].detach().cpu()
                        ori_valid_y_cpu = task_dict[type_]['ori_valid_y'].detach().cpu()
                        ori_test_y = task_dict[type_]['ori_test_y'].detach().cpu()

                        ori_train_mean = task_dict[type_]['ori_train_mean'].detach().cpu()
                        ori_valid_mean = task_dict[type_]['ori_valid_mean'].detach().cpu()
                        ori_test_mean = task_dict[type_]['ori_test_mean'].detach().cpu()
                        ori_pred_mean = task_dict[type_]['ori_pred_mean'].detach().cpu()

                        train_lower, train_upper = task_dict[type_]['model_train'].confidence_region()
                        valid_lower, valid_upper = task_dict[type_]['model_valid'].confidence_region()
                        test_lower, test_upper = task_dict[type_]['model_test'].confidence_region()
                        pred_lower, pred_upper = task_dict[type_]['model_pred'].confidence_region()

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

                    # clipping
                    train_mean = torch.clamp(train_mean, min=0)
                    valid_mean = torch.clamp(valid_mean, min=0)
                    test_mean = torch.clamp(test_mean, min=0)
                    pred_mean = torch.clamp(pred_mean, min=0)
                    
                    ori_train_mean = torch.clamp(ori_train_mean, min=0)
                    ori_valid_mean = torch.clamp(ori_valid_mean, min=0)
                    ori_test_mean = torch.clamp(ori_test_mean, min=0)
                    ori_pred_mean = torch.clamp(ori_pred_mean, min=0)
                    lower_list = [ori_train_lower, ori_valid_lower, ori_test_lower, ori_pred_lower]
                    lower_list = [x.detach().cpu() for x in lower_list]
                    upper_list = [ori_train_upper, ori_valid_upper, ori_test_upper, ori_pred_upper]
                    upper_list = [x.detach().cpu() for x in upper_list]
                    ori_y_list = [ori_train_y, ori_valid_y_cpu, ori_test_y]
                    all_mean = np.concatenate([ori_train_mean, ori_valid_mean, ori_test_mean, ori_pred_mean])

                    utils.plot_gp(all_mean, upper_list, lower_list, ori_y_list, nation, nation_plot_root, max_value, type_, dates, epoch, args)

    else:
        model_weight_path = os.path.join(nation_weight_root, 'weight.pt')
        model_weight = torch.load(model_weight_path)
        model.load_state_dict(model_weight)
        model.eval()
        likelihood.eval()

        # save model prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_train = likelihood(model(train_x))
            model_valid = likelihood(model(valid_x))
            model_test = likelihood(model(test_x))
            model_pred = likelihood(model(pred_x))

        train_mean = model_train.mean
        valid_mean = model_valid.mean
        test_mean = model_test.mean
        pred_mean = model_pred.mean

        # save confidenace region value
        task_dict = {
            'conf': {},
            'dead': {}
            }

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

            task_dict[type_]['min_value'] = min_value
            task_dict[type_]['max_value'] = max_value
            task_dict[type_]['model_train'] = model_train
            task_dict[type_]['model_valid'] = model_valid
            task_dict[type_]['model_test'] = model_test
            task_dict[type_]['model_pred'] = model_pred
            task_dict[type_]['valid_mean'] = valid_mean[:, j]
            task_dict[type_]['ori_train_mean'] = ori_train_mean
            task_dict[type_]['ori_valid_mean'] = ori_valid_mean
            task_dict[type_]['ori_test_mean'] = ori_test_mean
            task_dict[type_]['ori_pred_mean'] = ori_pred_mean
            task_dict[type_]['ori_train_y'] = task_ori_train_y
            task_dict[type_]['ori_valid_y'] = task_ori_valid_y
            task_dict[type_]['ori_test_y'] = task_ori_test_y

        best_task_dict = task_dict

    # visualize
    for j, type_ in enumerate(type_list):
        with torch.no_grad():
            ori_train_y = best_task_dict[type_]['ori_train_y'].detach().cpu()
            ori_valid_y_cpu = best_task_dict[type_]['ori_valid_y'].detach().cpu()
            ori_test_y = best_task_dict[type_]['ori_test_y'].detach().cpu()

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
        performance_dict[f'{type_}-mae'] = mae_loss
        performance_dict[f'{type_}-rmse'] = rmse_loss
        performance_dict[f'{type_}-mape'] = mape_loss
        performance_dict[f'{type_}-nme'] = nme_loss

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
        utils.plot_gp(all_mean, upper_list, lower_list, ori_y_list, nation, nation_plot_root, max_value, type_, dates, 'best-', args)

    if not args.ignore_wandb:
        wandb.log(performance_dict)
