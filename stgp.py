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
parser.add_argument("--model_type", type=str, default='STGP', choices=['MTGP', 'STGP', 'ARIMA', 'SVR', 'LINEAR', 'POLYNOMIAL'])
parser.add_argument("--data_dim", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=2)
parser.add_argument("--num_mixture", type=int, default=2)

## GP
parser.add_argument("--kernel_name", type=str, default='RBF')
parser.add_argument("--dkl", action="store_true", default=False, help="Deep Kernel Learning")
parser.add_argument("--jitter", type=float, default=1e-6)
# ---------------------------------------------------------

## Training ------------------------------------------------
parser.add_argument("--max_epoch", type=int, default=5000)
parser.add_argument("--init_lr", type=float, default=0.1)
parser.add_argument("--optim", type=str, default='adam')
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--tolerance", type=int, default=20)
parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)
parser.add_argument("--freq", type=int, default=10)
parser.add_argument("--eval_criterion", type=str, default="mae",
                help="Validation criterion (Default : MAE)")
# ---------------------------------------------------------

args = parser.parse_args()
model_type = args.model_type


utils.set_seed(args.seed)
if args.device == 'cuda' and torch.cuda.is_available():
    device_name = 'cuda:0'
else:
    device_name = 'cpu'

device = torch.device(device_name)
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

if not args.ignore_wandb:
    wandb.init(project='gp-regression',
               entity='mlai_medical_ai')
    
    wandb.config.update(args)
    
    if args.dkl:
        dkl = "dkl"
    else:
        dkl = "no-dkl"

    if args.model_type in ['MTGP', 'STGP']:
        if dkl == 'dkl':
            run_name = f'seed: {args.seed}-{args.model_type}({dkl})-kernel: {args.kernel_name}-num_mixture: {args.num_mixture}-emb_dim: {args.emb_dim}-max_epoch: {args.max_epoch}-tolerance: {args.tolerance}-init_lr: {args.init_lr}-dropout: {args.dropout}-start_date: {args.start_date}'
            
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
kernel_date_lr = f'{args.kernel_name}-{args.start_date}-{args.init_lr}-{args.num_mixture}'
kernel_root = os.path.join(model_root, kernel_date_lr)

weight_root = paths.WEIGHTS_ROOT
model_root = os.path.join(weight_root, args.model_type)
kernel_date_lr = f'{args.kernel_name}-{args.start_date}-{args.init_lr}-{args.num_mixture}'
kernel_weight_root = os.path.join(model_root, kernel_date_lr)

conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []
type_list = ['conf', 'dead']

train_date = dates[0]
valid_date = dates[1]
test_date = dates[2]
pred_date = dates[3]
all_date = pd.concat([train_date, valid_date, test_date, pred_date]).reset_index(drop=True)

for nation_idx, nation in enumerate(nation_list):
    nation_plot_root = os.path.join(kernel_root, nation)
    nation_weight_root = os.path.join(kernel_weight_root, nation)
    os.makedirs(nation_weight_root, exist_ok=True)

    utils.set_seed(args.seed)
    for j, type_ in enumerate(type_list):      
        best_loss = 99999999999999999999

        train_x = data_dict[nation]['train_x']
        train_y = data_dict[nation]['train_y']
        valid_x = data_dict[nation]['valid_x']
        valid_y = data_dict[nation]['valid_y']
        test_x = data_dict[nation]['test_x']
        test_y = data_dict[nation]['test_y']
        pred_x = data_dict[nation]['pred_x']

        train_y = train_y[:, j]
        valid_y = valid_y[:, j]
        test_y = test_y[:, j]

        meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
        meta_df = pd.read_csv(meta_df_path)
        min_confirmation = meta_df['min_confirmation'][0]
        max_confirmation = meta_df['max_confirmation'][0]
        min_dead = meta_df['min_dead'][0]
        max_dead = meta_df['max_dead'][0]
        if type_ == 'conf':
            min_value = min_confirmation
            max_value = max_confirmation
        else:
            min_value = min_dead
            max_value = max_dead 

        ori_train_y = train_y * (max_value - min_value) + min_value
        ori_valid_y = valid_y * (max_value - min_value) + min_value
        ori_test_y = test_y * (max_value - min_value) + min_value
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = models.ExactGPModel(train_x, train_y, likelihood, args)
        model.to(device)
        likelihood.to(device)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch) 
        
        best_mae = 0
        best_rmse = 0
        best_mape = 0
        best_nme = 0

        for i in range(args.max_epoch):
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

            ori_train_mean = train_mean * (max_value - min_value) + min_value
            ori_valid_mean = valid_mean * (max_value - min_value) + min_value
            ori_test_mean = test_mean * (max_value - min_value) + min_value
            ori_pred_mean = pred_mean * (max_value - min_value) + min_value

            # Clipping
            train_mean = torch.clamp(train_mean, min=0)
            valid_mean = torch.clamp(valid_mean, min=0)
            test_mean = torch.clamp(test_mean, min=0)
            pred_mean = torch.clamp(pred_mean, min=0)
            
            ori_train_mean = torch.clamp(ori_train_mean, min=0)
            ori_valid_mean = torch.clamp(ori_valid_mean, min=0)
            ori_test_mean = torch.clamp(ori_test_mean, min=0)
            ori_pred_mean = torch.clamp(ori_pred_mean, min=0)
            # mae 
            with torch.no_grad(), gpytorch.settings.fast_pred_var(),gpytorch.settings.cholesky_jitter(1e-2):
                mll_value = -mll(model_valid, valid_y).item()
                mae_value = mae(ori_valid_y, ori_valid_mean).item()
            if args.eval_criterion == 'mae':
                cur_loss = mae_value
            else:
                cur_loss = mll_value

            if cur_loss < best_loss:
                best_loss = cur_loss

                mae_loss, rmse_loss, mape_loss, nme_loss = \
                    metrics.get_all_metrics([mae, rmse, mape, nme], ori_test_y, ori_test_mean)
                best_mae = mae_loss
                best_rmse = rmse_loss
                best_mape = mape_loss
                best_nme = nme_loss

                if not args.ignore_wandb:
                    wandb.run.summary[f'{nation}-{type_}-best epoch'] = i
                    wandb.run.summary[f'{nation}-{type_}-mae'] = best_mae
                    wandb.run.summary[f'{nation}-{type_}-rmse'] = best_rmse
                    wandb.run.summary[f'{nation}-{type_}-mape'] = best_mape
                    wandb.run.summary[f'{nation}-{type_}-nme'] = best_nme

                best_weight = model.state_dict()
                best_flag = False
            
            model.train()
            likelihood.train()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print(f'Iter {i+1}/{args.max_epoch} - Loss: {loss.item()}')
            optimizer.step()
            scheduler.step()

        model.load_state_dict(best_weight)
        likelihood = model.likelihood
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

            # variance
            train_variance = model_train.variance.detach().cpu().numpy()
            valid_variance = model_valid.variance.detach().cpu().numpy()
            test_variance = model_test.variance.detach().cpu().numpy()
            pred_variance = model_pred.variance.detach().cpu().numpy()
            # mean값 역변환
            ori_train_mean = train_mean * (max_value - min_value) + min_value
            ori_valid_mean = valid_mean * (max_value - min_value) + min_value
            ori_test_mean = test_mean * (max_value - min_value) + min_value
            ori_pred_mean = pred_mean * (max_value - min_value) + min_value

            # Clipping
            train_mean = torch.clamp(train_mean, min=0)
            valid_mean = torch.clamp(valid_mean, min=0)
            test_mean = torch.clamp(test_mean, min=0)
            pred_mean = torch.clamp(pred_mean, min=0)
            
            ori_train_mean = torch.clamp(ori_train_mean, min=0)
            ori_valid_mean = torch.clamp(ori_valid_mean, min=0)
            ori_test_mean = torch.clamp(ori_test_mean, min=0)
            ori_pred_mean = torch.clamp(ori_pred_mean, min=0)
            if args.device == 'cuda':
                train_mean = train_mean.detach().cpu().numpy()
                valid_mean = valid_mean.detach().cpu().numpy()
                test_mean = test_mean.detach().cpu().numpy()
                pred_mean = pred_mean.detach().cpu().numpy()

                ori_train_y = ori_train_y.detach().cpu().numpy()
                ori_valid_y_cpu = ori_valid_y.detach().cpu().numpy()
                ori_test_y = ori_test_y.detach().cpu().numpy()

                ori_train_mean = ori_train_mean.detach().cpu().numpy()
                ori_valid_mean = ori_valid_mean.detach().cpu().numpy()
                ori_test_mean = ori_test_mean.detach().cpu().numpy()
                ori_pred_mean = ori_pred_mean.detach().cpu().numpy()

            all_mean = np.concatenate([ori_train_mean, ori_valid_mean, ori_test_mean, ori_pred_mean])
            
            # confidence region fill
            train_lower, train_upper = model_train.confidence_region()
            valid_lower, valid_upper = model_valid.confidence_region()
            test_lower, test_upper = model_test.confidence_region()
            pred_lower, pred_upper = model_pred.confidence_region()

            if args.device == 'cuda':
                train_lower = train_lower.detach().cpu()
                train_upper = train_upper.detach().cpu()

                valid_lower = valid_lower.detach().cpu()
                valid_upper = valid_upper.detach().cpu()

                test_lower = test_lower.detach().cpu()
                test_upper = test_upper.detach().cpu()
                pred_lower = pred_lower.detach().cpu()
                pred_upper = pred_upper.detach().cpu()

            ori_train_lower = train_lower * (max_value - min_value) + min_value
            ori_train_upper = train_upper * (max_value - min_value) + min_value
            ori_valid_lower = valid_lower * (max_value - min_value) + min_value
            ori_valid_upper = valid_upper * (max_value - min_value) + min_value
            ori_test_lower = test_lower * (max_value - min_value) + min_value
            ori_test_upper = test_upper * (max_value - min_value) + min_value
            ori_pred_lower = pred_lower * (max_value - min_value) + min_value
            ori_pred_upper = pred_upper * (max_value - min_value) + min_value

            if args.device == 'cuda':
                ori_train_lower = ori_train_lower.numpy()
                ori_valid_lower = ori_valid_lower.numpy()
                ori_test_lower = ori_test_lower.numpy()
                ori_pred_lower = ori_pred_lower.numpy()

                ori_train_upper = ori_train_upper.numpy()
                ori_valid_upper = ori_valid_upper.numpy()
                ori_test_upper = ori_test_upper.numpy()
                ori_pred_upper = ori_pred_upper.numpy()
                
            lower_list = [ori_train_lower, ori_valid_lower, ori_test_lower, ori_pred_lower]
            lower_list = [x.tolist() for x in lower_list]
            upper_list = [ori_train_upper, ori_valid_upper, ori_test_upper, ori_pred_upper]
            upper_list = [x.tolist() for x in upper_list]

            ori_y_list = [ori_train_y, ori_valid_y_cpu, ori_test_y]
            all_mean = np.concatenate([ori_train_mean, ori_valid_mean, ori_test_mean, ori_pred_mean])
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
                    'lower_list': lower_list,
                    'upper_list': upper_list,
                    'ori_y_list': [x.tolist() for x in ori_y_list], 
                    'strat_date': args.start_date,
                    'pred_end_date': args.pred_end_date
                }
            variance_dict = {
                'train_variance': train_variance,
                'valid_variance': valid_variance,
                'test_variance': test_variance,
                'pred_variance': pred_variance
            }

        nation_save_root = f'./{nation}'
        os.makedirs(nation_save_root, exist_ok=True)
        model_pred_path = os.path.join(nation_weight_root, f'{type_}.json')
        variance_path = os.path.join(nation_weight_root, f'{type_}_var.npy')
        with open(model_pred_path, 'w') as fp:
            json.dump(pred_dict, fp)
        
        np.save(variance_path, variance_dict)
        utils.plot_gp(all_mean, upper_list, lower_list, ori_y_list, nation, nation_plot_root, max_value, type_, dates, 'best-', args)

        if type_ == 'conf':
            conf_mae_list.append(best_mae)
            conf_rmse_list.append(best_rmse)
            conf_mape_list.append(best_mape)
            conf_nme_list.append(best_nme)

        else:
            dead_mae_list.append(best_mae)
            dead_rmse_list.append(best_rmse)
            dead_mape_list.append(best_mape)
            dead_nme_list.append(best_nme)
        
conf_mae = sum(conf_mae_list) / len(conf_mae_list)
conf_rmse = sum(conf_rmse_list) / len(conf_rmse_list)
conf_mape = sum(conf_mape_list) / len(conf_mape_list)
conf_nme = sum(conf_nme_list) / len(conf_nme_list)

print(f"Confirmation MAE : {conf_mae}")
print(f"Confirmation RMSE : {conf_rmse}")
print(f"Confirmation MAPE : {conf_mape}")
print(f"Confirmation NME : {conf_nme}")

dead_mae = sum(dead_mae_list) / len(dead_mae_list)
dead_rmse = sum(dead_rmse_list) / len(dead_rmse_list)
dead_mape = sum(dead_mape_list) / len(dead_mape_list)
dead_nme = sum(dead_nme_list) / len(dead_nme_list) 

print(f"Dead MAE {dead_mae}")
print(f"Dead RMSE {dead_rmse}")
print(f"Dead MAPE {dead_mape}")
print(f"Dead NME {dead_nme}")

if not args.ignore_wandb:
    wandb_dict = {
        f"conf_mae": conf_mae,
        f"conf_rmse": conf_rmse,
        f"conf_mape": conf_mape,
        f"conf_nme": conf_nme,
        f"dead_mae": dead_mae,
        f"dead_rmse": dead_rmse,
        f"dead_mape": dead_mape,
        f"dead_nme": dead_nme,
    }
    wandb.log(wandb_dict)

