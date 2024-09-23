import os
import wandb
import argparse
import warnings
import torch
import pandas as pd

# custom
import data
import utils
import paths

from base import baseline

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# IGNORE WARNINGS
warnings.filterwarnings(action='ignore')

# COMMON
parser = argparse.ArgumentParser(description="gp-regression for the confirmation and dead prediction")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

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


## GP
parser.add_argument("--kernel_name", type=str, default='RBF')

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
parser.add_argument("--tolerance", type=int, default=20)
parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction)

parser.add_argument("--eval_criterion", type=str, default="mae",
                help="Validation criterion (Default : MAE)")
# ---------------------------------------------------------

args = parser.parse_args()
model_type = args.model_type


utils.set_seed(args.seed)
args.device = 'cpu'
print(f"Device : {args.device}")
device = args.device
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
data_dict: nation-wise data in dictionary type - train/valid/test/pred_only
dates: dates to be used for plotting - train/valid/test/pred_only
'''
data_dict, dates = utils.get_data(nation_list, data_root, args.train_ratio, args.valid_ratio, args.start_date, args.obs_end_date, args.pred_end_date, device, args)
# ---------------------------------------------------------

# wandb setting
if not args.ignore_wandb:
    wandb.init(project='gp-regression',
               entity='mlai_medical_ai')
    
    wandb.config.update(args)
    if args.model_type == 'ARIMA':
        run_name = f'seed: {args.seed}-{args.model_type}-start_date: {args.start_date}'
    elif args.model_type == "SVR":
        run_name = f'seed: {args.seed}-{args.model_type}-start_date: {args.start_date}-gamma: {args.gamma}-C: {args.C}-epsilon: {args.epsilon}'
    else:
        run_name = f''
    wandb.run.name = run_name

# run baseline
baseline.run_baseline(model_type, args, dates, data_dict, nation_list)