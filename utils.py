import os
import random
import torch
import wandb
import gpytorch
import numpy as np
import pandas as pd
import torch.nn as nn

from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

import models
import paths
import utils

def get_weights(args, num_case, bucket_weights):
    sp = num_case.shape
    depth = depth.view(-1).cpu().numpy()
    assert depth.dtype == np.float32
    weights = np.array(list(map(lambda v: bucket_weights[get_bin_idx(args, v)], num_case)))
    weights = torch.tensor(weights, dtype=torch.float32).view(*sp)

    return weights

def get_bin_idx(args, x):
    return min(int(x * args.num_bins), args.num_bins - 1)

def set_seed(random_seed=1000):
    '''
    Set Seed for Reproduction
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_x_and_y(df):
    confirm = torch.tensor(df[['confirmation']].to_numpy())
    dead = torch.tensor(df[['dead']].to_numpy())
    cols = df.columns.tolist()
    cols.remove('date')
    cols.remove('confirmation')
    cols.remove('dead')
    
    cols = ['inoculation', 'temperature', 'humidity', 'precipitation', 'stringency', 'is_holiday', 'variants_enc', 'time']
    df_x = df[cols]
    data_x = torch.tensor(df_x.to_numpy())
    data_y = torch.stack([confirm, dead], -1).T.squeeze()
    x = data_x
    y = data_y
    return x, y

def gaussian_f(X, a, b, c):
    y = a * np.exp(-0.5 * ((X-b)/c)**2)
    return y

def linear_f(X, a, b):
    y = a * X + b
    return y

def split_data(nation, df, train_ratio=0.8, val_ratio=0.1, start_date='2020-02-27', obs_end_date='2023-02-08', pred_end_date='2023-12-31'):
    tr_start_index = df[df['date'] == start_date].index.values[0]
    end_index = df[df['date'] == obs_end_date].index.values[0]
    pred_end_index = df[df['date'] == pred_end_date].index.values[0]

    len_data = end_index - tr_start_index # 1076
    len_train = int(len_data * train_ratio) # 860
    len_val = int(len_data * val_ratio) # 107
    # len_test = 109
    
    val_start_index = tr_start_index + len_train
    test_start_index = val_start_index + len_val
    pred_start_index = tr_start_index + len_data

    df.loc[len_train:, ['stringency', 'inoculation']] = -999
    df = df.replace(-999, np.NaN)

    # smooth y values (too much noise)
    conf_col = df['confirmation']
    dead_col = df['dead']
    ewm_conf_col = conf_col.ewm(span=15).mean()
    ewm_dead_col = dead_col.ewm(span=15).mean()
    df['confirmation'] = ewm_conf_col
    df['dead'] = ewm_dead_col

    guess = (0.5, 0.5)
    func = linear_f
    col_params = {}

    for col in ['stringency', 'inoculation']:
        # Get x & y
        x = df[:len_train].index.astype(float).values
        y = df[:len_train][col].values
        
        # Curve fit column and get curve parameters
        params = curve_fit(func, x, y, guess)
        # Store optimized parameters
        col_params[col] = params[0]

    # extrapolate
    for col in ['stringency', 'inoculation']:
        # Get the index values for NaNs in the column
        x = df[pd.isnull(df[col])].index.astype(float).values
        # Extrapolate those points with the fitted function
        df[col][x] = func(x, *col_params[col])
    
    # clipping
    df['inoculation'] = df['inoculation'].clip(lower=0, upper=1)
    df['stringency'] = df['stringency'].clip(lower=0)
    
    # extrapolate
    variants_list = df.columns[9:]
    for variant in variants_list:
        df.loc[len_train:, variant] = df.loc[len_train, variant]
    df[variants_list].iloc[len_train:] = df[variants_list].iloc[len_train]

    df['temperature'][val_start_index:pred_end_index] = df.loc[val_start_index-730: pred_end_index - 731, 'temperature']
    df['humidity'][val_start_index:pred_end_index] = df.loc[val_start_index-730: pred_end_index - 731, 'humidity']
    df['precipitation'][val_start_index:pred_end_index] = df.loc[val_start_index-730: pred_end_index - 731, 'precipitation']

    # add time feature
    df['time'] = np.array([i for i in range(len(df))])
    
    # import pdb; pdb.set_trace()
    variants_array = np.array(df[variants_list])
    # Get the index of the maximum value in each row
    max_indices = np.argmax(variants_array, axis=1)
    # Create the encoding vector based on the indices of the maximum values
    encoding_vector = max_indices + 1
    mapping = {1: 0, 2: 1, 5: 2, 7: 3, 8: 4, 
               9: 5, 10: 6, 11: 7, 12: 8, 13: 9, 
               14: 10, 15: 11, 16: 12, 17: 13, 18: 14}
    variants_enc = np.vectorize(mapping.get)(encoding_vector)
    df['variants_enc'] = variants_enc

    # train
    train_df = df[tr_start_index  : val_start_index].reset_index(drop=True)
    valid_df = df[val_start_index : test_start_index].reset_index(drop=True)
    test_df = df[test_start_index : pred_start_index].reset_index(drop=True)
    pred_df = df[pred_start_index : pred_end_index].reset_index(drop=True)
    
    # get date
    train_date = train_df['date']
    valid_date = valid_df['date']
    test_date = test_df['date']
    pred_date = pred_df['date']
    
    # df to tensor
    train_x, train_y = get_x_and_y(train_df)
    valid_x, valid_y = get_x_and_y(valid_df)
    test_x, test_y = get_x_and_y(test_df) 
    pred_x, _ = get_x_and_y(pred_df)

    train_y = train_y.T
    valid_y = valid_y.T
    test_y = test_y.T

    # get variants
    train_var = train_x[:,6]
    valid_var = valid_x[:,6]
    test_var = test_x[:,6]
    pred_var = pred_x[:,6]
    
    mms_x = MinMaxScaler()
    mms_y = MinMaxScaler()

    train_x = mms_x.fit_transform(train_x)
    train_y = mms_y.fit_transform(train_y)

    train_x[:,6] = train_var.numpy() # replace variants column with categorical variable
    
    meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
    meta_df = pd.read_csv(meta_df_path)
    population = meta_df['population'][0]
    conf_ratio_max = mms_y.data_max_[0]
    dead_ratio_max = mms_y.data_max_[1]

    conf_ratio_min = mms_y.data_min_[0]
    dead_ratio_min = mms_y.data_min_[1]

    # floating point issue may occur
    conf_max = conf_ratio_max * population
    dead_max = dead_ratio_max * population
    conf_min = conf_ratio_min * population
    dead_min = dead_ratio_min * population

    meta_df = meta_df[['population','min_confirmation', 'max_confirmation', 'min_inoculation','max_inoculation', 'min_dead', 'max_dead']]

    meta_df['max_confirmation'][0] = conf_max
    meta_df['max_dead'][0] = dead_max
    meta_df['min_confirmation'][0] = conf_min
    meta_df['min_dead'][0] = dead_min

    meta_df.to_csv(meta_df_path)
    
    valid_x = mms_x.transform(valid_x)
    valid_y = mms_y.transform(valid_y)
    test_x = mms_x.transform(test_x)
    test_y = mms_y.transform(test_y)
    pred_x = mms_x.transform(pred_x)
    
    valid_x[:,6] = valid_var.numpy()
    test_x[:,6] = test_var.numpy()
    pred_x[:,6] = pred_var.numpy()

    train_x = torch.tensor(train_x).to(torch.float32)
    train_y = torch.tensor(train_y).to(torch.float32)
    valid_x = torch.tensor(valid_x).to(torch.float32)
    valid_y = torch.tensor(valid_y).to(torch.float32)
    test_x = torch.tensor(test_x).to(torch.float32)
    test_y = torch.tensor(test_y).to(torch.float32)
    pred_x = torch.tensor(pred_x).to(torch.float32)

    return train_x, train_y, valid_x, valid_y, test_x, test_y, pred_x, train_date, valid_date, test_date, pred_date

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def get_data(nation_list, data_root, train_ratio, valid_ratio, start_date, obs_end_date, pred_end_date, device, args):
    data_dict = {}
    for nation in nation_list:
        file_name = f'{nation}.csv'
        file_path = os.path.join(data_root, file_name)

        df = pd.read_csv(file_path)
        train_x, train_y, valid_x, valid_y, test_x, test_y, pred_x, train_date, valid_date, test_date, pred_date = \
            split_data(nation, df, train_ratio, valid_ratio, start_date, obs_end_date, pred_end_date)
        
        data_dict[nation] = {
            'train_x': train_x.to(device),
            'train_y': train_y.to(device),
            'valid_x': valid_x.to(device),
            'valid_y': valid_y.to(device),
            'test_x': test_x.to(device),
            'test_y': test_y.to(device),
            'pred_x': pred_x.to(device)
        }   
        
    return data_dict, (train_date, valid_date, test_date, pred_date)

def gp_preprocess(data_dict, device, args):
    num_tasks = args.num_tasks
    model_type = args.model_type
    emb_dim = args.emb_dim
    kernel_name = args.kernel_name
    num_mixture = args.num_mixture
    prior_scale = args.prior_scale
    train_x_list = []
    train_y_list = []
    valid_x_list = []
    valid_y_list = []
    test_x_list = []
    test_y_list = []
    pred_x_list = []
    likelihood_list = []
    model_list = []
    for key in data_dict.keys():
        train_x = data_dict[key]['train_x']
        train_y = data_dict[key]['train_y']
        valid_x = data_dict[key]['valid_x']
        valid_y = data_dict[key]['valid_y']
        test_x = data_dict[key]['test_x']
        test_y = data_dict[key]['test_y']
        pred_x = data_dict[key]['pred_x']

        if model_type == 'MTGP':
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            model = models.MultiTaskGP(train_x.to(device), train_y.to(device), likelihood,
                                    dkl=args.dkl,
                                    kernel_name=kernel_name, 
                                    num_tasks=2, 
                                    embedding_dim=emb_dim,
                                    num_variants=args.num_variants,
                                    num_mixture=num_mixture,
                                    prior_scale=prior_scale)
            likelihood_list.append(likelihood)
            model_list.append(model)
            train_x_list.append(train_x)
            train_y_list.append(train_y)
            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y)
            test_x_list.append(test_x)
            test_y_list.append(test_y)
            pred_x_list.append(pred_x)

        elif model_type == 'STGP':
            # conf model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = models.SingleTaskGP(train_x.to(device), train_y[:,0].to(device), likelihood,
                                    dkl=args.dkl,
                                    kernel_name=kernel_name, 
                                    embedding_dim=emb_dim,
                                    num_variants=args.num_variants,
                                    num_mixture=num_mixture,
                                    prior_scale=prior_scale)
            likelihood_list.append(likelihood)
            model_list.append(model)
            # conf data
            train_x_list.append(train_x)
            train_y_list.append(train_y[:, 0])
            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y[:, 0])
            test_x_list.append(test_x)
            test_y_list.append(test_y[:, 0])
            pred_x_list.append(pred_x)
            # dead model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = models.SingleTaskGP(train_x.to(device), train_y[:,1].to(device), likelihood,
                                    dkl=args.dkl,
                                    kernel_name=kernel_name, 
                                    embedding_dim=emb_dim,
                                    num_variants=args.num_variants, 
                                    num_mixture=num_mixture,
                                    prior_scale=prior_scale)
            likelihood_list.append(likelihood)
            model_list.append(model)

            # dead data
            train_x_list.append(train_x)
            train_y_list.append(train_y[:, 1])
            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y[:, 1])
            test_x_list.append(test_x)
            test_y_list.append(test_y[:, 1])
            pred_x_list.append(pred_x)
        else:
            raise AssertionError
        
        data_ = (train_x_list, train_y_list, tuple(valid_x_list), tuple(valid_y_list), test_x_list, test_y_list, pred_x_list)
    return likelihood_list, model_list, data_

def get_optimizer(model, args):
    optim = args.optim
    init_lr = args.init_lr
    if optim == 'adam':
        optimizer_list = []
        for i in range(len(model.models)):
            model_ = model.models[i]
            optimizer = torch.optim.Adam(model_.parameters(), lr=init_lr)  # Includes GaussianLikelihood parameters
            optimizer_list.append(optimizer)
    return optimizer_list

# train, valid, test, pred_only dates
def get_grid_days(dates, args):
    train_date = dates[0]
    valid_date = dates[1]
    test_date = dates[2]
    pred_date = dates[3]
    
    days = pd.date_range(start=args.start_date, end=args.pred_end_date, freq='D').values.astype('datetime64[D]')
    first_day_of_month = pd.date_range(start=args.start_date, end=args.pred_end_date, freq='MS').values.astype('datetime64[D]')
    days_list = []
    color_list = []
    cnt = 0
    for i, day in enumerate(days):
        if day in first_day_of_month:
            if cnt % 6 == 0:
                if i == 3:
                    continue
                days_list.append(i)
                color_list.append('black')
            cnt += 1

    # train/valid/test days
    train_start = int(np.where(days == np.datetime64(train_date[0]))[0])
    valid_start = int(np.where(days == np.datetime64(valid_date[0]))[0])
    test_start = int(np.where(days == np.datetime64(test_date[0]))[0])
    pred_start = int(np.where(days == np.datetime64(pred_date[0]))[0])
    pred_end = int(np.where(days == np.datetime64(pred_date.iloc[-1]))[0])
    days_list.append(train_start)
    color_list.append('red')
    days_list.append(valid_start)
    color_list.append('red')
    days_list.append(test_start)
    color_list.append('red')
    days_list.append(pred_start)
    color_list.append('red')
    days_list.append(pred_end)
    color_list.append('red')

    days_list, color_list = zip(*sorted(zip(days_list, color_list)))
    return days_list, color_list

def plot(epoch, types, model_type, dates, ori_train_y, ori_valid_y, ori_test_y, all_mean, 
         nation, max_value, args):
    """
    model_type: type of model. in ['ARIMA', 'SVR', 'LINEAR', 'POLYNOMIAL']
    dates: list of dates. [train_date, valid_date, test_date, pred_date]
    ori_train_y, ori_valid_y, ori_test_y: observed value of conf / dead at each splits (train, valid, test)
    ori_train_mean, ori_valid_mean, ori_mean, pro_pred mean: predicted value of conf /dead at each splits. (train, valid, test, pred_only)
    nation: name of nation.
    max_value: max value of observed data.
    """
    
    plot_root = paths.PLOTS_ROOT
    model_root = os.path.join(plot_root, model_type)
    plot_dir = model_root
    
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
    
    # observed value
    days_list, color_list = get_grid_days(dates, args)
    days_list = list(days_list)
    train_date = dates[0]
    valid_date = dates[1]
    test_date = dates[2]
    pred_date = dates[3]
    all_date = pd.concat([train_date, valid_date, test_date, pred_date]).reset_index(drop=True)
    plt.clf()
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20, 16))
    plt.ylim([-0.1 * max_value, 1.01 * max_value])

    plt.plot(train_date, ori_train_y, 'ko')
    plt.plot(valid_date, ori_valid_y, 'ro')
    plt.plot(test_date, ori_test_y, 'go')

    plt.plot(all_date, all_mean, 'b')

    legends = ['Observed Data-Train', 'Observed Data-Valid', 'Observed Data-Test', 'Mean']

    plt.legend(legends)
    plt.title(f'{nation} - {types}')
    plt.xticks(days_list,rotation=75, ha='right', va='top')
    
    for i, color in enumerate(color_list):
        plt.gca().get_xticklabels()[i].set_color(color)
    plt.tight_layout()
    fig = plt.gcf()
    if not args.ignore_wandb:
        wandb.log({f"{nation}-{types}": wandb.Image(fig)})
    plt.savefig(f'{plot_dir}/{nation}-{types}.png')

def plot_gp(all_mean, upper_list, lower_list, ori_y_list, nation, nation_plot_root, max_value, type_, dates, epoch, args):
    ori_train_upper = upper_list[0]
    ori_valid_upper = upper_list[1]
    ori_test_upper = upper_list[2]
    ori_pred_upper = upper_list[3]

    ori_train_lower = lower_list[0]
    ori_valid_lower = lower_list[1]
    ori_test_lower = lower_list[2]
    ori_pred_lower = lower_list[3]

    ori_train_y = ori_y_list[0]
    ori_valid_y_cpu = ori_y_list[1]
    ori_test_y = ori_y_list[2]

    train_date = dates[0]
    valid_date = dates[1]
    test_date = dates[2]
    pred_date = dates[3]
    all_date = pd.concat([train_date, valid_date, test_date, pred_date]).reset_index(drop=True)

    days_list, color_list = utils.get_grid_days(dates, args)
    days_list = list(days_list)
    f, ax = plt.subplots(1, 1, figsize=(20, 16))
    plt.rcParams.update({'font.size': 25})
    ax.set_ylim([-0.1 * max_value, 1.01 * max_value])
    
    ax.plot(train_date, ori_train_y, 'ko')
    ax.plot(valid_date, ori_valid_y_cpu, 'ro')
    ax.plot(test_date, ori_test_y, 'go')
    ax.plot(all_date, all_mean, 'b')
    
    # confidence region
    if type(ori_train_lower) != list:
        ori_train_lower = ori_train_lower.numpy()
        ori_valid_lower = ori_valid_lower.numpy()
        ori_test_lower = ori_test_lower.numpy()
        ori_pred_lower = ori_pred_lower.numpy()

        ori_train_upper = ori_train_upper.numpy()
        ori_valid_upper = ori_valid_upper.numpy()
        ori_test_upper = ori_test_upper.numpy()
        ori_pred_upper = ori_pred_upper.numpy()

    ax.fill_between(train_date, ori_train_lower, ori_train_upper, alpha=0.5)
    ax.fill_between(valid_date, ori_valid_lower, ori_valid_upper, alpha=0.5)
    ax.fill_between(test_date, ori_test_lower, ori_test_upper, alpha=0.5)
    ax.fill_between(pred_date, ori_pred_lower, ori_pred_upper, alpha=0.5)

    xticks = [all_date[day_idx] for day_idx in days_list]
    xticklabels = [date for date in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=75, ha='right', va='top')
    for color_idx, color in enumerate(color_list):
        tick = ax.get_xticklabels()[color_idx]
        tick.set_color(color)

    legends = ['Observed Data-Train', 'Observed Data-Valid', 'Observed Data-Test', 'Mean', 'Confidence-Train', 'Confidence-Valid', 'Confidence-Test', 'Confidence-PredOnly']
    ax.tick_params(axis='both', labelsize=25)
    ax.set_title(f'{nation}-{type_}', fontsize=25)
    ax.legend(labels=legends)
    type_root = os.path.join(nation_plot_root, type_)
    if not os.path.isdir(type_root):
        os.makedirs(type_root, exist_ok=True)
    f.tight_layout()
    
    f.savefig(f'{type_root}/{epoch}epoch.png')