import pandas as pd
from .arima import run_arima
from .svr import run_svr
from .linear_regression import run_linear_regression
from .polynomial_regression import run_polynomial_regression

import utils
import paths

def run_baseline(model_type, args, dates, data_dict, nation_list):
    if model_type == 'ARIMA':
        pred_conf_dict, pred_dead_dict = run_arima(args, data_dict, nation_list)
    
    elif model_type == 'SVR':
        pred_conf_dict, pred_dead_dict = run_svr(args, data_dict, nation_list)

    elif model_type == 'LINEAR':
        pred_conf_dict, pred_dead_dict = run_linear_regression(args, data_dict, nation_list)

    elif model_type == "POLYNOMIAL":
        pred_conf_dict, pred_dead_dict = run_polynomial_regression(args, data_dict, nation_list)

    for nation in nation_list:
        train_y = data_dict[nation]['train_y']
        valid_y = data_dict[nation]['valid_y']
        test_y = data_dict[nation]['test_y']

        pred_conf = pred_conf_dict[nation]
        pred_dead = pred_dead_dict[nation]
        meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
        meta_df = pd.read_csv(meta_df_path)

        min_conf = meta_df['min_confirmation'][0]
        max_conf = meta_df['max_confirmation'][0]
        min_dead = meta_df['min_dead'][0]
        max_dead = meta_df['max_dead'][0]

        ori_train_y = train_y[:, 0].numpy() * (max_conf - min_conf) + min_conf
        ori_valid_y = valid_y[:, 0].numpy() * (max_conf - min_conf) + min_conf
        ori_test_y = test_y[:, 0].numpy() * (max_conf - min_conf) + min_conf

        utils.plot(-1, 'conf', model_type, dates, ori_train_y, ori_valid_y, ori_test_y, pred_conf, 
            nation, max_conf, args)
        ori_train_y = train_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        ori_valid_y = valid_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        ori_test_y = test_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        utils.plot(-1, 'dead', model_type, dates, ori_train_y, ori_valid_y, ori_test_y, pred_dead, 
                nation, max_dead, args)
    return pred_conf_dict, pred_dead_dict