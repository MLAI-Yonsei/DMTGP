import os, paths, itertools
import matplotlib.pyplot as plt
import torch, wandb

import numpy as np
import pandas as pd
import torch.nn as nn

import utils, metrics

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings(action='ignore')


def run_linear_regression(args, data_dict, nation_list):
    ## Set criterion
    mae = nn.L1Loss()
    rmse = metrics.RMSELoss()
    mape = metrics.MAPE()
    nme = metrics.NME()

    pred_conf_dict = {}; conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
    pred_dead_dict = {}; dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []

    # run linear regression model for each nation
    for i, nation in enumerate(nation_list):
        print("-"*30)
        print(f"{nation} data")
        
        ## Step 1] Get Data --------------------------------------------------------------
        meta_df_path = f'{paths.DATA_ROOT}/meta_{nation}.csv'
        meta_df = pd.read_csv(meta_df_path)

        min_confirmation = meta_df['min_confirmation'][0]
        max_confirmation = meta_df['max_confirmation'][0]
        
        min_dead = meta_df['min_dead'][0]
        max_dead = meta_df['max_dead'][0]

        train_x = data_dict[nation]['train_x']
        train_y = data_dict[nation]['train_y']
        valid_x = data_dict[nation]['valid_x']
        
        test_x = data_dict[nation]['test_x']
        test_y = data_dict[nation]['test_y']
        
        pred_x = data_dict[nation]['pred_x']
        # -------------------------------------------------------------------------------
        
        ## Step 2] Fitting Linear Regression ----------------------------------------------------------        
        ## fitting training data
        linear_regression_conf = LinearRegression()
        linear_regression_conf.fit(train_x, train_y[:,0])
        
        linear_regression_dead = LinearRegression()
        linear_regression_dead.fit(train_x, train_y[:,1])
        # -------------------------------------------------------------------------------

        
        ## Step 3] Restore transform
        # predict and restore train values
        tr_pred_conf = linear_regression_conf.predict(train_x)
        tr_pred_dead = linear_regression_dead.predict(train_x)
        tr_pred_conf = tr_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        tr_pred_dead = tr_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore validation values
        val_pred_conf = linear_regression_conf.predict(valid_x)
        val_pred_dead = linear_regression_dead.predict(valid_x)
        val_pred_conf = val_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        val_pred_dead = val_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore test values
        te_pred_conf = linear_regression_conf.predict(test_x)
        te_pred_dead = linear_regression_dead.predict(test_x)
        te_pred_conf = te_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        te_pred_dead = te_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore prediction values
        pred_pred_conf = linear_regression_conf.predict(pred_x)
        pred_pred_dead = linear_regression_dead.predict(pred_x) 
        pred_pred_conf = pred_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        pred_pred_dead = pred_pred_dead * (max_dead - min_dead) + min_dead
        
        # restore test y values
        ori_test_y_conf = test_y[:, 0].numpy() * (max_confirmation - min_confirmation) + min_confirmation
        ori_test_y_dead = test_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        # -------------------------------------------------------------------------------

        ## Step 4] Calculate Test Loss
        mae_loss_conf, rmse_loss_conf, mape_loss_conf, nme_loss_conf = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_conf, te_pred_conf)
        conf_mae_list.append(mae_loss_conf); conf_rmse_list.append(rmse_loss_conf); conf_mape_list.append(mape_loss_conf); conf_nme_list.append(nme_loss_conf)
        
        mae_loss_dead, rmse_loss_dead, mape_loss_dead, nme_loss_dead = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_dead, te_pred_dead)
        dead_mae_list.append(mae_loss_dead); dead_rmse_list.append(rmse_loss_dead); dead_mape_list.append(mape_loss_dead); dead_nme_list.append(nme_loss_dead)
        # -------------------------------------------------------------------------------
        print(mae_loss_conf, mae_loss_dead)
        # Step 5] save all predictions to plot figures -----------------------------------
        full_pred_conf = np.concatenate((tr_pred_conf, val_pred_conf, te_pred_conf, pred_pred_conf))
        full_pred_dead = np.concatenate((tr_pred_dead, val_pred_dead, te_pred_dead, pred_pred_dead))
        pred_conf_dict[nation] = full_pred_conf
        pred_dead_dict[nation] = full_pred_dead
        # -------------------------------------------------------------------------------

    ## Step 6] Compute Total Loss ----------------------------------------------------------
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
    
    print(f"Dead MAE : {dead_mae}")
    print(f"Dead RMSE  : {dead_rmse}")
    print(f"Dead MAPE : {dead_mape}")
    print(f"Dead NME : {dead_nme}")


    if not args.ignore_wandb:
        wandb_dict = {
            f"{args.model_type}-conf_mae": conf_mae,
            f"{args.model_type}-conf_rmse": conf_rmse,
            f"{args.model_type}-conf_mape": conf_mape,
            f"{args.model_type}-conf_nme": conf_nme,
            f"{args.model_type}-dead_mae": dead_mae,
            f"{args.model_type}-dead_rmse": dead_rmse,
            f"{args.model_type}-dead_mape": dead_mape,
            f"{args.model_type}-dead_nme": dead_nme,
        }
        wandb.log(wandb_dict)
    
    return pred_conf_dict, pred_dead_dict
    # -------------------------------------------------------------------------------
