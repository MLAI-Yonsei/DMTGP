import os, paths, itertools
import matplotlib.pyplot as plt
import torch, wandb

import numpy as np
import pandas as pd
import torch.nn as nn

import utils, metrics

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

import warnings
warnings.filterwarnings(action='ignore')


def run_svr(args, data_dict, nation_list):
    ## Set criterion
    mae = nn.L1Loss()
    rmse = metrics.RMSELoss()
    mape = metrics.MAPE()
    nme = metrics.NME()

    if args.eval_criterion == "mae":
        eval_criterion = nn.L1Loss()
    elif args.eval_criterion == "rmse":
        eval_criterion = metrics.RMSELoss()
    elif args.eval_criterion == "mape":
        eval_criterion = metrics.MAPE()
    elif args.eval_acriterion == "nme":
        eval_criterion = metrics.NME()

    pred_conf_dict = {}; conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
    pred_dead_dict = {}; dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []

    best_gamma_list = []; best_C_list = []; best_epsilon_list = []

    # run svr model for each nation
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
        valid_y = data_dict[nation]['valid_y']
        
        test_x = data_dict[nation]['test_x']
        test_y = data_dict[nation]['test_y']
        
        pred_x = data_dict[nation]['pred_x']
        # -------------------------------------------------------------------------------
        
        ## Step 2] Fitting SVR ----------------------------------------------------------
        best_val_loss = 99999999999
        params = list(itertools.product(args.gamma, args.C, args.epsilon))
        for param in params:
            ## fitting training data
            svr = SVR(kernel='rbf'
                        ,gamma=param[0], C=param[1], epsilon = param[2])
            
            mor = MultiOutputRegressor(svr)
            mor.fit(train_x, train_y)
            
            ## validation
            val_pred = mor.predict(valid_x)
            val_loss_conf = eval_criterion(torch.Tensor(valid_y[:,0]), torch.Tensor(val_pred[:len(valid_x[:,0]), 0]))
            val_loss_dead = eval_criterion(torch.Tensor(valid_y[:,1]), torch.Tensor(val_pred[:len(valid_x[:,1]), 1]))
            val_loss = val_loss_conf + val_loss_dead

            ## Calculate validation loss and choose best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_gamma = param[0]
                best_C = param[1]
                best_epsilon = param[2]

        best_gamma_list.append(best_gamma)
        best_C_list.append(best_C)
        best_epsilon_list.append(best_epsilon)
        # -------------------------------------------------------------------------------
        
        ## Step 3] Fit Best Model ----------------------------------------------------
        print(f"{nation} params :: Best Gamma : {best_gamma} / Best C : {best_C} / Best epsilon : {best_epsilon}")
        # Fitting Best Model
        svr = SVR(kernel='rbf',
                gamma=best_gamma, C=best_C, epsilon=best_epsilon)
        
        mor = MultiOutputRegressor(svr)
        mor.fit(train_x, train_y)
        # -------------------------------------------------------------------------------
        
        ## Step 4] Restore transform ----------------------------------------------------
        # predict
        tr_pred = mor.predict(train_x)
        val_pred = mor.predict(valid_x)
        te_pred = mor.predict(test_x)
        pred_pred = mor.predict(pred_x)
        
        # Restore predictions and y values
        tr_pred_conf = tr_pred[:,0] * (max_confirmation - min_confirmation) + min_confirmation
        tr_pred_dead = tr_pred[:,1] * (max_dead - min_dead) + min_dead
        
        val_pred_conf = val_pred[:,0] * (max_confirmation - min_confirmation) + min_confirmation
        val_pred_dead = val_pred[:,1] * (max_dead - min_dead) + min_dead
        
        te_pred_conf = te_pred[:,0] * (max_confirmation - min_confirmation) + min_confirmation
        te_pred_dead = te_pred[:,1] * (max_dead - min_dead) + min_dead
        
        pred_pred_conf = pred_pred[:,0] * (max_confirmation - min_confirmation) + min_confirmation
        pred_pred_dead = pred_pred[:,1] * (max_dead - min_dead) + min_dead
        
        ori_test_y_conf = test_y[:, 0].numpy() * (max_confirmation - min_confirmation) + min_confirmation
        ori_test_y_dead = test_y[:, 1].numpy() * (max_dead - min_dead) + min_dead

        ## Step 5] Calculate Test Loss -----------------------------------------------------------------
        # Calculate test Loss
        mae_loss_conf, rmse_loss_conf, mape_loss_conf, nme_loss_conf = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_conf, te_pred_conf)
        
        conf_mae_list.append(mae_loss_conf); conf_rmse_list.append(rmse_loss_conf); conf_mape_list.append(mape_loss_conf); conf_nme_list.append(nme_loss_conf)
        
        mae_loss_dead, rmse_loss_dead, mape_loss_dead, nme_loss_dead = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_dead, te_pred_dead)
        dead_mae_list.append(mae_loss_dead); dead_rmse_list.append(rmse_loss_dead); dead_mape_list.append(mape_loss_dead); dead_nme_list.append(nme_loss_dead)
        
        print(mae_loss_conf, mae_loss_dead)
        # -------------------------------------------------------------------------------

        ## Step 6] Save all predictions to plot figures --------------------------------
        full_pred_conf = np.concatenate((tr_pred_conf, val_pred_conf, te_pred_conf, pred_pred_conf))
        full_pred_dead = np.concatenate((tr_pred_dead, val_pred_dead, te_pred_dead, pred_pred_dead))
        pred_conf_dict[nation] = full_pred_conf
        pred_dead_dict[nation] = full_pred_dead
        # -------------------------------------------------------------------------------

    ## Step 7] Compute Total Loss --------------------------------------------------------
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
