import itertools, datetime, os
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import paths
import utils
import metrics
import wandb

import warnings
warnings.filterwarnings(action='ignore')


def run_arima(args, data_dict, nation_list):

    # Set criterion
    mae = nn.L1Loss()
    rmse = metrics.RMSELoss()
    mape = metrics.MAPE()
    nme = metrics.NME()

    pred_conf_dict = {}; conf_mae_list = []; conf_rmse_list = []; conf_mape_list = []; conf_nme_list = []
    pred_dead_dict = {}; dead_mae_list = []; dead_rmse_list = []; dead_mape_list = []; dead_nme_list = []

    # run arima model for each nation
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

        train_y = data_dict[nation]['train_y']
        valid_y = data_dict[nation]['valid_y']
        test_y = data_dict[nation]['test_y']
        
        pred_x_len = len(data_dict[nation]['pred_x'])
        
        ori_test_y_conf = test_y[:, 0].numpy() * (max_confirmation - min_confirmation) + min_confirmation
        ori_test_y_dead = test_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        # -------------------------------------------------------------------------------
        
        ## Step 2] Run ARIMA ------------------------------------------------------------
        p = range(0, args.max_p+1)
        d = range(0, args.max_d+1)
        q = range(0, args.max_q+1)
        pdq = list(itertools.product(p, d, q))
        # -------------------------------------------------------------------------------
        
        ## Phase 1] Deal with Confirmation  --------------------------------------------
        ## Step 2-1] Fitting ARIMA -----------------------------------------------------
        aic = []; params = []
        for i in pdq:
            try:
                model = ARIMA(train_y[:,0].numpy(), order=(i))
                model_fit = model.fit()
                aic.append(round(model_fit.aic, 4))
                params.append((i))
            except:
                continue

        conf_optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
        # -------------------------------------------------------------------------------
        
        ## Step 2-2] Reproduce Best Model -----------------------------------------------
        conf_model = ARIMA(train_y[:,0].numpy(), order=conf_optimal[0][0])
        conf_model_fit = conf_model.fit()
        
        # predict Validation, Test, Prediction
        pred_conf = conf_model_fit.forecast(steps=(len(valid_y[:,0]) + len(test_y[:,0]) + pred_x_len))
        # -------------------------------------------------------------------------------
        
        ## Step 2-3] Restore transform--------------------------------------------------
        # predict and restore train values
        tr_pred_conf = train_y[:,0] * (max_confirmation - min_confirmation) + min_confirmation
    
        # predict and restore validation values
        val_pred_conf = pred_conf[:len(valid_y[:,0])] * (max_confirmation - min_confirmation) + min_confirmation        
        
        # predict and restore test values
        te_pred_conf = pred_conf[len(valid_y[:,0]):len(valid_y[:,0])+len(test_y[:,0])] * (max_confirmation - min_confirmation) + min_confirmation
        scaled_te_pred_conf = pred_conf[len(valid_y[:,0]):len(valid_y[:,0])+len(test_y[:,0])]
        
        # predict and restore prediction values 
        pred_pred_conf = pred_conf[len(valid_y[:,0])+len(test_y[:,0]):] * (max_confirmation - min_confirmation) + min_confirmation
        # -------------------------------------------------------------------------------
        
        ## Step 2-4] Calculate Test Loss ------------------------------------------------
        mae_loss_conf, rmse_loss_conf, mape_loss_conf, nme_loss_conf = \
            metrics.get_all_metrics([mae, rmse, mape, nme],\
                                    ori_test_y_conf, te_pred_conf)
        
        scaled_mae_loss_conf, scaled_rmse_loss_conf, scaled_mape_loss_conf, scaled_nme_loss_conf = \
            metrics.get_all_metrics([mae, rmse, mape, nme],\
                                    test_y[:, 0].numpy(), scaled_te_pred_conf)
            
        conf_mae_list.append(mae_loss_conf)
        conf_rmse_list.append(rmse_loss_conf)
        conf_mape_list.append(mape_loss_conf)
        conf_nme_list.append(nme_loss_conf)
        print(f"Best (p, d, q) and AIC for Confirmation : {conf_optimal}")
        print(f"MAE : {mae_loss_conf} / RMSE : {rmse_loss_conf}/ MAPE : {mape_loss_conf}/ NME : {nme_loss_conf}")
        print("-"*30)
        print(f"Best (p, d, q) and AIC for Confirmation : {conf_optimal}")
        print(f"MAE : {scaled_mae_loss_conf} / RMSE : {scaled_rmse_loss_conf}/ MAPE : {scaled_mape_loss_conf}/ NME : {scaled_nme_loss_conf}")
        print("-"*30)
        # -------------------------------------------------------------------------------
        
        ## Step 2-5] Save all predictions to plot figures -------------------------------
        full_pred_conf = np.concatenate((tr_pred_conf, val_pred_conf, te_pred_conf, pred_pred_conf))
        pred_conf_dict[nation] = full_pred_conf
        # -------------------------------------------------------------------------------
        
        
        ## Phase 2] Deal with Dead  --------------------------------------------
        ## Step 2-1] Fitting ARIMA ---------------------
        aic = []; params = []
        for i in pdq:
            try:
                model = ARIMA(train_y[:,1].numpy(), order=(i))
                model_fit = model.fit()
                aic.append(round(model_fit.aic, 4))
                params.append((i))
            except:
                continue

        dead_optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
        # -------------------------------------------------------------------------------
        
        ## Step 2-2] Reproduce Best Model -----------------------------------------------
        dead_model = ARIMA(train_y[:,1].numpy(), order=dead_optimal[0][0])
        dead_model_fit = dead_model.fit()
        
        # predict Validation, Test, Prediction
        pred_dead = dead_model_fit.forecast(steps=(len(valid_y[:,1]) + len(test_y[:,1]) + pred_x_len))
        # -------------------------------------------------------------------------------

        
        ## Step 2-3] Restore transform--------------------------------------------------
        # predict and restore train values
        tr_pred_dead = train_y[:,1] * (max_dead - min_dead) + min_dead
        
        # predict and restore validation values
        val_pred_dead = pred_dead[:len(valid_y[:,1])] * (max_dead - min_dead) + min_dead
        
        # predict and restore test values
        te_pred_dead = pred_dead[len(valid_y[:,1]):len(valid_y[:,1])+len(test_y[:,1])] * (max_dead - min_dead) + min_dead
        scaled_te_pred_dead = pred_dead[len(valid_y[:,1]):len(valid_y[:,1])+len(test_y[:,1])]
        
        # predict and restore prediction values 
        pred_pred_dead = pred_dead[len(valid_y[:,1])+len(test_y[:,1]):] * (max_dead - min_dead) + min_dead
        # -------------------------------------------------------------------------------
        
        ## Step 2-4] Calculate Test Loss ------------------------------------------------
        mae_loss_dead, rmse_loss_dead, mape_loss_dead, nme_loss_dead = \
            metrics.get_all_metrics([mae, rmse, mape, nme],\
                                    ori_test_y_dead, te_pred_dead)
            
        scaled_mae_loss_dead, scaled_rmse_loss_dead, scaled_mape_loss_dead, scaled_nme_loss_dead = \
            metrics.get_all_metrics([mae, rmse, mape, nme],\
                                    test_y[:, 1].numpy(), scaled_te_pred_dead)
            
        dead_mae_list.append(mae_loss_dead)
        dead_rmse_list.append(rmse_loss_dead)
        dead_mape_list.append(mape_loss_dead)
        dead_nme_list.append(nme_loss_dead)
        print(f"Best (p, d, q) and AIC for Confirmation : {dead_optimal}")
        print(f"MAE : {mae_loss_dead} / RMSE : {rmse_loss_dead}/ MAPE : {mape_loss_dead}/ NME : {nme_loss_dead}")
        print("-"*30)
        print(f"Best (p, d, q) and AIC for Confirmation : {dead_optimal}")
        print(f"MAE : {scaled_mae_loss_dead} / RMSE : {scaled_rmse_loss_dead}/ MAPE : {scaled_mape_loss_dead}/ NME : {scaled_nme_loss_dead}")
        print("-"*30)
        # -------------------------------------------------------------------------------
        
        ## Step 2-5] Save all predictions to plot figures -------------------------------
        full_pred_dead = np.concatenate((tr_pred_dead, val_pred_dead, te_pred_dead, pred_pred_dead))
        pred_dead_dict[nation] = full_pred_dead
        # -------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    
    ## Step 3] Calculate total loss --------------------------------------------------
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
    # --------------------------------------------------------------------------------