import os, paths, itertools
import matplotlib.pyplot as plt
import torch, wandb

import numpy as np
import pandas as pd
import torch.nn as nn

import utils, metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings(action='ignore')


def run_polynomial_regression(args, data_dict, nation_list):
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

    degree_list = [4]
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
        valid_y = data_dict[nation]['valid_y']
        
        test_x = data_dict[nation]['test_x']
        test_y = data_dict[nation]['test_y']
        
        pred_x = data_dict[nation]['pred_x']
        # -------------------------------------------------------------------------------
        
        ## Step 2] Fitting Linear Regression ----------------------------------------------------------        
        ## fitting (confirmation) training data and select best model
        best_conf_val_loss = 999999999999; best_conf_degree = 0
        for conf_degree in degree_list:
            poly = PolynomialFeatures(degree=conf_degree, include_bias=True)
            train_x_poly = poly.fit_transform(train_x)
            valid_x_poly = poly.fit_transform(valid_x)
            
            regression_conf = LinearRegression()
            regression_conf.fit(train_x_poly, train_y[:,0])
            val_pred_conf = regression_conf.predict(valid_x_poly)
            
            conf_val_loss = eval_criterion(torch.Tensor(valid_y[:,0]), torch.Tensor(val_pred_conf))
            print("*"*conf_degree)
            if conf_val_loss < best_conf_val_loss:
                conf_val_loss = best_conf_val_loss
                best_conf_degree = conf_degree
        
        print(f"Best degree for confirmation of {nation} : {best_conf_degree}")
            

        ## fitting (dead) training data
        best_dead_val_loss = 999999999999; best_dead_degree = 0
        for dead_degree in degree_list:
            poly = PolynomialFeatures(degree=dead_degree, include_bias=True)
            train_x_poly = poly.fit_transform(train_x)
            valid_x_poly = poly.fit_transform(valid_x)
            
            regression_dead = LinearRegression()
            regression_dead.fit(train_x_poly, train_y[:,1])
            val_pred_dead = regression_dead.predict(valid_x_poly)
            
            dead_val_loss = eval_criterion(torch.Tensor(valid_y[:,1]), torch.Tensor(val_pred_dead))
            
            if dead_val_loss < best_dead_val_loss:
                dead_val_loss = best_dead_val_loss
                best_dead_degree = dead_degree
        
        print(f"Best degree for dead of {nation} : {best_dead_degree}")
        # -------------------------------------------------------------------------------
        
        
        ## Step 3] Reproduce Best Model -------------------------------------------------
        ## Confirmation
        poly_conf = PolynomialFeatures(degree=best_conf_degree, include_bias=True)
        train_x_conf_poly = poly_conf.fit_transform(train_x)
        valid_x_conf_poly = poly_conf.fit_transform(valid_x)
        test_x_conf_poly = poly_conf.fit_transform(test_x)
        pred_x_conf_poly = poly_conf.fit_transform(pred_x)
        
        regression_conf = LinearRegression()
        regression_conf.fit(train_x_conf_poly, train_y[:,0])
        
        ## Dead
        poly_dead = PolynomialFeatures(degree=best_dead_degree, include_bias=True)
        train_x_dead_poly = poly_dead.fit_transform(train_x)
        valid_x_dead_poly = poly_dead.fit_transform(valid_x)
        test_x_dead_poly = poly_dead.fit_transform(test_x)
        pred_x_dead_poly = poly_dead.fit_transform(pred_x)
        
        regression_dead = LinearRegression()
        regression_dead.fit(train_x_dead_poly, train_y[:,0])
        # -------------------------------------------------------------------------------

        
        ## Step 4] Restore transform
        # predict and restore train values
        tr_pred_conf = regression_conf.predict(train_x_conf_poly)
        tr_pred_dead = regression_dead.predict(train_x_dead_poly)
        tr_pred_conf = tr_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        tr_pred_dead = tr_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore validation values
        val_pred_conf = regression_conf.predict(valid_x_conf_poly)
        val_pred_dead = regression_dead.predict(valid_x_dead_poly)
        val_pred_conf = val_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        val_pred_dead = val_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore test values
        te_pred_conf = regression_conf.predict(test_x_conf_poly)
        te_pred_dead = regression_dead.predict(test_x_dead_poly)
        te_pred_conf = te_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        te_pred_dead = te_pred_dead * (max_dead - min_dead) + min_dead
        
        # predict and restore prediction values
        pred_pred_conf = regression_conf.predict(pred_x_conf_poly)
        pred_pred_dead = regression_dead.predict(pred_x_dead_poly) 
        pred_pred_conf = pred_pred_conf * (max_confirmation - min_confirmation) + min_confirmation
        pred_pred_dead = pred_pred_dead * (max_dead - min_dead) + min_dead
        
        # restore test y values
        ori_test_y_conf = test_y[:, 0].numpy() * (max_confirmation - min_confirmation) + min_confirmation
        ori_test_y_dead = test_y[:, 1].numpy() * (max_dead - min_dead) + min_dead
        # -------------------------------------------------------------------------------

        ## Step 5] Calculate Test Loss
        mae_loss_conf, rmse_loss_conf, mape_loss_conf, nme_loss_conf = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_conf, te_pred_conf)
        conf_mae_list.append(mae_loss_conf); conf_rmse_list.append(rmse_loss_conf); conf_mape_list.append(mape_loss_conf); conf_nme_list.append(nme_loss_conf)
        
        mae_loss_dead, rmse_loss_dead, mape_loss_dead, nme_loss_dead = metrics.get_all_metrics([mae, rmse, mape, nme],\
                                                                    ori_test_y_dead, te_pred_dead)
        dead_mae_list.append(mae_loss_dead); dead_rmse_list.append(rmse_loss_dead); dead_mape_list.append(mape_loss_dead); dead_nme_list.append(nme_loss_dead)
        # -------------------------------------------------------------------------------

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
