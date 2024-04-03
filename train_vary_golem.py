from Caulimate.Data.SimLinGau import SimLinGau
from Caulimate.Data.SimDAG import simulate_random_dag, simulate_weight, simulate_time_vary_weight
from Caulimate.Utils.Tools import check_array, check_tensor, makedir, linear_regression_initialize, load_yaml
from Caulimate.Utils.Visualization import save_DAG
from model.golemmodel import GolemModel
import torch
import torch.nn as nn
import torch.optim as optim
import os, pwd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb
import pytorch_lightning as fpl
import wandb
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import TensorDataset, DataLoader
from config import *
from data import synthetic_data
from data.tdrl_data import sim_dataset
from model import GolemModel, iVAE, joint_gaussian, latent_joint_gaussian
from loss import golem_loss, latent_variable_graphical_lasso_loss
from torch.utils.data import Dataset, DataLoader, random_split
from model.tvcd import tdrl
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
    args = load_yaml(os.path.join('configs', 'golem.yaml'))
    args.save_dir = os.path.join(args.save_dir, f'{args.dataset}_{args.d_X}_{args.distance}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    wandb_logger = WandbLogger(project='golem', name=datetime.now().strftime("%Y%m%d-%H%M%S"))#, save_dir=log_dir)
    
    rs = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GolemModel(args, args.d_X, in_dim=1, equal_variances=True, seed=1,)
    
    if args.dataset == 'synthetic':
        dataset_name = f'golem_{args.graph_type}_{args.degree}_{args.noise_type}_{args.num}_{args.d_X}_{args.cos_len}'
        if args.load_data == True:
            # X = np.load(f'./dataset/{dataset_name}/X.npy')
            # Bs = np.load(f'./dataset/{dataset_name}/Bs.npy')
            dataset = SimLinGau(args.num, args.noise_type, args.d_X, args.degree, args.cos_len, args.save_dir, args.seed, vary_func=np.cos)
            X, Bs, B_bin, coords = dataset.X, dataset.Bs, dataset.B_bin, dataset.coords
            # X = data['X']
            # Bs = data['Bs']
            # B_bin = data['B_bin']
            # coords = data['coords']
        else:
            pass
            # B_scale = 0.1
            # B_ranges = ((B_scale * -2.0, B_scale * -0.5),
            #                     (B_scale * 0.5, B_scale * 2.0))
            # Bs, B_bin = simulate_time_vary_weight(args.d_X, args.num, args.cos_len, B_ranges, args.degree, args.graph_type, args.vary_type, args.seed)
            # Bs = check_array(Bs)

            # X = np.zeros((args.num, args.d_X))
            # for i in range(args.num):
            #     X[i:i+1, :] = simulate_graph.simulate_linear_sem(Bs[i], 1, args.noise_type, rs)
            # #self.X = np.matmul(self.X, np.linalg.inv(np.eye(self.d_X) - self.Bs))
            
            # makedir('./dataset/{}'.format(dataset_name), remove_exist=True)
            # np.save(f'./dataset/{dataset_name}/X.npy', X)
            # np.save(f'./dataset/{dataset_name}/Bs.npy', Bs)
    # elif args.dataset == 'CESM': 
    #     X, Bs = CESM_dataset.load_data(args)
        
                
    X = check_tensor(X, dtype=torch.float32)
    X = X - X.mean(dim=0)
    T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
    Bs_gt = check_tensor(Bs, dtype=torch.float32)
    tensor_dataset = TensorDataset(X, T, Bs_gt)
    data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    golem_loss = golem_loss(args)
    
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f'--- Load checkpoint from {args.checkpoint_path}')
    else:
        B_init = linear_regression_initialize(X, args.distance)
        B_init = check_tensor(B_init, dtype=torch.float32)
        if args.regression_init:
            for epoch in tqdm(range(args.init_epoch)):
                for batch_X, batch_T, batch_B in data_loader:
                    optimizer.zero_grad()  
                    B_pred = model(T)
                    B_label = check_tensor(B_init).repeat(batch_T.shape[0], 1, 1)
                    loss = golem_loss(batch_X, batch_T, B_pred, B_init)
                    loss['total_loss'].backward()
                    optimizer.step()

            #save_epoch_log(args, model, B_init, X, T, -1)
            print(f"--- Init F based on linear regression, ultimate loss: {loss['total_loss'].item()}")
        
    for epoch in range(args.epoch):
        model.train()
        for X_batch, T_batch, B_batch in data_loader:
            optimizer.zero_grad()
            B_pred = model(T_batch)
            losses = golem_loss(X_batch, T_batch, B_pred)
            losses['total_loss'].backward()
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
    
            optimizer.step()
            
        if epoch % 100 == 0:
            model.eval()
            Bs_pred = check_array(model(T).permute(0, 2, 1))
            # if args.dataset != 'synthetic':
            #     Bs_gt = Bs_pred
            #     for i in range(args.num):
            #         Bs_gt[i] = postprocess(Bs_gt[i])
            if args.dataset != 'synthetic':
                Bs_gt = None
            save_DAG(args.num, args.save_dir, epoch, Bs_pred, Bs_gt, graph_thres=args.graph_thres, add_value=False)
            print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}', 'checkpoint.pth'))
            
            # pred_B = model(T)
            # print(pred_B[0], dataset.B)
            # fig = plot_solutions([B_gt.T, B_est, W_est, M_gt.T, M_est], ['B_gt', 'B_est', 'W_est', 'M_gt', 'M_est'], add_value=True, logger=self.logger)
            # self.logger.experiment.log({"Fig": [wandb.Image(fig)]})