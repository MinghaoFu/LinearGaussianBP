import torch
import torch.nn as nn
import torch.optim as optim
import os, pwd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ipdb
import pytorch_lightning as pl

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
from utils import *
from data import synthetic_data

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model.dyn_vae import TV_VAE

np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

class CustomDataset(Dataset):
    def __init__(self, X, T, B):
        self.X = X
        self.T = T
        self.B = B

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.B[idx]


def train_model(args, model, criterion, data_loader, optimizer, m_true, X, T_tensor, B_init, pre_epoch=0):
    model.train()  
    if args.pretrain:
        
        for epoch in tqdm(range(args.init_epoch)):
            for batch_X, T, _ in data_loader:
                optimizer.zero_grad()  
                B = model(T)
                B_label = check_tensor(B_init).repeat(T.shape[0], 1, 1)
                loss = criterion(batch_X, T, B, B_label)
                loss['total_loss'].backward()
                optimizer.step()

        save_epoch_log(args, model, m_true, X, T_tensor, -1)
        print(f"--- Init F based on linear regression, ultimate loss: {loss['total_loss'].item()}")
        
    for epoch in range(args.epoch):
        model.train()
        if epoch < pre_epoch:
            continue
        for batch in data_loader:
            X, T, B_label = batch
            optimizer.zero_grad()  
            B = model(T) 
            loss = criterion(X, T, B)
            loss['total_loss'].backward()
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            optimizer.step()
        
        if epoch % (args.epoch // 100) == 0:
            save_epoch_log(args, model, m_true, X, T_tensor, epoch)
            print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }')
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'epoch_{epoch}', 'checkpoint.pth'))
            
        #optimizer.schedule()

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')  # GPU available
    else:
        device = torch.device('cpu')

    makedir(args.save_dir)
    if self.condition == 'golem':
        import pdb; pdb.set_trace() 
        dataset = synthetic_data.generate_data(args)
        X = check_tensor(dataset.X, dtype=torch.float32)
        X = X - X.mean(dim=0)
        T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
        model = TV_VAE(args)
        tensor_dataset = TensorDataset(X, T)
        data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
        if torch.cuda.is_available():
            model = model.cuda()
        if args.optimizer == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        for epoch in range(args.epoch):
            for X_batch, T_batch in data_loader:
                model.train()
                optimizer.zero_grad()
                losses = model.compute_vae_loss(X_batch, T_batch)
                losses['total_loss'].backward()
                
                if args.gradient_noise is not None:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
        
                optimizer.step()
            print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
    if args.time_varying is False:  
        dataset = synthetic_data.generate_data(args)
        X = check_tensor(dataset.X, dtype=torch.float32)
        X = X - X.mean(dim=0)
        B = check_tensor(dataset.B, dtype=torch.float32)
        I = check_tensor(torch.eye(args.d_X))
        inv_I_minus_B = torch.inverse(check_tensor(I - B))
        X_cov = torch.cov(X.T)
        if args.d_L > 0:
            C = check_tensor(dataset.C, dtype=torch.float32)
            BC = check_tensor(dataset.BC)#np.concatenate((np.zeros((args.max_d_L, args.max_d_L + args.d_X)), np.concatenate((C, B), axis=1)), axis=0)
            model = latent_joint_gaussian(args, dataset)
            est_X_cov, nll = model.log_gaussian_likelihood(B, C, check_tensor(dataset.EX_cov, dtype=torch.float32), check_tensor(dataset.EL_cov, dtype=torch.float32), args.num, X_cov)
        elif args.d_L == 0:
            model = joint_gaussian(args)
            est_X_cov, nll = model.log_gaussian_likelihood(B, check_tensor(dataset.EX_cov, dtype=torch.float32), args.num, X_cov)
    
        print('--- Population covariance - sample covariance = {}'.format(torch.norm(est_X_cov - X_cov).item()))
        print('--- True nll: {}'.format(nll.item()))

        if torch.cuda.is_available():
            model = model.cuda()
        if args.optimizer == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            loss = model(args, X_cov)
            loss['score'].backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 5.)
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            
            for i in range(model.B.shape[0]):
                model.B.grad[i, i] = 0
            
            optimizer.step()

            if (epoch % (args.epoch // 100) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
                model.eval()
                print(f'--- Epoch {epoch}, Loss: { {l: loss[l].item() for l in loss.keys()} }') #lr: {optimizer.get_lr()}
                print('--- Estimated covariance - sample covariance = {}'.format(torch.norm(model.est_X_cov - X_cov).item()))
                est_B = model.B.cpu().detach().numpy()
                est_B_postprocess = postprocess(est_B, graph_thres=args.graph_thres)
                est_EX_cov = model.EX_cov.cpu().detach().numpy()
                fig_dir = os.path.join(args.save_dir, 'figs')
                makedir(fig_dir)
                B_labels = [f'M{i}' for i in range(dataset.B.shape[1])]
                make_dots(dataset.B, B_labels, fig_dir, 'B')
                make_dots(est_B, B_labels, fig_dir, 'est_B')
                make_dots(est_B_postprocess, B_labels, fig_dir, 'est_B_postprocess')
                # print(dataset.B)
                # print(est_B_postprocess)
                # print(C, est_C)
                if args.d_L != 0:
                    L_labels = [f'L{i}' for i in range(dataset.C.shape[1])]
                    est_C = model.C.cpu().detach().numpy()
                    est_C[np.abs(est_C) <= args.graph_thres] = 0
                    est_EL_cov = model.EL_cov
                    est_BC = np.concatenate((np.zeros((args.max_d_L, args.max_d_L + args.d_X)), np.concatenate((est_C, est_B_postprocess), axis=1)), axis=0)
                    make_dots(dataset.BC, L_labels + B_labels, fig_dir, 'BC')
                    make_dots(est_BC, L_labels + B_labels, fig_dir, 'est_BC')
                    
    else:
        if args.synthetic:
            dataset = synthetic_data.generate_data(args)
            #dataset = sim_dataset.TimeVaryingDataset('/home/minghao.fu/workspace/LatentTimeVaryingCausalDiscovery/data/pnl_modular_1/source/data.npz')    
            args.num, args.d_X, args.d_L, args.lags, args.length = dataset.X.shape[0], 8, 8, 2, 1
            X = check_tensor(dataset.X, dtype=torch.float32)
            X = X - X.mean(dim=0)
            T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
            model = TV_VAE(args)
            tensor_dataset = TensorDataset(X, T)
            data_loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)
            if torch.cuda.is_available():
                model = model.cuda()
            if args.optimizer == 'ADAM':
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
            elif args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
            for epoch in range(args.epoch):
                for X_batch, T_batch in data_loader:
                    model.train()
                    optimizer.zero_grad()
                    losses = model.compute_vae_loss(X_batch, T_batch)
                    losses['total_loss'].backward()
                    
                    if args.gradient_noise is not None:
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
            
                    optimizer.step()
                print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
                #if (epoch % (args.epoch // args.evaluate_num) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
                    
                    # model.eval()
                    # BT, CT, CT_1 = model.compute_vae_loss(X, T)
                    # BT, CT, CT_1 = BT.cpu().detach().numpy(), CT.cpu().detach().numpy(), CT_1.cpu().detach().numpy()
                    # epoch_save_dir = os.path.join(args.save_dir, f'Epoch {epoch}')
                    # makedir(epoch_save_dir)
                    
                    # for i in range(args.num):
                    #     if i == 0:
                    #         avg_acc = count_accuracy(dataset.Bs[i], BT[i])
                    #     else:
                    #         acc = count_accuracy(dataset.Bs[i], dataset.Bs[i])
                    #         for key in acc.keys():
                    #             avg_acc[key] += acc[key]
                    #     if i == args.num - 1:
                    #         for key in avg_acc.keys():
                    #             avg_acc[key] /= args.num
                                
                    # print('--- Accuracy: {}'.format(avg_acc))
                    
                    # for i in range(5):
                    #     sub_fig_dir = os.path.join(epoch_save_dir, f't_{i}')
                    #     Bt = postprocess(BT[i], graph_thres=args.graph_thres)
                    #     Ct = CT[i]
                    #     Ct[Ct < args.graph_thres] = 0
                    #     B_labels = [f'M{i}' for i in range(args.d_X)]
                    #     C_labels = [f'L{i}' for i in range(args.d_L)]
                    #     makedir(sub_fig_dir)
                    #     plot_solution(dataset.Bs[i], BT[i], Bt, os.path.join(sub_fig_dir, 'test_plot_solution'))
                    #     # make_dots(dataset.B[i], B_labels, sub_fig_dir, f'B_gt{i}')
                    #     # make_dots(dataset.C[i], C_labels, sub_fig_dir, f'C_gt{i}')
                    #     BCt = np.concatenate((np.zeros((args.d_L, args.d_L + args.d_X)), np.concatenate((Ct, Bt), axis=1)), axis=0)
                    #     make_dots(dataset.BCs[i], C_labels + B_labels, sub_fig_dir, f'BC_gt{i}')
                    #     make_dots(BCt, C_labels + B_labels, sub_fig_dir, f'BC_est{i}')
                    #     if args.assume_time_lag:
                    #         Ct_1 = CT_1[i][CT_1[i] < args.graph_thres]

        else:
            args.data_path = './data/CESM2_pacific_SST.pkl'
            data = climate.generate_data(args)
            T = args.num * np.arange(data.shape[0]) / data.shape[0]
            T_tensor = check_tensor(T)
            data_tensor = check_tensor(data)
            B_init = linear_regression_initialize(data, args.distance)
            #print('B initialization: \n{}'.format(B_init))
            
def main_tdrl(args):
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('./configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    
    args = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(args, default_flow_style=False))
    print("#################################")
    
    pl.seed_everything(args.seed)

    makedir(args.save_dir)
    dataset = sim_dataset.TimeVaryingDataset('/home/minghao.fu/workspace/LatentTimeVaryingCausalDiscovery/data/case2_nonstationary_causal/data.npz')    
    args.num, args.d_X, args.d_L, args.lags, args.length = \
        dataset.data['xt'].shape[0], dataset.data['xt'].shape[2], dataset.data['yt'].shape[2], 2, 1
    # X = check_tensor(dataset.X, dtype=torch.float32)
    # X = X - X.mean(dim=0)
    # T = check_tensor(torch.arange(args.num), dtype=torch.float32).reshape(-1, 1)
    # tensor_dataset = TensorDataset(X, T)
    model = TV_VAE(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    num_validation_samples = int(args.eval_rate * len(dataset))
    train_data, val_data = random_split(dataset, [len(dataset)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size,
                              num_workers=args.n_cpus,
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            num_workers=args.n_cpus,
                            shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
        
    if args.optimizer == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon, weight_decay=0) # make_optimizer(model, args)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    model = tdrl.ModularShifts(input_dim=args.d_X,
                        length=args.length,
                        obs_dim=0,
                        dyn_dim=args.d_L,
                        lag=args.lag,
                        nclass=1)
    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')
    trainer = pl.Trainer(default_root_dir=args.save_dir,
                         #gpus=args.n_gpus, 
                         #val_check_interval = args.val_check_interval,
                         check_val_every_n_epoch=None,
                         max_epochs=args.n_epochs,
                         callbacks=[checkpoint_callback])
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    for epoch in range(args.epoch):
        for X_batch, T_batch in data_loader:
            model.train()
            optimizer.zero_grad()
            losses = model.compute_vae_loss(X_batch, T_batch)
            losses['total_loss'].backward()
            
            if args.gradient_noise is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad += check_tensor(torch.normal(0, args.gradient_noise, size=param.grad.size()))
    
            optimizer.step()
        print(f'--- Epoch {epoch}, Loss: { {l: losses[l].item() for l in losses.keys()} }')
        #if (epoch % (args.epoch // args.evaluate_num) == 0): # or markov_equivalence(dataset.B.T, est_B_postprocess.T)
            
            # model.eval()
            # BT, CT, CT_1 = model.compute_vae_loss(X, T)
            # BT, CT, CT_1 = BT.cpu().detach().numpy(), CT.cpu().detach().numpy(), CT_1.cpu().detach().numpy()
            # epoch_save_dir = os.path.join(args.save_dir, f'Epoch {epoch}')
            # makedir(epoch_save_dir)
            
            # for i in range(args.num):
            #     if i == 0:
            #         avg_acc = count_accuracy(dataset.Bs[i], BT[i])
            #     else:
            #         acc = count_accuracy(dataset.Bs[i], dataset.Bs[i])
            #         for key in acc.keys():
            #             avg_acc[key] += acc[key]
            #     if i == args.num - 1:
            #         for key in avg_acc.keys():
            #             avg_acc[key] /= args.num
                        
            # print('--- Accuracy: {}'.format(avg_acc))
            
            # for i in range(5):
            #     sub_fig_dir = os.path.join(epoch_save_dir, f't_{i}')
            #     Bt = postprocess(BT[i], graph_thres=args.graph_thres)
            #     Ct = CT[i]
            #     Ct[Ct < args.graph_thres] = 0
            #     B_labels = [f'M{i}' for i in range(args.d_X)]
            #     C_labels = [f'L{i}' for i in range(args.d_L)]
            #     makedir(sub_fig_dir)
            #     plot_solution(dataset.Bs[i], BT[i], Bt, os.path.join(sub_fig_dir, 'test_plot_solution'))
            #     # make_dots(dataset.B[i], B_labels, sub_fig_dir, f'B_gt{i}')
            #     # make_dots(dataset.C[i], C_labels, sub_fig_dir, f'C_gt{i}')
            #     BCt = np.concatenate((np.zeros((args.d_L, args.d_L + args.d_X)), np.concatenate((Ct, Bt), axis=1)), axis=0)
            #     make_dots(dataset.BCs[i], C_labels + B_labels, sub_fig_dir, f'BC_gt{i}')
            #     make_dots(BCt, C_labels + B_labels, sub_fig_dir, f'BC_est{i}')
            #     if args.assume_time_lag:
            #         Ct_1 = CT_1[i][CT_1[i] < args.graph_thres]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--exp',
        type=str,
        default='tvcd',
    )
    args = argparser.parse_args()
    main(args)