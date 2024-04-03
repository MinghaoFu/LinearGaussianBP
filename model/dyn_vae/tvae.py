import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .base import MLP, GaussianMLP, Normal
from .encoder import RNNEncoder
from .decoder import TimeLagTransitionDecoder

from utils import check_tensor

class generate_Bs(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.Bs_encoder = MLP(1, args.d_X * args.d_X, args.d_X, args.n_mlp_layers, 'lrelu')
    
    def forward(self, x_o):
        Bs = self.Bs_encoder(x_o).reshape(T.shape[0], self.args.d_X, self.args.d_X)
        for i in range(self.args.d_X):
            Bs[:, i, i] = 0
        return Bs
    
class TV_VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.generate_Bs = generate_Bs(args)
        self.encoder = RNNEncoder(
            args=self.args,
        )
        self.decoder = TimeLagTransitionDecoder(args)
    
    def compute_vae_loss(self, X, T):
        """ Returns the VAE loss """
        Bs = self.generate_Bs(T)
        Is = check_tensor(torch.eye(self.args.d_X).unsqueeze(0).repeat(T.shape[0], 1, 1))
        X = torch.bmm((Is - Bs), X.unsqueeze(-1)).squeeze(-1)
        latent_sample, latent_mean, latent_logvar, encoder_output = self.encoder(X, T)
        
        recon, pred, latent_mean_pred, latent_logvar_pred, curr2m_mat, pre2m_mat = self.decoder(latent_mean, latent_logvar, T)
        recon = torch.bmm((Is - Bs), recon.unsqueeze(-1)).squeeze(-1)
        
        # evaluation
        if self.training == False:
            return Bs, curr2m_mat, pre2m_mat
        
        losses = {}
        recon_loss = self.compute_reconstruction_loss(X, recon)
        losses['recon_loss'] = recon_loss.sum()
        pred_recon_loss = self.compute_pred_reconstruction_loss(X, latent_mean, latent_logvar, pred, latent_mean_pred, latent_logvar_pred)
        losses['pred_recon_loss'] = pred_recon_loss.sum()

        kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices=None)
        losses['kl_loss'] = kl_loss.sum()
        losses['sparsity_loss'] = self.compute_sparsity_loss(curr2m_mat, pre2m_mat, Bs)
        
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses 
        
    def compute_reconstruction_loss(self, X, recon):
        # reconstruction loss
        recon_loss = (recon - X).pow(2).mean(dim=-1) 
        return self.args.reconstruction_loss_coeff * recon_loss
    
    def compute_pred_reconstruction_loss(self, X, latent_mean, latent_logvar, pred, latent_mean_pred, latent_logvar_pred):
        recon_loss = (pred - X).pow(2).mean(dim=-1)
        kl_divergences = 0.5 * (latent_logvar_pred - latent_logvar - 1
                                + (latent_logvar.exp() + (latent_mean - latent_mean_pred).pow(2))
                                / latent_logvar_pred.exp())
        kl_divergences = kl_divergences.sum(dim=-1)
        return recon_loss + kl_divergences
    
    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        # -- instantaneousKL divergence, N(0, 1) as prior
        if True:
            kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat((torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
            all_logvars = torch.cat((torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape((self.args.vae_subsample_elbos, batchsize))

        return self.args.kl_coeff * kl_divergences
    
    def compute_sparsity_loss(self, Bt, Ct, Ct_1=None):
        sparsity_loss = 0
        sparsity_loss += self.args.sparsity_Bt * torch.norm(Bt, p=1)
        sparsity_loss += self.args.sparsity_Ct * torch.norm(Ct, p=1)
        if self.args.assume_time_lag:
            sparsity_loss += self.args.sparsity_Ct_1 * torch.norm(Ct_1, p=1)
            
        return sparsity_loss
    
    def training_step(self, batch, batch_idx):
        x, y, c = batch['xt'], batch['yt'], batch['ct'] # c: domain index
        losses = self.compute_vae_loss(X, T)
        return losses
        