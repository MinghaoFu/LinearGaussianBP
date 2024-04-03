from datetime import datetime
import argparse
import yaml
import os

def load_yaml(file_path):
    class args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.save_dir = os.path.join(self.save_dir, format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return args(**yaml_data)

def get_args(rest_args):
    parser = argparse.ArgumentParser(description="Example script with argparse")
    parser.add_argument('--train', action='store_true', help='Train or not')
    parser.add_argument('--pretrain', action='store_true', help='Pre Train or not')
    parser.add_argument('--ddp', action='store_true', help='Training with distributed data parallel or not')
    parser.add_argument('--pre_epoch', type=int, default=0, help='The checkpoint ID for pretrain')
    parser.add_argument('--epoch', type=int, default=1000, help='Epochs')
    parser.add_argument('--init_epoch', type=int, default=200, help='Epochs for initialization')
    parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
    parser.add_argument('--lag', type=int, default=10, help='Lag value')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data or not')
    parser.add_argument('--time_varying', action='store_true', help='Time-varying or static causal discovery.')
    parser.add_argument('--sparse', action='store_true', help='Sparse matrix or not')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev', help='Noise type')
    parser.add_argument('--seed', type=int, default=69, help='Random seed')
    parser.add_argument('--gt_init', action='store_true', help='Initialization with ground truth or not')
    # model
    parser.add_argument('--embedding_dim', type=int, default=5, help='Embedding dim for t')
    parser.add_argument('--spectral_norm', action='store_true', help='Apply spectral norm to model')
    parser.add_argument('--tol', type=float, default=0, help='Tolerance')
    parser.add_argument('--graph_thres', type=float, default=0.3, help='Threshold for generating dag')
    
    parser.add_argument('--DAG', type=float, default=0.8, help='Lambda DAG.')
    # save
    parser.add_argument('--save_dir', type=str, default='./results/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), help='Saving directory')
    # generate data
    parser.add_argument('--num', type=int, default=1000, help='Number parameter')
    parser.add_argument('--scale', type=float, default=0.5, help='Variance for gaussian distribution')
    parser.add_argument('--pi', type=float, default=10, help='For DGP,sin(i/pi)')
    parser.add_argument('--distance', type=int, default=2, help='Distance of the largest edge')
    parser.add_argument('--max_d_L', type=int, default=1, help='Number of maximum latent variable')
    parser.add_argument('--d_L', type=int, default=1, help='Number of latent variable')
    parser.add_argument('--d_X', type=int, default=10, help='Number of measured variable')
    
    parser.add_argument('--degree', type=int, default=2, help='Graph degree')
    parser.add_argument('--condition', type=str, default='golem', help='Causal graph condition')
    # training lr
    parser.add_argument('--decay_type', type=str, default='step', choices=['step', 'multi step', 'cosine'], help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD', 'RMSprop'], help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gradient_noise', type=float, default=None, help='Variance of gradient noise')
    # step decay, (or cosine annealing decay)
    parser.add_argument('--step_size', type=int, default=1000, help='After step_size epochs, lr = gamma * lr, or cosine annealing start')
    parser.add_argument('--gamma', type=float, default=0.5, help='After step_size epochs, lr = gamma * lr')
    # multi step decay
    parser.add_argument('--decay', type=list, default=[200, 400, 800, 1000], help='Epoch milestones for lr = gamma * lr')
    # Adam, RMSprop
    parser.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999], help='Betas for optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for optimizer')
    # SGD
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    # vae
    parser.add_argument('--vae_subsample_elbos', type=int, default=128)
    # loss
    parser.add_argument('--kl_coeff', type=float, default=0.1, help='weight for the KL term')
    parser.add_argument('--reconstruction_loss_coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--sparsity_Bt', type=float, default=0.001, help='Sparsity of causal effects in measured variables.')
    parser.add_argument('--sparsity_Ct', type=float, default=0.001, help='Sparsity of causal effects from latent to measured.')
    parser.add_argument('--sparsity_Ct_1', type=float, default=0.001, help='Sparsity of causal effects in from time-lag to latent.')
    return parser.parse_args()