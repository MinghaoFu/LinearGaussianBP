import logging

import igraph as ig
import networkx as nx
import numpy as np
import os

from Caulimate import *
from .generate_non_linear_data import generate_data as generate_non_linear_data

class SyntheticDataset:
    """Generate synthetic data.

    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, args, n, max_d_L, d_L, d_X, graph_type, condition, degree, noise_type, C_scale, B_scale, seed=1):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            graph_type ('ER' or 'SF'): Type of graph.
            degree (int): Degree of graph.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            B_scale (float): Scaling factor for range of B.
            seed (int): Random seed. Default: 1.
        """
        self.args = args
        self.n = n
        self.max_d_L = max_d_L
        self.d_L = d_L
        self.d_X = d_X
        self.graph_type = graph_type
        self.condition = condition
        self.degree = degree
        self.noise_type = noise_type
        self.C_ranges = ((C_scale * -2.0, C_scale * -0.5),
                         (C_scale * 0.5, C_scale * 2.0))
        self.B_ranges = ((B_scale * -2.0, B_scale * -0.5),
                         (B_scale * 0.5, B_scale * 2.0))
        self.rs = np.random.RandomState(seed)    # Reproducibility
        self.seed = seed

        self._setup()
        self._logger.debug("Finished setting up dataset class.")

    def _setup(self):
        """Generate B_bin, B and X."""
        if self.args.condition == 'non-linear':
            if self.args.load_data == True:
                self.Bs_bin = np.load(os.path.join(self.args.data_path, 'Bs_bin.npy'))
                self.Bs = np.load(os.path.join(self.args.data_path, 'Bs.npy'))
                self.X = np.load(os.path.join(self.args.data_path, 'X.npy'))
                self.L = np.load(os.path.join(self.args.data_path, 'L.npy'))
                self.Cs = np.load(os.path.join(self.args.data_path, 'Cs.npy'))
                self.BCs = np.load(os.path.join(self.args.data_path, 'BCs.npy'))
                self.Xh = np.load(os.path.join(self.args.data_path, 'Xh.npy'))
            else:
                # TODO check if Bs_bin is DAGs
                self.Bs, self.Bs_bin = SyntheticDataset.simulate_time_varying_graph(self.d_X, self.n, self.args, self.rs)
                Is = np.repeat(np.eye(self.d_X)[np.newaxis, :, :], self.n, axis=0)
                self.L, self.Xh, Ub, Mb, Lb, self.Cs = generate_non_linear_data(self.args, 1, self.n, self.d_L, self.d_X, noisy=1)
                self.BCs = np.concatenate((np.zeros((self.n, self.d_L, self.d_L + self.d_X)), np.concatenate((self.Cs, self.Bs), axis=2)), axis=1)
                self.X = np.matmul(np.linalg.inv(Is - self.Bs), np.expand_dims(self.Xh, axis=2))
                self.X = np.squeeze(self.X, axis=2)
                # save data if generate tn the first time
                makedir(self.args.data_path, remove_exist=True)
                np.save(os.path.join(self.args.data_path, 'Bs_bin.npy'), self.Bs_bin)
                np.save(os.path.join(os.path.join(self.args.data_path, 'Bs_bin.npy'), self.Bs))
                np.save(os.path.join(self.args.data_path, 'X.npy'), self.X)
                np.save(os.path.join(self.args.data_path, 'L.npy'), self.L)
                np.save(os.path.join(self.args.data_path, 'Cs.npy'), self.Cs)
                np.save(os.path.join(self.args.data_path, 'BCs.npy'), self.BCs)
                np.save(os.path.join(self.args.data_path, 'Xh.npy'), self.Xh)
                
            return self.Cs.shape[0], self.Cs.shape[1], self.Cs.shape[2]
            
        elif self.condition == 'sufficient':
            '''
                X = BX + CL + E 
            '''
            self.C_bin = np.ones((self.d_X, self.max_d_L))
            #self.B_bin = SyntheticDataset.simulate_chain_graph(self.d_X, self.rs)
            self.B_bin = SyntheticDataset.simulate_random_dag(self.d_X, self.degree, self.graph_type, self.rs)
        elif self.condition == 'necessary':
            self.B_bin = SyntheticDataset.simulate_chain_graph(self.d_X, self.rs)
            #self.B_bin = SyntheticDataset.simulate_random_dag(self.d_X, self.degree, self.graph_type, self.rs)
            inds = [[0, 1, 2, 3], [2, 3, 4, 5]]#sample_n_different_integers(self.d_L * 4, 0, self.d_X, self.seed).reshape(self.d_L, -1)
            self.C_bin = np.zeros((self.d_X, self.max_d_L))
            for idx, ind in enumerate(inds):
                self.C_bin[ind, idx] = 1
            for ind in inds:
                for i in ind:
                    for j in ind:
                        self.B_bin[i, j] = 0

        elif self.condition == 'golem':
            if self.args.load_data == True:
                self.X = np.load('./dataset/X.npy')
                self.Bs = np.load('./dataset/Bs.npy')
            else:
                self.B_bin = SyntheticDataset.simulate_random_dag(self.d_X, self.degree,
                                                                self.graph_type, self.rs)
                #self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs)
                self.Bs, self.B_bin = SyntheticDataset.simulate_time_varying_graph(self.d_X, self.n, self.args, self.rs)
                self.X = np.zeros((self.n, self.d_X))
                for i in range(self.n):
                    self.X[i:i+1, :] = SyntheticDataset.simulate_linear_sem(self.Bs[i], 1, self.noise_type, self.rs)
                #self.X = np.matmul(self.X, np.linalg.inv(np.eye(self.d_X) - self.Bs))
                makedir('./dataset', remove_exist=True)
                np.save('./dataset/X.npy', self.X)
                np.save('./dataset/Bs.npy', self.Bs)
            
            return # skip the following steps
        elif self.condition == 'Silva':
            self.C_bin = SyntheticDataset.simulate_random_dag(self.d_L, self.degree,
                                                            self.graph_type, self.rs)
            self.C = SyntheticDataset.simulate_weight(self.C_bin, self.C_ranges, self.rs)
            assert is_dag(self.C)
            self.L = SyntheticDataset.simulate_linear_sem(self.C, self.n, self.noise_type, self.rs)
            self.B_bin = SyntheticDataset.simulate_random_Silva_C(self.d_X, self.d_L, self.rs)
            self.B = SyntheticDataset.simulate_weight(self.B_bin, self.B_ranges, self.rs)
            self.X = SyntheticDataset.simulate_linear_fa(self.B, self.L, self.n, self.noise_type, self.rs)
        elif self.condition == 'customized':
            self.B_bin = SyntheticDataset.simulate_random_dag(self.d_X, self.degree,
                                                            self.graph_type, self.rs)
            self.C = SyntheticDataset.simulate_weight(self.C_bin, self.C_ranges, self.rs)

            self.L = SyntheticDataset.simulate_linear_sem(self.C, self.n, self.noise_type, self.rs)
            self.X = SyntheticDataset.simulate_linear_fa(self.B, self.L, self.n, self.noise_type, self.rs)
        else:
            raise ValueError(self.condition)
        
        #assert is_dag(self.B_bin)
        
        
        I_B = np.eye(self.d_X)    
        self.EL_cov = np.eye(self.max_d_L)
        self.L = self.rs.multivariate_normal(np.zeros(self.max_d_L), self.EL_cov, size=self.n)
        self.C = SyntheticDataset.simulate_weight(self.C_bin, self.C_ranges, self.rs)
        self.BC = np.concatenate((np.zeros((self.max_d_L, self.max_d_L + self.d_X)), np.concatenate((self.C, self.B), axis=1)), axis=0)
        self.EX_cov = np.eye(self.d_X) 
        self.EX = np.random.multivariate_normal(np.zeros(self.d_X), self.EX_cov, size=self.n)
        self.X = ((self.C @ self.L.T).T + self.EX) @ np.linalg.inv(I_B - self.B)
            
        print(f'--- norm(estimated covariance - data covariance) = {self.cal_population_sample_cov()}')
            
    def cal_population_sample_cov(self):
        X_cov = np.cov(self.X.T)
        if self.d_L == 0:
            I = np.eye(self.d_X)
            est_X_cov = np.linalg.inv(I - self.B).T @ np.linalg.inv(I - self.B)
            dis = np.linalg.norm(est_X_cov - X_cov)
        else:
            I = np.eye(self.d_X)
            est_X_cov = np.linalg.inv(I - self.B).T @ (self.C @ self.C.T + self.EX_cov) @ np.linalg.inv(I - self.B)
            dis = np.linalg.norm(est_X_cov - X_cov)
        return dis
    
    @staticmethod 
    def simulate_time_varying_graph(d, n, args, rs=np.random.RandomState(1)):
        B_bin = SyntheticDataset.simulate_sparse_graph(d, 0.1, rs)
        Bs = np.expand_dims(args.synthetic_scale * np.cos(np.arange(n) / 100), axis=(1,2)) #np.repeat(np.random.uniform(-0.45, 0.45, size=(d, d))[None, :, :], n, axis=0) +
        Bs = Bs * B_bin
        Bs[np.abs(Bs) < args.synthetic_scale * args.synthetic_B_thres] = 0
        
        Bs_bin = Bs.copy()
        Bs_bin[np.abs(Bs_bin) != 0] = 1
        
        return Bs, Bs_bin
        
    @staticmethod
    def simulate_sparse_graph(d, edge_ub, rs=np.random.RandomState(1)):
        G = nx.gnp_random_graph(d, edge_ub, directed=True)
        while not nx.is_directed_acyclic_graph(G):
            G = nx.gnp_random_graph(d, edge_ub, directed=True)
        B = nx.adjacency_matrix(G).toarray()
        
        return B
    
    @staticmethod
    def simulate_chain_graph(d, rs=np.random.RandomState(1)):
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P
        G = nx.DiGraph()
        num_nodes = d
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)
        B_bin = np.array(nx.to_numpy_matrix(G, dtype=int))
        return _random_permutation(B_bin)
    
    @staticmethod
    def simulate_er_dag(d, degree, rs=np.random.RandomState(1)):
        """Simulate ER DAG using NetworkX package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _get_acyclic_graph(B_und):
            return np.tril(B_und, k=-1)

        def _graph_to_adjmat(G):
            return nx.to_numpy_matrix(G)

        p = float(degree) / (d - 1)
        G_und = nx.generators.erdos_renyi_graph(n=d, p=p, seed=rs)
        B_und_bin = _graph_to_adjmat(G_und)    # Undirected
        B_bin = _get_acyclic_graph(B_und_bin)
        return B_bin

    @staticmethod
    def simulate_sf_dag(d, degree):
        """Simulate ER DAG using igraph package.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        m = int(round(degree / 2))
        # igraph does not allow passing RandomState object
        G = ig.Graph.Barabasi(n=d, m=m, directed=True)
        B_bin = np.array(G.get_adjacency().data)
        return B_bin
    
    @staticmethod
    def simulate_random_Silva_C(d_X, d_L, rs=np.random.RandomState(1)):
        """Simulate random relation from latent confounder to observed variable with Silva assumption.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        B = np.zeros((d_X, d_L))
        rp = sorted([rs.randint(0, d_X - d_L * 3) for _ in range(d_L - 1)])
        rp.append(-1)
        j = 0
        for i in range(d_X):
            B[i, j] = 1
            if i == rp[j] + 3:
                j += 1

        rs.shuffle(B)
    
        return B.T
    
    @staticmethod
    def simulate_random_dag(d, degree, graph_type, rs=np.random.RandomState(1)):
        """Simulate random DAG.

        Args:
            d (int): Number of nodes.
            degree (int): Degree of graph.
            graph_type ('ER' or 'SF'): Type of graph.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] binary adjacency matrix of DAG.
        """
        def _random_permutation(B_bin):
            # np.random.permutation permutes first axis only
            P = rs.permutation(np.eye(B_bin.shape[0]))
            return P.T @ B_bin @ P

        if graph_type == 'ER':
            B_bin = SyntheticDataset.simulate_er_dag(d, degree, rs)
        elif graph_type == 'SF':
            B_bin = SyntheticDataset.simulate_sf_dag(d, degree)
        else:
            raise ValueError("Unknown graph type.")
        return _random_permutation(B_bin)

    @staticmethod
    def simulate_weight(B_bin, B_ranges, rs=np.random.RandomState(1)):
        """Simulate the weights of B_bin.

        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = rs.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = rs.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

    @staticmethod
    def simulate_linear_sem(B, n, noise_type, rs=np.random.RandomState(1)):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """
        def _simulate_single_equation(X, B_i):
            """Simulate samples from linear SEM for the i-th node.

            Args:
                X (numpy.ndarray): [n, number of parents] data matrix.
                B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

            Returns:
                numpy.ndarray: [n,] data matrix.
            """
            if noise_type == 'gaussian_ev':
                # Gaussian noise with equal variances
                N_i = rs.normal(scale=1.0, size=n)
            elif noise_type == 'gaussian_nv':
                # Gaussian noise with non-equal variances
                scale = rs.uniform(low=1.0, high=2.0)
                N_i = rs.normal(scale=scale, size=n)
            elif noise_type == 'exponential':
                # Exponential noise
                N_i = rs.exponential(scale=1.0, size=n)
            elif noise_type == 'gumbel':
                # Gumbel noise
                N_i = rs.gumbel(scale=1.0, size=n)
            else:
                raise ValueError("Unknown noise type.")
            
            return X @ B_i + N_i

        d = B.shape[0]
        X = np.zeros([n, d])
        G = nx.DiGraph(B)
        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        for i in ordered_vertices:
            parents = list(G.predecessors(i))
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])
            
        return X
    
    @staticmethod
    def simulate_linear_fa(B, L, n, noise_type, rs=np.random.RandomState(1)):
        """Simulate samples from linear factor analysis with specified type of noise.

        Args:
            B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
            n (int): Number of samples.
            noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).

        Returns:
            numpy.ndarray: [n, d] data matrix.
        """
        d = B.shape[1]
        if noise_type == 'gaussian_ev':
            # Gaussian noise with equal variances
            N_i = rs.normal(scale=1.0, size=(n, d))
        elif noise_type == 'gaussian_nv':
            # Gaussian noise with non-equal variances
            scale = rs.uniform(low=1.0, high=2.0)
            N_i = rs.normal(scale=scale, size=(n, d))
        elif noise_type == 'exponential':
            # Exponential noise
            N_i = rs.exponential(scale=1.0, size=(n, d))
        elif noise_type == 'gumbel':
            # Gumbel noise
            N_i = rs.gumbel(scale=1.0, size=(n, d))
        else:
            raise ValueError("Unknown noise type.")
        return L @ B + N_i

def generate_data(args):
    n, max_d_L, d_L, d_X = args.num, args.max_d_L, args.d_L, args.d_X
    graph_type, degree, condition = 'ER', args.degree, args.condition   # ER2 graph
    C_scale = 1.0
    B_scale = 1.0
    noise_type = args.noise_type

    dataset = SyntheticDataset(args, n, max_d_L, d_L, d_X, graph_type, condition, degree,
                               noise_type, C_scale, B_scale, seed=args.seed)
    return dataset

if __name__ == '__main__':
    n, d_L, d_X = 5000, 3, 20
    graph_type, degree = 'SF', 2  # ER2 graph
    C_scale = 1.0
    B_scale = 1.0
    noise_type = 'gaussian_ev'

    dataset = SyntheticDataset(n, d_L, d_X, graph_type, degree,
                               noise_type, C_scale, B_scale, seed=1)
    print("dataset.L.shape: {}".format(dataset.L.shape))
    print("dataset.C.shape: {}".format(dataset.C.shape))
    print("dataset.C_bin.shape: {}".format(dataset.C.shape))
    print("dataset.X.shape: {}".format(dataset.X.shape))
    print("dataset.B.shape: {}".format(dataset.B.shape))
    print("dataset.B_bin.shape: {}".format(dataset.B.shape))
