import torch
import torch.nn as nn

import numpy as np


from .base import check_tensor

def sample_n_different_integers(n, low, high, random_seed=None):
    # Create a random number generator with a specified random seed (or without)
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Check if the interval contains enough unique integers
    if high - low < n:
        raise ValueError("Interval does not contain enough unique integers.")

    # Create an array of all integers in the interval
    all_integers = np.arange(low, high)

    # Shuffle the integers and take the first 'n' as the sample
    rng.shuffle(all_integers)
    sampled_integers = all_integers[:n]
    
    return sampled_integers

def top_k_abs_tensor(tensor, k):
    d = tensor.shape[0]
    abs_tensor = torch.abs(tensor)
    _, indices = torch.topk(abs_tensor.view(-1), k)
    
    flat_tensor = tensor.view(-1)
    flat_zero_tensor = torch.zeros_like(flat_tensor)
    flat_zero_tensor[indices] = flat_tensor[indices]
    
    zero_tensor = check_tensor(flat_zero_tensor.view(d, d))
    
    
    # batch_size, d, _ = tensor.shape
    # values, indices = torch.topk(tensor.view(batch_size, -1), k=k, dim=-1)
    # result = torch.zeros_like(tensor).view(batch_size, -1)
    # result.scatter_(1, indices, values)
    # result = result.view(batch_size, d, d)
    return zero_tensor

def random_zero_array(arr, zero_ratio, constraint=None):
    '''
        Randomly set some elements in an array to 0
    '''
    if constraint is None:
        original_shape = arr.shape
        arr = arr.flatten()
        inds = np.random.choice(np.arange(len(arr)), size=int(len(arr) * zero_ratio), replace=False)
        arr[inds] = 0
        result = arr.reshape(original_shape)
    return result
