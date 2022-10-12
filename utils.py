import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import preprocess_adult_data

        
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def check(g, y, n_group):
    idx = len(np.unique(g * 2 + y))
    is_full = (True if idx == n_group else False)
    return is_full


def read_dataset(
    p_l=1, 
    p_u=1, 
    seed = 0,
    which = 'adult', 
    attr = 'sex', 
    full_attr = True, 
    contain_group = False,
):
    
    data = preprocess_adult_data(attr = attr)
        
    x, y, g_id = data
    y = y.reshape(-1, 1)
    g = np.argmax(x[:, g_id], 1).reshape(-1, 1)
    n = y.shape[0]
    n_group = len(np.unique(g))
    
    nl = p_l if p_l > 1 else int(n*0.07*p_l)
    nu = p_u if p_l > 1 else int(n*0.63*p_u)
    nt = int(n*0.3)
    assert nl+nu <= int(n*0.7), f"label + unlabel > {int(n*0.7)}"
    
    same_seeds(seed)
    while True:
        x_t, x_r, y_t, y_r, g_t, g_r = train_test_split(
            x, y, g, 
            train_size=nt, 
            test_size=n-nt,
        )
        if check(g_t, y_t, n_group*2):
            break

    while True:
        x_l, x_u, y_l, y_u, g_l, g_u = train_test_split(
            x_r, y_r, g_r, 
            train_size=nl, 
            test_size=nu,
        )
        
        if check(g_u, y_u, n_group*2):
            if not full_attr:
                break
            elif check(g_l, y_l, n_group*2):
                break
            
    # delete the columns whose values are all the same.        
    std = x_l.std(0)
    use_id = np.where(std != 0)[0]
    if not contain_group:
        use_id = [i for i in use_id if i not in g_id]
        
    return {
        'x': (x_l[:, use_id], x_u[:, use_id], x_t[:, use_id]), 
        'g': (g_l, g_u, g_t), 
        'y': (y_l, y_u, y_t), 
        'n': (nl, nu, nt), 
    }


def evaluate(pred, y, group, metric):
    acc = (pred == y).mean()
    
    ns = len(np.unique(group))
    group = np.eye(ns)[group.flatten()].T
    
    if metric == 'dp':
        cond = np.ones((y.shape[0], 1))
    elif metric == 'opp':
        cond = y
    else:
        cond = np.c_[1- y, y]
    
    fair = 0
    for i in range(cond.shape[1]):
        cond_i = cond[:, [i]]
        value = (group @ (cond_i * pred)) / (group @ cond_i)
        diff = value.max() - value.min()
        fair += diff * cond_i.sum() / cond.sum()
    
    return acc, fair