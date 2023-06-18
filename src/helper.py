import numpy as np
import torch
import random
import os
import time
import beartype
from beartype import beartype

def exists(val):
    return val is not None

def beartype_jit(func):
    """decorator to enable beartype only if USE_BEARTYPE is set to 1"""
    return beartype(func) if os.environ.get('USE_BEARTYPE', '0') == '1' else func


def get_set_seed():
    SEED = int(time.time()) #35202   #35202  #123  #
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    return SEED