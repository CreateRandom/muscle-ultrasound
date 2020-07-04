import random

import numpy as np
import torch


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)