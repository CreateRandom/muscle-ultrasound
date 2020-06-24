# for now hard code the type of problem based on the attribute
import random

import numpy as np
import torch

problem_kind = {'Sex': 'binary', 'Age': 'regression', 'BMI': 'regression',
            'Muscle': 'multi', 'Class': 'binary'}

problem_legal_values = {'Class': ['NMD', 'no NMD']}

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)