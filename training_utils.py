# for now hard code the type of problem based on the attribute
import random

import numpy as np
import torch

problems = {'Sex': 'binary', 'Age': 'regression', 'BMI': 'regression',
            'Muscle': 'multi', 'Class': 'multi'}


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)