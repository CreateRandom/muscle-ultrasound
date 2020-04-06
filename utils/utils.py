import operator
from functools import reduce

import numpy as np
import itertools
def repeat_to_length(elems, length):

    n_repeats = int(np.ceil(length / len(elems)))
    repeated = list(itertools.repeat(elems, n_repeats))
    repeated = flatten_list(repeated)
    return repeated[:length]

def flatten_list(elems):
    return reduce(operator.add, elems)
