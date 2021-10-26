import torch as to
import itertools
import numpy as np
from system_model import system_model, grab_key


# Assume the following layout.
#  0    1 | 2    3    4
#  5    6 | 7    8    9
# 10   11  12   13   14
# 15 | 16  17 | 18   19
# 20 | 21  22 | 23   24
state_space = to.as_tensor(list(itertools.product(to.arange(25), to.arange(5), to.arange(4))))
state_space = to.vstack((state_space, to.as_tensor([25, 5, 5])))
card_s = state_space.shape[0]

print('\n---- STATE SPACE ----')
print('\n{}\n\n'.format(state_space))
print('State space has cardinality = {}'.format(card_s))

# Hash table to store the index to state mapping.
indices = dict(zip(np.arange(card_s), state_space))


