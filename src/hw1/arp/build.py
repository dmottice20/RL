import pandas as pd
import torch as to
import numpy as np
import os
from tqdm import tqdm


##############################################################
# This script loads the problem in a standard computer form. #
##############################################################
print(os.getcwd())
print(to.get_num_threads())

# Read in the data.
data = to.as_tensor(pd.read_csv('data/ARP_3650_data.csv').values)
print("The shape of this loaded data is...", data.shape)
c, v, o, p = to.hsplit(data, 4)

M = 3650

# Define the state space.
S = to.arange(1, M+1)
card_s = len(S)

# Define the action space.
A = to.arange(1, M+1+1)
card_a = len(A)

# Construct transition probability matrices and reward vectors.
P = to.zeros(size=(card_a, card_s, card_s))
r = to.zeros(size=(card_a, card_s, 1))

# ACTION 1 MATRIX
indices = to.zeros(2, 2*3650)

for s in S:
    if s < M - 1:
        indices[0, s - 1] = s - 1
        indices[1, s - 1] = s
    elif s == M - 1:
        indices[0, s - 1] = M - 1 - 1
        indices[1, s - 1] = M - 1
    elif s == M:
        indices[0, s - 1] = M - 1
        indices[1, s - 1] = M - 1

for s in S:
    if s < M:
        indices[0, s + 3650 - 1] = s - 1
        indices[1, s + 3650 - 1] = M - 1
    else:
        indices[0, M + 3650 - 1] = s - 1
        indices[1, M + 3650 - 1] = M - 1

indices = indices.type(to.IntTensor)

values1 = [p[s-1].item() for s in S]
values1[-1], values1[-2] = 1, 1
values2 = [1 - p[s-1].item() for s in S]
values2[-1], values2[-2] = 0, 0
values = np.append(np.array(values1), np.array(values2))

P_action1 = to.sparse_coo_tensor(indices, values.reshape(7300, 1))
P[0] = P_action1

# BUILD ACTION 3651 : purchase oldest car possible.
indices = to.zeros(2, 3650)
for s in S:
    indices[0, s - 1] = s - 1
    indices[1, s - 1] = M - 1

values = to.ones(3650, 1)
P_actionlast = to.sparse_coo_tensor(indices, values)
P[M] = P_actionlast

# Build actions 2,...N,...M
# In python.... 1,...,N,...M-1
for a in tqdm(range(1, M), desc='building transition matrices for actions 2 to M'):
    # Two non-zero elements per row.
    indices = to.zeros(2, 3650*2)
    # p(s) for the taken action to next.
    for s in S:
        indices[0, s-1] = s - 1
        indices[1, s-1] = a - 2 + 1
    # p of failing for taking an action.
    for s in S:
        indices[0, s+3650-1] = s - 1
        indices[1, s+3650-1] = M - 1
    # The actual values.
    values1 = to.from_numpy(np.repeat(p[a-2].item(), 3650))
    values2 = to.from_numpy(np.repeat(1-p[a-2].item(), 3650))
    values = np.append(values1.numpy(), values2.numpy())

    # Add to the overall P matrix.
    P_actionN = to.sparse_coo_tensor(indices, values.reshape(7300, 1))
    P[a] = P_actionN

# Save the data.
to.save(P, 'data/arp_transition_tensor.pt')
to.save(r, 'data/arp_rewards_tensor.pt')
