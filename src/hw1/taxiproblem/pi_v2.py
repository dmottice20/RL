import mdptoolbox
import os
import torch as to
import numpy as np
import itertools
from ismember import ismember

# Import the data.
if os.getcwd().split('/')[-1] == 'taxiproblem':
    P = to.load('data/transition_matrices.pt')
    R = to.load('data/reward_vectors.pt')
else:
    P = to.load('src/hw1/taxiproblem/data/transition_matrices.pt')
    R = to.load('src/hw1/taxiproblem/data/reward_vectors.pt')

P = P.numpy()
R = R.numpy()
R = R.reshape((7, 401))
R = R.transpose()

for a in range(7):
    i = 0
    for row in P[a]:
        if sum(row) != 1:
            print('row {} of action {} does not sum to 1.'.format(i, a))
            print(sum(row))
        i += 1

pim = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.95)
pim.run()
pi = np.array(pim.policy)

r = R.transpose()
print(r.shape)
Pd = np.zeros((401, 401))
rd = np.zeros((401, 1))
for s in range(len(pi)):
    Pd[s, :] = P[pi[s]][s, :]
    rd[s] = r[pi[s]][s]


def calculate_limiting_distribution(Pd):
    """
    :param Pd - induced DTMC from optimal policy
    :return pi_d - limiting distribution of Pd
    """
    # Right now -- this is for irreducible which is wrong.
    A_top_row = np.identity(Pd.shape[0]) - Pd.transpose()
    A_bottom_row = np.repeat(1, Pd.shape[0])
    A = np.vstack((A_top_row, A_bottom_row))
    A = np.delete(A, 0, axis=0)
    b = np.vstack((np.asarray([np.repeat(0, Pd.shape[0] - 1)]).transpose(), [1]))
    pi = np.matmul(np.linalg.inv(A), b)

    return pi

S_ta = to.as_tensor(list(itertools.product(to.arange(25), to.arange(5), to.arange(4))))
S_ta = to.vstack((S_ta, to.as_tensor([25, 5, 5])))
card_s = S_ta.shape[0]

# Loop over the states and remove any where element 2 == element 3.
# i.e. remove the states where the destination equals the customer's location.
S_t = to.zeros(size=(401, 3), dtype=to.int)
j = 0
for i in range(card_s):
    if S_ta[i, 1] == S_ta[i, 2]:
        continue
    else:
        S_t[j, :] = S_ta[i, :]
        j += 1

S_t[-1] = to.tensor([25, 5, 5])
# Grab the possible initial starting states.
starting_states = []
for i in range(401):
    if S_t[i][1].item() < 4:
        starting_states.append(i)


pi_d = calculate_limiting_distribution(Pd)

V = np.array([pim.V]).reshape(401, 1)

print('Answer to part b (i). is...')
print(np.max(V[starting_states]))
print('Answer to part b (iii). is ...')
print(np.mean(V[starting_states]))

# T = min{n >= 0: X_n = DELTA}
# where delta is index 400.
# m_i = E(T | X_0 = i)
# m = (I - B)^-1 e
# where B is square matrix with index 400 removed.
# and e is vector of ones size (400, 1)
B = np.delete(Pd, 400, 0)
B = np.delete(B, 400, 1)
e = np.ones((B.shape[0], 1))
m = np.matmul(np.linalg.inv(np.identity(B.shape[0]) - B), e)
print('Expected actions to terminal...{}'.format(np.mean(m[starting_states])))


