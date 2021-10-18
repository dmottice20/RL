import mdptoolbox
import os
import torch as to
import numpy as np

# Import the data.
if os.getcwd().split('/')[-1] == 'taxiproblem':
    P = to.load('data/transition_matrices.pt')
    R = to.load('data/reward_vectors.pt')
else:
    P = to.load('src/hw1/taxiproblem/data/transition_matrices.pt')
    R = to.load('src/hw1/taxiproblem/data/reward_vectors.pt')

print(type(P))
print(P.shape)
print(type(R))
print(R.shape)

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
        i+=1

pim = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.95)
pim.run()
pi = pim.policy
print(pi)
x = 5+ 5

