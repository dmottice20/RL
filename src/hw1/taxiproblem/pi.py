import torch as to
import numpy as np
import timeit
import itertools
import os

# Import the data.
if os.getcwd().split('/')[-1] == 'taxiproblem':
    P = to.load('data/transition_matrices.pt')
    r = to.load('data/reward_vectors.pt')
else:
    P = to.load('src/hw1/taxiproblem/data/transition_matrices.pt')
    r = to.load('src/hw1/taxiproblem/data/reward_vectors.pt')


# Rebuild state and action spaces.
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
card_s = S_t.shape[0]

# Construct action space.
A = to.arange(7, dtype=to.int8)
card_a = A.shape[0]

start = timeit.timeit()

# Choose an initial policy.
d = to.zeros(size=(card_s,1), dtype=to.int)
dtm1 = d.clone()

for s in range(card_s):
    QsaBest = -np.inf
    for a in A:
        a = a.item()
        Qsa = r[a-1][s]
        if Qsa > QsaBest:
            QsaBest = Qsa
            d[s] = a

n = 0
rd = 0 * r[0]
v = 0
Pd = to.zeros(size=(card_s, card_s))
gamma = 0.95

while not to.equal(d, dtm1):
    dtm1 = d.clone()
    for s in range(card_s):
        Pd[s, :] =P[d[s].item()][s, :]
        rd[s] = r[d[s].item()][s]

    # Policy Evaluation...
    v = to.linalg.solve(to.eye(card_s) - gamma * Pd, rd)

    # Policy improvement...
    for s in range(card_s):
        QsaBest = -np.inf
        for a in A:
            a = a.item()
            Qsa = r[a][s] + gamma * to.matmul(P[a][s, :], v)
            if Qsa > QsaBest:
                QsaBest = Qsa.clone()
                d[s] = a
    print(n)
    n += 1

end = timeit.timeit()

print('Value fx...\n', v)
print('Decision rule...\n', d)



