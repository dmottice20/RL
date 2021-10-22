import torch as to
import numpy as np
import time
import itertools
import os

# Import the data.
if os.getcwd().split('/')[-1] == 'taxiproblem':
    P = to.load('data/transition_matrices.pt')
    r = to.load('data/reward_vectors.pt')
else:
    P = to.load('src/hw1/taxiproblem/data/transition_matrices.pt')
    r = to.load('src/hw1/taxiproblem/data/reward_vectors.pt')

P = P.numpy()
r = r.numpy()
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
S_t = S_t.numpy()

# Construct action space.
A = {}
for k in range(25 + 1):
    if k != 25:
        A[k] = [0, 1, 2, 3, 4, 5]
    else:
        A[k] = [6]


def run_policy_iteration(P, r, A, S, lamb):
    """

    :param P: transition probability matrix of shape (card_a, card_s, card_s)
    :param r: reward vectors of shape (card_a, card_s, 1)
    :param A: action space (in this ex. not state dependent)
    :param S: state space
    :param lamb: discount factor for IH MDP
    :return: NOT SURE YET
    """
    print("Running policy iteration...")
    start = time.time()
    card_s = S_t.shape[0]
    d = np.zeros((card_s, 1))
    dtm1 = d.copy()
    # Execute policy iteration
    for s in range(card_s):
        QsaBest = -np.inf
        for a in A[S[s][0]]:
            # print(a)
            Qsa = r[a - 1][s - 1]
            # Update best alternative
            if Qsa > QsaBest:
                QsaBest = Qsa
                d[s - 1] = a

    n = 0
    rd = 0 * r[0]
    v = 0
    Pd = np.zeros((card_s, card_s))

    while not np.array_equal(d, dtm1):
        dtm1 = d.copy()
        for s in range(card_s):
            Pd[s - 1, :] = P[int(d[s - 1]) - 1][s - 1, :]
            rd[s - 1] = r[int(d[s - 1]) - 1][s - 1]

        # Policy Evaluation...
        v = np.linalg.solve(np.identity(card_s) - lamb * Pd, rd)

        # Policy Improvement
        for s in range(card_s):
            QsaBest = -np.inf
            for a in A[S[s][0]]:
                Qsa = r[a - 1][s - 1] + lamb * P[a - 1][s - 1, :] @ v
                if Qsa > QsaBest:
                    QsaBest = Qsa.copy()
                    d[s - 1] = a

        n += 1

    end = time.time()

    return v, d, Pd, end - start


v_star, d_star, Pd, timer = run_policy_iteration(P, r, A, S_t, lamb=0.95)

# Grab the possible initial starting states.
starting_states = []
for i in range(401):
    if S_t[i][1].item() < 4:
        starting_states.append(i)

print('Answer to part b (i). is...')
print(np.max(v_star[starting_states]))
print('Answer to part b (iii). is ...')
print(np.mean(v_star[starting_states]))
