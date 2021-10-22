import torch as to
import numpy as np
from scipy.sparse import eye
import tqdm

# Based off of Sutto and Barto (2018), Example 6.6 (pg 132)
# ADDITIONAL: There is a chance the agent slips south (down) 
# after each action.

# STEP 0: Pre-processing.
# Create the state space.
# S = {0,1,...,38}
S = to.arange(39, dtype=to.int)
card_s = len(S)

# Create the action space.
# A = {0, ..., 3} where
# 0 is N, 1 is E, 2 is S,
# and 3 is W
A = to.arange(4, dtype=to.int)
card_a = len(A)

# Discount factor.
gamma = 1 - 1e10*np.spacing(1)

# Probability of slipping.
p_slip = 0.35

# Load transition probability matrix and reward function into memory.
gamP = dict()
r = dict()
for a in A:
    a = a.item()
    gamP[a] = to.zeros(card_s, card_s)
    r[a] = to.zeros(card_s, 1)

for s in S:
    s = s.item()
    for a in A:
        a = a.item()
        # No Slip!
        # If in terminal state, remain there.
        if s == 38:
            stp1 = 38
            gamP[a][s, stp1] = gamma
        elif s == 37:
            stp1 = 37
            gamP[a][s, stp1] = gamma
        else:
            # Go North...
            if a == 0:
                if s > 11:
                    stp1 = s - 12
                else:
                    stp1 = s
            # Go east...
            elif a == 1:
                if s not in [11, 23, 35]:
                    stp1 = s + 1
                else:
                    stp1 = s
            # Go south...
            elif a == 2:
                if s == 36:
                    stp1 = s 
                elif s == 35:
                    stp1 = 38
                elif s >= 25 and s <= 35:
                    stp1 = 37
                else:
                    stp1 = s + 12
            elif a == 3:
                if s not in [0, 12, 24, 36]:
                    stp1 = s - 1
                else:
                    stp1 = s
            else:
                raise Exception('Transition Function Error!')
        
            gamP[a][s, stp1] = gamma * (1 - p_slip)
            if stp1 == 37:
                r[a][s] = -100 * gamma * (1 - p_slip)
            else:
                r[a][s] = -gamma * (1 - p_slip)

            # Yes slip!
            pds = stp1
            if pds == 37 or pds == 38:
                stp1 = pds
            elif pds == 36:
                stp1 = pds
            elif pds == 35:
                stp1 = 38
            elif pds >= 25 and pds <= 34:
                stp1 = 37
            else:
                stp1 = pds + 12

            gamP[a][s, stp1] += gamma*p_slip
            if stp1 == 37:
                r[a][s] = r[a][s]-100*gamma*p_slip
            else:
                r[a][s] = r[a][s]-gamma*p_slip

# STEP 1: INITIALIZATION.
# Choose an initial policy.
d = to.zeros(card_s, 1, dtype=to.int)
dtm1 = d.clone()

# Initial policy improvement given the assuming zero v vector.
# Take all of the reward vectors for each vector and create a
# new matrix that is (card_s, card_a). i.e. (39, 4)
R = to.hstack((to.hstack((r[0], r[1])), to.hstack((r[2], r[3]))))
d = to.argmin(R, 1)
d = to.reshape(d, (39, 1))

# Initialize counter.
n = 0
rd = to.zeros(card_s, 1)

# Initialize the induced transition probability matrix.
Pd = 0*gamP[1]
# Initialize the state-action function.
Q = to.zeros(card_s, card_a)


def policy_evaluation(Pd, rd):
    return to.linalg.solve((to.from_numpy(eye(card_s).toarray()) - Pd).float(), rd)


while not to.equal(d, dtm1.long()):
    dtm1 = d.clone()

    for a in A:
        a = a.item()
        ind, _ = to.where(d == a)
        if len(ind) != 0:
            Pd[ind, :] = gamP[a][ind, :]
            rd[ind, 0] = R[ind, a]

    # Policy Evaluation.
    v = policy_evaluation(Pd, rd)
    # Policy Improvement.
    for a in A:
        a = a.item()
        Q[:, a] = R[:, a] + to.matmul(gamP[a], v).reshape(card_s)
    d = to.argmax(Q, 1)
    d = to.reshape(d, (card_s, 1))

    n += 1

Vstar_a = to.reshape(v[:36], (3, 12))
Vstar_b = to.hstack((v[37], to.zeros(11)))
Vstar = to.vstack((Vstar_a, Vstar_b.reshape(1, 12)))
print(Vstar)

# Save the vstar...
to.save(Vstar, 'Vstar_SCW.pt')
