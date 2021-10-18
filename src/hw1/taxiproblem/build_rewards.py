import torch as to
import itertools
from tqdm import tqdm

# Construct state space.
# Assume the following layout.
#  1    2 | 3    4    5
#  6    7 | 8    9   10
# 11   12  13   14   15
# 16 | 17  18 | 19   20
# 21 | 22  23 | 24   25
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

# Construct the action space.
A = to.arange(7, dtype=to.int8)
card_a = A.shape[0]
# where 0 is L, i.e. k_t - 1,
#       1 is R, i.e. k_t + 1,
#       2 is U, i.e. k_t - 5,
#       3 is D, i.e. k_t + 5,
#       4 is picking up,
#       5 is dropping off,
#       6 is do nothing.

# Create a set of locations where 0,1,2,3
# correspond to R, G, Y, and B respectively.
locations = {
    0: 0,
    1: 4,
    2: 20,
    3: 23,
}

# Create a set of barriers.
# Each key is the state location where movement is restricted.
# Each value is
barriers = {
    1: 1,
    2: 0,
    6: 1,
    7: 0,
    15: 1,
    20: 1,
    16: 0,
    21: 0,
    17: 1,
    22: 1,
    18: 0,
    23: 0
}

# Construct the transition matrices.
r = to.zeros(size=(card_a, card_s, 1))
for i in tqdm(range(card_s), desc='building reward vectors'):
    for a in A:
        a = a.item()
        if i != 400:
            # For the kinematic actions...
            # i.e. 0,1,2,3
            if a < 4:
                r[a][i] = -1
            # for the pick-up actions...
            elif a == 4:
                # Is it a good or bad pick-up action?
                # Is a customer already in the car? That's bad.
                if S_t[i][1].item() == 4:
                    r[a][i] = -10
                # Is the taxi at the right location? If not, that's bad.
                elif S_t[i][0].item() != locations[S_t[i][1].item()]:
                    r[a][i] = -10
                else:
                    r[a][i] = -1
            # for the drop off actions...
            elif a == 5:
                # Is it a good or bad drop-off action?
                # Is a customer even onboard? If not, that's bad
                if S_t[i][1].item() != 4:
                    r[a][i] = -10
                # Is the taxi at the correct destination? If not that's bad.
                elif S_t[i][0].item() != locations[S_t[i][2].item()]:
                    r[a][i] = -10
                else:
                    r[a][i] = 20
            # for the do nothing actions...
            else:
                r[a][i] = -1
        else:
            r[a][i] = 0


to.save(r, 'data/reward_vectors.pt')
print('EL FIN!')

