import torch as to
import itertools
from tqdm import tqdm


# Construct the state space...
# Assume the following layout.
#  0    1 | 2    3    4
#  5    6 | 7    8    9
# 10   11  12   13   14
# 15 | 16  17 | 18   19
# 20 | 21  22 | 23   24
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

# Create a set of locations where 0,1,2,3
# correspond to R, G, Y, and B respectively.
locations = {
    0: 0,
    1: 4,
    2: 20,
    3: 23,
}

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

# Construct the state dependent action space.
# where 0 is L, 1 is R, 2 is U, 3 is D,
# 4 is pickup, 5 is drop-off, and 6 is do nothing.
A = {
    0: [1, 3, 4, 5],
    1: [0, 1, 3, 4, 5],
    2: [1, 3, 4, 5],
    3: [0, 1, 3, 4, 5],
    4: [0, 3, 4, 5],
    5: [1, 2, 3, 4, 5],
    6: [0, 2, 3, 4, 5],
    7: [1, 2, 3, 4, 5],
    8: [0, 1, 2, 3, 4, 5],
    9: [0, 2, 3, 4, 5],
    10: [1, 2, 3, 4, 5],
    11: [0, 1, 2, 3, 4, 5],
    12: [0, 1, 2, 3, 4, 5],
    13: [0, 1, 2, 3, 4, 5],
    14: [0, 2, 3, 4, 5],
    15: [2, 3, 4, 5],
    16: [1, 2, 3, 4, 5],
    17: [0, 2, 3, 4, 5],
    18: [1, 2, 3, 4, 5],
    19: [0, 2, 3, 4, 5],
    20: [2, 4, 5],
    21: [1, 2, 4, 5],
    22: [0, 2, 4, 5],
    23: [1, 2, 4, 5],
    24: [0, 2, 4, 5],
    25: [6]
}

P = to.zeros(size=(7, card_s, card_s))
r = to.zeros(size=(7, card_s))

# LIZ STATE : (taxi row, taxi col, passenger location, destination)
#                   1,     2,                3,           4
# MY STATE: (taxi location, passenger location, destination)
#                   0,              1,              2
for j in range(card_s):
    for a in A[S_t[j][0].item()]:
        if a == 6:
            P[a][j, j] = 1
            r[a][j] = -1
            r[a][card_s-1] = 0
        # Drop-off actions...
        elif a == 5 and S_t[j][1].item() == 4:
            # Check if taxi is at correct location.
            if S_t[j][0].item() == locations[S_t[j][2].item()]:
                # If here, it is successful.
                # Move to terminal state...
                P[a][j, card_s - 1] = 1
                r[a][j] = r[a][j] + 20
            else:
                # If here it is unsuccessful.
                P[a][j, j] = 1
                r[a][j] = r[a][j] - 10
        elif a == 5 and S_t[j][1].item() < 4:
            # Bad drop off.
            # Stay in same state.
            P[a][j, j] = 1
            r[a][j] = r[a][j] - 10

        # PICK UP ACTIONS...
        elif a == 4:
            # Is it successful?
            # Make sure passenger is not already in the car.
            if S_t[j][1].item() == 4:
                # If so, nothing happens, will punish in contribution fx.
                P[a][j, j] = 1
            elif S_t[j][0].item() == locations[S_t[j][1].item()]:
                # Make sure taxi stays the same but passenger moves to
                # index 4, i.e. in the taxi.
                # Find that new state.
                for k in range(card_s):
                    if S_t[k][1].item() == 4:
                        if S_t[k][0].item() == S_t[j][0].item():
                            if S_t[k][2].item() == S_t[j][2].item():
                                P[a][j, k] = 1
                                r[a][j] = r[a][j] - 1
            else:
                # UNSUCESSFUL PICK UP
                P[a][j, j] = 1
                r[a][j] = r[a][j] - 10

        # TAXI MOVES!!!!!
        # UP MOVES!!!!! Update by -5
        elif a == 2:
            r[a][j] = r[a][j] - 1
            if S_t[j][0].item() > 4:
                for k in range(card_s):
                    if S_t[j][0].item() - 5 == S_t[k][0].item() and to.equal(S_t[j][1:], S_t[k][1:]):
                        P[a][j, k] = 1
            else:
                P[a][j, j] = 1

        # DOWN MOVES!!! Update by + 5
        elif a == 3:
            r[a][j] = r[a][j] - 1
            # Now, is this one possible?
            if S_t[j][0].item() < 20:
                for k in range(card_s):
                    if S_t[j][0].item() + 5 == S_t[k][0].item() and to.equal(S_t[j][1:], S_t[k][1:]):
                        P[a][j, k] = 1
            else:
                P[a][j, j] = 1

        # LEFT MOVES!!!! Update by -1
        elif a == 0:
            r[a][j] = r[a][j] - 1
            if S_t[j][0].item() not in [0, 5, 10, 15, 20]:
                # Calculate new_state = matches with taxi -1
                for k in range(card_s):
                    if S_t[j][0].item() - 1 == S_t[k][0].item() and to.equal(S_t[j][1:], S_t[k][1:]):
                        new_state = k
                # Check if this state has barrier restrictions.
                # If so,...
                if S_t[j][0].item() in [2, 16, 21, 18, 23]:
                    # Check if it is the specific action...
                    if a == barriers[S_t[j][0].item()]:
                        # If so, enforce...
                        P[a][j, j] = 1
                    else:
                        # If not, move to new_state...
                        P[a][j, new_state] = 1
                # If not, move to new_state...
                else:
                    P[a][j, new_state] = 1
            else:
                P[a][j, j] = 1

        # RIGHT MOVES!!!! Update by + 1
        elif a == 1:
            r[a][j] = r[a][j] - 1
            if S_t[j][0].item() not in [4, 9, 14, 19, 24]:
                # Calculate new_state = matches with taxi +1
                for k in range(card_s):
                    if S_t[j][0].item() + 1 == S_t[k][0].item() and to.equal(S_t[j][1:], S_t[k][1:]):
                        new_state = k
                # Check if this state has barrier restrictions.
                # If so,...
                if S_t[j][0].item() in [1, 7, 15, 20, 17, 22]:
                    # Check if it is the specific action...
                    if a == barriers[S_t[j][0].item()]:
                        # If so, enforce...
                        P[a][j, j] = 1
                    else:
                        # If not, move to new_state...
                        P[a][j, new_state] = 1
                # If not, move to new_state...
                else:
                    P[a][j, new_state] = 1
            else:
                P[a][j, j] = 1







"""for a in range(7):
    if a != 6:
        P[a][400, 400] = 0"""

P[6] = to.eye(card_s)
# Logic Checks.
for a in range(7):
    i = 0
    for row in P[a]:
        if sum(row) != 1:
            print('row {} of action {} does not sum to 1.'.format(i, a))

        i += 1

# Save the tensors...
to.save(P, 'data/transition_matrices.pt')
to.save(r, 'data/reward_vectors.pt')