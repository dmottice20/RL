import torch as to
import itertools
from tqdm import tqdm

# Build a dict storing the possible kinematic locations of the passenger.
L_possible = {
    0: to.tensor([0, 0]),
    1: to.tensor([3, 0]),
    2: to.tensor([0, 4]),
    3: to.tensor([4, 4])
}

# Randomly initialize the locations of the customer and their
# desired destination.
rand = to.randint(4, (1,)).item()
supply = L_possible.get(rand)
D_possible = L_possible
D_possible.pop(rand, None)
rand = to.randint(3, (1,)).item()
demand = D_possible.get(rand)
print('Starting customer location is...{}'.format(supply))
print('Desired location for customer is...{}'.format(demand))

# Construct state space.
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

# Construct action space.
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
P = to.zeros(size=(card_a, card_s, card_s))
# Loop over indices {0,..., 400}.
# S_t[i] = taxi_location, passenger_location, desired destination.
for i in tqdm(range(card_s), desc='building transition matrices'):
    for a in A:
        a = a.item()
        for j in range(card_s):
            # If NOT in the terminal state...
            if i != 400:
                # Look at transitions when moving to L, update by -1.
                if a == 0:
                    # Is action possible?
                    if S_t[i][0].item() not in [0, 5, 10, 15, 20]:
                        if S_t[i][0].item() - 1 == S_t[j][0].item() and to.equal(S_t[i][1:], S_t[j][1:]):
                            # Check if state has barrier restrictions part a...
                            if S_t[i][0].item() in barriers.keys():
                                # Check if state action pair violates barrier restrictions...
                                if a == barriers[S_t[i][0].item()]:
                                    # Enforce the null action.
                                    P[a][i, i] = 1
                                else:
                                    P[a][i, j] = 1
                            else:
                                P[a][i, j] = 1

                    # Check if there is no transition update currently there.
                    if j == 400:
                        if sum(P[a][i, :]) == 0:
                            P[a][i, i] = 1

                # Look at transition when moving to R, update by + 1.
                elif a == 1:
                    # Is action possible.
                    if S_t[i][0].item() not in [4, 9, 14, 19, 24]:
                        if S_t[i][0].item() + 1 == S_t[j][0].item() and to.equal(S_t[i][1:], S_t[j][1:]):
                            # Check if state has barrier restrictions part b...
                            if S_t[i][0].item() in barriers.keys():
                                # Check if state action pair violates barrier restrictions...
                                if a == barriers[S_t[i][0].item()]:
                                    # Enforce the null action.
                                    P[a][i, i] = 1
                                else:
                                    P[a][i, j] = 1
                            else:
                                P[a][i, j] = 1

                    # Check if there is no transition update currently there.
                    if j == 400:
                        if sum(P[a][i, :]) == 0:
                            P[a][i, i] = 1

                # Look at transition when moving to U, update by -5.
                elif a == 2:
                    # Is action possible???
                    if S_t[i][0].item() > 4:
                        if S_t[i][0].item() - 5 == S_t[j][0].item() and to.equal(S_t[i][1:], S_t[j][1:]):
                            # Check if state has barrier restrictions part c...
                            if S_t[i][0].item() in barriers.keys():
                                # Check if state action pair violates barrier restrictions...
                                if a == barriers[S_t[i][0].item()]:
                                    # Enforce the null action.
                                    P[a][i, i] = 1
                                else:
                                    P[a][i, j] = 1
                            else:
                                P[a][i, j] = 1

                    # Check if there is no transition update currently there.
                    if j == 400:
                        if sum(P[a][i, :]) == 0:
                            P[a][i, i] = 1

                # Look at transition when moving to D, update by +5.
                elif a == 3:
                    # Now, is this one possible?
                    if S_t[i][0].item() < 20:
                        if S_t[i][0].item() + 5 == S_t[j][0].item() and to.equal(S_t[i][1:], S_t[j][1:]):
                            # Check if state has barrier restrictions part d...
                            if S_t[i][0].item() in barriers.keys():
                                # Check if state action pair violates barrier restrictions...
                                if a == barriers[S_t[i][0].item()]:
                                    # Enforce the null action.
                                    P[a][i, i] = 1
                                else:
                                    P[a][i, j] = 1
                            else:
                                P[a][i, j] = 1

                    # Check if there is no transition update currently there.
                    if j == 400:
                        if sum(P[a][i, :]) == 0:
                            P[a][i, i] = 1

                # Look at transitions when picking up passenger.
                elif a == 4:
                    # Check if the passenger is already in car.
                    if S_t[i][1].item() == 4:
                        # If so, nothing happens, will punish in contribution fx.
                        P[a][i, i] = 1
                    else:
                        # Check if the taxi is in the same location as the customer.
                        if S_t[i][0].item() == locations[S_t[i][1].item()]:
                            if S_t[j][1].item() == 4 and S_t[i][2].item() == S_t[j][2].item() and S_t[i][0].item() == \
                                    S_t[j][0].item():
                                P[a][i, j] = 1

                    # Check if there is no transition update currently there.
                    if j == 400:
                        if sum(P[a][i, :]) == 0:
                            P[a][i, i] = 1

                # Look at transitions when dropping off passenger.
                elif a == 5:
                    # Ensure passenger is in car.
                    if S_t[i][1].item() == 4:
                        # Ensure at correct location
                        if S_t[i][0].item() == locations[S_t[i][2].item()]:
                            if j == 400:
                                P[a][i, j] = 1
                        # If not, nothing happens.
                        else:
                            P[a][i, i] = 1
                    # If not, nothing happens.
                    else:
                        P[a][i, i] = 1
            # IF IN THE TERMINAL STATE...
            else:
                # Each state can be visited with a probability
                # 1/4 * 1/3 where 1/4 comes from P(customer starting location)
                # and 1/3 comes from P(customer destination)
                # as long as i and j != 400 (i.e. terminal state)
                # and second element != 4
                # only action is do nothing.
                """if a == 6:
                    if S_t[j][1].item() < 4:
                        P[a][i, j] = 1 / len(locations.keys()) * 1 / (len(locations.keys()) - 1) * 1 / 25
                    if j != 400:
                        P[a][j, j] = 1"""


# Update that all actions except for the last have absorbing terminal states.
for a in A:
    a = a.item()
    if a != 6:
        P[a][400, 400] = 1

P[6] = to.eye(card_s)
# Logic Checks.
for a in A:
    a = a.item()
    i = 0
    for row in P[a]:
        if sum(row) != 1:
            print('row {} of action {} does not sum to 1.'.format(i, a))

        i += 1

# Save the tensors...
to.save(P, 'data/transition_matrices.pt')
