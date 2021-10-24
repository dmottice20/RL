import torch as to


locations = {
    0: 0,
    1: 4,
    2: 20,
    3: 23,
}

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

actions = {
    # Move left, i.e. -1
    0: -1,
    # Move right, i.e. +1
    1: 1,
    # Move up, i.e. -5
    2: -5,
    # Move down, i.e. +5
    3: 5
}


# Define the system model...
# s = (taxi_location, customer_location, desired_destination)
#       s[0].item()     s[1].item()       s[2].item()
# a = 0, 1, 2, 3, 4, 5, 6 for L, R, U, D, pickup, drop.
def system_model(s, a):
    # If doing nothing...
    if a == 6:
        stp1 = s.clone()
    # If dropping off...
    elif a == 5:
        # Ensure passenger is in the car.
        if s[1].item() == 4:
            # See if taxi is at passenger's destination...
            if s[0].item() == locations[s[2].item()]:
                # Move to the terminal state...
                stp1 = to.tensor([25, 5, 5])
            # If not, nothing happens...
            else:
                stp1 = s.clone()
        else:
            # If not, nothing will happen...
            stp1 = s.clone()
    # If picking up the passenger...
    elif a == 4:
        # Check if the passenger is already in the car...
        if s[1].item() == 4:
            stp1 = s.clone()
        # If not, good...
        else:
            # Check if the taxi is in same location as taxi...
            if s[0].item() == locations[s[1].item()]:
                # Keep taxi_location and destination same but update passenger location
                # to be 4.
                stp1 = to.tensor([s[0].item(), 4, s[2].item()])
            else:
                stp1 = s.clone()
    # If performing some kinematic actions...
    elif 0 <= a < 4:
        # Is the action possible?
        possible = False
        if a == 0:
            if s[0].item() not in [0, 5, 10, 15, 20]:
                possible = True
        elif a == 1:
            if s[0].item() not in [4, 9, 14, 19, 24]:
                possible = True
        elif a == 2:
            if s[0].item() > 4:
                possible = True
        elif a == 3:
            if s[0].item() < 20:
                possible = True

        if possible:
            # Is there a barrier restriction?
            if s[0].item() in barriers.keys():
                # If so, does this action violate it?
                if a == barriers[s[0].item()]:
                    # Enforce the null action...
                    stp1 = s.clone()
                else:
                    # If not, update the action...
                    stp1 = to.hstack((to.tensor(s[0].item() + actions[a]), s[1:]))
            else:
                stp1 = to.hstack((to.tensor(s[0].item() + actions[a]), s[1:]))
        else:
            stp1 = s.clone()
    else:
        raise ValueError('Action {} is not know in state {}'.format(a, s))

    return stp1


def contribution_function(s, a, stp1):
    # Rewards for doing nothing...
    if a == 6:
        if to.equal(stp1, to.tensor([25, 5, 5])):
            contribution = 0
        else:
            contribution = -1
    # Rewards for dropping off passengers...
    elif a == 5:
        # Was it successful?
        if to.equal(stp1, to.tensor([25, 5, 5])):
            contribution = 20
        else:
            contribution = -10
    # Rewards for picking-up passengers...
    elif a == 4:
        # Was the passenger on board already?
        if s[1].item() == 4:
            contribution = -10
        # Are we at the right location?
        elif stp1[1].item() != 4:
            contribution = -10
        # Otherwise, we are successful...
        else:
            contribution = -1
    elif 0 <= a < 4:
        # No matter what, every kinematic action
        # earns a contribution of -1...
        contribution = -1
    else:
        raise ValueError('S_t = {},\n a_t = {},\n S_t+1 = {}'.format(s, a, stp1))

    return contribution


def grab_idx(states_index, state):
    idx = next((k for k in states_index if to.equal(states_index[k], state.int())), None)
    if idx is None:
        raise ValueError('Value {} does not match a state in states_index.'.format(state))
    else:
        return idx

