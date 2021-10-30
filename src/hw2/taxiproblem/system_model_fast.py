import torch as to


# Use matrices instead of hash tables to store location and barrier information.
# Locations: col 1 - location # (variable), col 2 - location on the gridworld (cell #).
locations = to.tensor([
    [0, 0], [1, 4], [2, 20], [3, 23]
])
# barriers: col 1 - location on gridworld, col 2 - action not allowed there
barriers = to.tensor([
    [1, 1], [2, 0], [15, 1], [20, 1], [16, 0], [21, 0], [17, 1], [22, 1], [18, 0], [23, 0]
])
# actions: col 1 - index number for kinematic action, col 2 - impact on the gridworld.
actions = to.tensor([
    [0, -1], [1, 1], [2, -5], [3, 5]
])


# Create a fx. for system model and contribution fx.
def system_model(state_idx, action, state_space):
    """
    Fx. to govern the system transition.
    :param state_idx: the current state of the system.
    :param action: the action taken at time t.
    :param state_space: state space matrix.
    :return: next_state_idx
    """
    state = state_space[state_idx, :]
    # If in the terminal state, stay.
    if state_idx == 500:
        next_state_idx = 500
    # Otherwise.
    # I am performing some kinematic action...
    elif action < 4:
        # Is the action possible given the current state...
        possible = True
        if action == 0 and state[0].item() in [0, 5, 10, 15, 20]:
            possible = False
        elif action == 1 and state[0].item() in [4, 9, 14, 19, 24]:
            possible = False
        elif action == 2 and state[0].item() < 5:
            possible = False
        elif action == 3 and state[0].item() > 19:
            possible = False

        # Is a barrier restricting this action?
        if barriers[:, 1][barriers[:, 0] == state[0].item()].shape[0] == 0:
            possible = False

        if possible:
            # If possible, then update taxi's position.
            next_state = to.tensor([
                state[0].item() + actions[action, 1].item(),
                state[1].item(),
                state[2].item()
            ])
            next_state_idx = grab_key(indices, next_state)
        else:
            # Not possible, therefore, must stay.
            next_state_idx = state_idx

    # Performing a pick-up action...
    elif action == 4:
        # Check if a passenger is in the car already.
        if state[1].item() == 4:
            # If so, stay in the same spot.
            next_state_idx = state_idx
        else:
            # If not, check if in same spot as location.
            if state[0].item() == locations[state[1].item()]:
                # Update passenger location to be equal to 4 (onboard).
                next_state = to.tensor([
                    state[0].item(),
                    4,
                    state[2].item()
                ])
                # Return the idx.
                next_state_idx = grab_key(indices, next_state)
            else:
                # stay in the same spot.
                next_state_idx = state_idx

    # Performing a drop-off action...
    elif action == 5:
        # Ensure a passenger is in the car.
        if state[1].item() == 4:
            # Ensure at the correct location.
            if state[0].item() == locations[state[2].item()]:
                next_state_idx = 500
            else:
                # Otherwise, stay.
                next_state_idx = state_idx
        else:
            # Otherwise, stay.
            next_state_idx = state_idx

    else:
        raise ValueError('Transition Fx. Error!')

    # Return the next state of the system.
    return next_state_idx
