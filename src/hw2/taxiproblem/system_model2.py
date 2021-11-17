import torch as to
import numpy as np

# Define the locations to cell on a gridworld mapping in a hash-table.
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

# Define the set of actions.
# Each key is the action index and the value is the kinematic effect on the gridworld.
actions = {
    0: -1,  # Go Left.
    1: +1,  # Go Right.
    2: -5,  # Go Up.
    3: +5  # Go Down.
}


# Fx. to grab the key of a dictionary given the value of a torch tensor.
def grab_key(dictionary, state):
    for key, value in dictionary.items():
        if to.equal(value, state.int()):
            return key


def system_model(state_idx, action, indices):
    """
    Fx. to govern the system transition.
    :param state_idx: the current state of the system.
    :param action: the action taken at time t.
    :return: next_state_idx
    """
    state = indices[state_idx]
    # If in the terminal state, stay.
    if state_idx == 400:
        next_state_idx = 400
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
        if action == barriers.get(state_idx):
            possible = False

        if possible:
            # If possible, then update taxi's position.
            next_state = to.tensor([
                state[0].item() + actions[action],
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
                next_state_idx = 400
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


def contribution_fx(state_idx, state_tp1_idx, action, indices):
    """
    Fx. to compute the contribution earned.
    :param state_idx:
    :param action:
    :param indices:
    :return:
    """
    state = indices[state_idx]
    state_tp1 = indices[state_tp1_idx]
    # If in the terminal state, the reward is 0.
    if state_idx == 400:
        contribution = 0
    else:
        # If the action is kinematic...
        if action < 4:
            contribution = -1
        # If the action is to pick-up...
        if action == 4:
            # Is there a passenger now on board as the result of the action.
            if state[1].item() != 4 and state_tp1[1].item() == 4:
                contribution = -1
            else:
                contribution = -10
        # If the action is to drop-off...
        if action == 5:
            # Are we now at the terminal state?
            if state_tp1_idx == 400:
                contribution = 20
            # Otherwise.
            else:
                contribution = -10

    return contribution


