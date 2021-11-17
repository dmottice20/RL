import torch as to
import numpy as np
import random


def approximate_bellman_equation(state_t, value, gamma, num_samples):
    v_hat = to.zeros(size=(num_samples, 4))
    states = to.zeros(size=(num_samples, 4), dtype=to.int)
    for a in np.arange(4):
        for i in np.arange(num_samples):
            # Obtain next state given current state, action & sampled outcome.
            states[i, a] = system_model(state_t, a)
            # Compute the value of that sampled state, action, outcome tuple.
            v_hat[i, a] = contribution_fx(state_t, a, states[i, a].item()) + gamma * value[states[i, a].item()]

    v_hat_t = to.max(to.mean(v_hat, 0)).item()
    a_star = to.argmax(to.mean(v_hat, 0)).item()
    sample = random.randint(0, num_samples - 1)

    return {
        'value estimate': v_hat_t,
        'next state': states[sample, a_star].item(),
        'reward': contribution_fx(state_t, a_star, states[sample, a_star].item())
    }


def system_model(state, action):
    if state == 38:
        next_state = 38
    elif state == 37:
        next_state = 37
    else:
        if action == 0:
            if state > 11:
                next_state = state - 12
            else:
                next_state = state
        elif action == 1:
            if state not in [11, 23, 35]:
                next_state = state + 1
            else:
                next_state = state
        elif action == 2:
            if state == 36:
                next_state = state
            elif state == 35:
                next_state = 38
            elif 25 <= state <= 34:
                next_state = 37
            else:
                next_state = state + 12
        elif action == 3:
            if state not in [0, 12, 24, 36]:
                next_state = state - 1
            else:
                next_state = state
        else:
            raise ValueError('Transition Fx. Error!')

    # Apply the possible slipper slope.
    if random.random() < 0.35:
        pds = next_state
        if pds == 37 or pds == 38:
            next_state = pds
        elif pds == 36:
            next_state = pds
        elif pds == 35:
            next_state = 38
        elif 25 <= pds <= 34:
            next_state = 37
        else:
            next_state = pds + 12

    return next_state


def contribution_fx(state_t, action, outcome):
    contribution = -1
    if outcome == 37:
        contribution = -100
    elif state_t == 38:
        contribution = 0

    return contribution
