import torch as to
import numpy as np
import pandas as pd
import random



# Construct the state space and action space.
M = 3650
state_space = to.arange(M)
action_space = to.arange(M + 1)


def system_model(state, action, prob):
    """
    Fx. to take the model transition function into code.
    :param state: state at time t
    :param action:  action taken at time t
    :param prob: probability of survival vector (1 x card_s)
    :return: next_state: state at time t+1
    """
    # If action is to do nothing...
    if action == 0:
        # If my state is M (M-1 in python), then I will stay in M for sure.
        if state == M - 1:
            next_state = state
        # If my state is M-1 (M-2 in python), then I will go to M for sure.
        elif state == M - 2:
            next_state = M - 1
        # Otherwise...
        else:
            # Probability(state) of going to next month.
            if random.random() < prob[state + 1]:
                next_state = state + 1
            # Otherwise, go to end.
            else:
                next_state = M - 1
    # If purchasing an M-1 old car (or M in python), go to M - 1.
    elif action == M:
        next_state = M - 1
    # Given: where a > 1, a means you buy a car of age (a - 2), could be 0.
    # In my code: where action > 0, action means you buy a car of age (action - 1), could be 0.
    else:
        # There is still some probability of failure from action - 1 to action in the epoch.
        if random.random() < prob[action - 1]:
            next_state = action
        else:
            next_state = M - 1
    return next_state


def contribution_fx(state, action, op_cost, value, cost):
    """
    Fx. to calculate contribution earned in epoch given state, action pair.
    :param state: state at time t
    :param action: action taken at time t
    :return: contribution: contribution earned
    """
    # If my action is to do nothing...
    if action == 0:
        # Only incur the operating cost.
        contribution = -op_cost[state - 1]
    # Otherwise...
    else:
        # Earn the trade-in value then subtract the cost of buying car of age (action-1) and its operating cost.
        contribution = value[state - 1] - cost[action - 1] - op_cost[action - 1]
    return contribution[0]


def approximate_bellman_equation(state_t, value, gamma, num_samples, prob, op_cost, val, cost):
    v_hat = to.zeros(size=(num_samples, 4))
    states = to.zeros(size=(num_samples, 4), dtype=to.int)
    for a in np.arange(4):
        for i in np.arange(num_samples):
            # Obtain next state given current state, action & sampled outcome.
            states[i, a] = system_model(state_t, a, prob)
            # Compute the value of that sampled state, action, outcome tuple.
            v_hat[i, a] = contribution_fx(state_t, a, op_cost, val, cost) + gamma * value[states[i, a].item()].item()

    v_hat_t = to.max(to.mean(v_hat, 0)).item()
    a_star = to.argmax(to.mean(v_hat, 0)).item()
    sample = random.randint(0, num_samples - 1)

    return {
        'value estimate': v_hat_t,
        'next state': states[sample, a_star].item(),
        'reward': contribution_fx(state_t, a_star, op_cost, val, cost)
    }
