import torch as to
import numpy as np
from scipy.sparse import eye
import tqdm
import random

gamma = 1


# Define the approximate bellman equation...
def approximate_bellman_equation(st, V, gamma, num_samples):
    VHAT = to.zeros(num_samples, 4)
    STP1 = to.zeros(num_samples, 4)
    for a in range(4):
        for i in range(num_samples):
            # obtain next state given current state, action & sampled outcome.
            STP1[i, a] = system_model(st, a)
            # Compute the value...
            VHAT[i, a] = contribution_function(st, a, int(STP1[i, a].item())) + gamma * V[0, int(STP1[i, a].item())]

    vhat = to.max(to.mean(VHAT, 0)).item()
    astar = to.argmax(to.mean(VHAT, 0)).item()
    sample = to.randint(num_samples, size=(1,)).item()

    return vhat, int(STP1[sample, astar].item()), contribution_function(st, astar, int(STP1[sample, astar].item()))

    return {
        'value estimate': to.max(to.mean(VHAT, 0)).item(),
        ''
    }


def system_model(s, a):
    if s == 38:
        stp1 = 38
    elif s == 37:
        stp1 = 37
    else:
        # When going north...
        if a == 0:
            if s > 11:
                stp1 = s - 12
            else:
                stp1 = s
        # When going east...
        elif a == 1:
            if s not in [11, 23, 35]:
                stp1 = s + 1
            else:
                stp1 = s
        # When going south...
        elif a == 2:
            if s == 36:
                stp1 = s
            elif s == 35:
                stp1 = 38
            elif s >= 25 and s <= 34:
                stp1 = 37
            else:
                stp1 = s + 12
        # When going west...
        elif a == 3:
            if s not in [0, 12, 24, 36]:
                stp1 = s - 1
            else:
                stp1 = s
        else:
            raise Exception('Transition Function Error!')

    # Apply possible, slippery slope where agent could move south.
    if random.random() < 0.35:
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

    return stp1


def contribution_function(st, a, stp1):
    # the normal contribution...
    c_n = -1
    if stp1 == 37:
        c_n = -100
    elif st == 38:
        c_n = 0

    return c_n


def exploit_action(state, q_table):
    return to.argmax(q_table[state, :]).item()
