import torch as to
import numpy as np
import random


def system_model_w(post_decision_state):
    # Apply the possible slippery slope -- where the agent would move south.
    if random.random() < 0.35:
        if post_decision_state == 37 or post_decision_state == 38:
            pre_decision_state = post_decision_state
        elif post_decision_state == 36:
            pre_decision_state = post_decision_state
        elif post_decision_state == 35:
            pre_decision_state = 38
        elif 25 <= post_decision_state <= 34:
            pre_decision_state = 37
        else:
            pre_decision_state = post_decision_state + 12
    else:
        # No slip happens.
        pre_decision_state = post_decision_state
    return pre_decision_state


def approximate_bellman_equation(state_t_pre, value, gamma):
    value_hat = to.zeros(size=(1, 4))
    states = to.zeros(size=(1, 4), dtype=to.int)
    for a in np.arange(4):
        states[:, a] = system_model_action(state_t_pre, a)
        # Compute the value.
        value_hat[:, a] = contribution_fx(state_t_pre, a) + gamma * value[states[:, a].item()]

    v_hat, a_star = to.max(value_hat).item(), to.argmax(value_hat).item()

    return {
        'value estimate': v_hat,
        'post-decision state': states[:, a_star].item(),
        'reward': contribution_fx(state_t_pre, a_star)
    }


def system_model_action(state_t_pre, action):
    # Stay in terminal states.
    if state_t_pre == 38:
        state_t_post = 38
    elif state_t_pre == 37:
        state_t_post = 37
    else:
        if action == 0:
            if state_t_pre > 11:
                state_t_post = state_t_pre - 12
            else:
                state_t_post = state_t_pre
        elif action == 1:
            if state_t_pre not in [11, 23, 35]:
                state_t_post = state_t_pre + 1
            else:
                state_t_post = state_t_pre
        elif action == 2:
            if state_t_pre == 36:
                state_t_post = state_t_pre
            elif state_t_pre == 35:
                state_t_post = 38
            elif 25 <= state_t_pre <= 34:
                state_t_post = 37
            else:
                state_t_post = state_t_pre + 12
        elif action == 3:
            if state_t_pre not in [0, 12, 24, 36]:
                state_t_post = state_t_pre - 1
            else:
                state_t_post = state_t_pre
        else:
            raise ValueError('Transition Fx. Error')

    return state_t_post


def contribution_fx(state_t_pre, action):
    state_t_post = system_model_action(state_t_pre, action)
    if state_t_post == 35:
        # If next to the goal, 0.65*(-1) + 0.35*0
        contribution = -0.65
    elif state_t_post == 37:
        # If in the cliff....
        contribution = -100
    elif 25 <= state_t_post <= 34:
        # Next to cliff...
        # 0.65*(-1) + (0.35)*(-100)
        contribution = -35.65
    else:
        contribution = -1

    return contribution
