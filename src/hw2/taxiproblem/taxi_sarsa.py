import torch as to
from tqdm import tqdm
import random
import itertools
import numpy as np
import math
from src.hw2.taxiproblem.supporting_functions import system_model, contribution_function, grab_idx


#############################################
# Solve the Taxi Problem using SARSA...     #
#############################################
# Define the various learning rate and exploration rate
# functions...
def constant_alpha(m): return 0.5


def constant_epsilon(m): return 0.1


# Construct state space...
a = to.as_tensor(list(itertools.product(to.arange(25), to.arange(5), to.arange(4))))
a = to.vstack((a, to.as_tensor([25, 5, 5])))
card_s = a.shape[0]
S_t = to.zeros(size=(401, 3), dtype=to.int)
j = 0
for i in range(card_s):
    if a[i, 1] == a[i, 2]:
        continue
    else:
        S_t[j, :] = a[i, :]
        j += 1

S_t[-1] = to.tensor([25, 5, 5])
card_s = S_t.shape[0]
states_index = dict(zip(range(card_s), S_t))
card_a = 7


# Define a function for the randomized policy to be used
# during SARSA...
def vfa_randomized_policy(s, q_table, m):
    s_idx = grab_idx(states_index, s)
    # If in the terminal state...
    if s_idx == 400:
        # Can only do nothing...
        a_star = 6
    else:
        # Can choose the max action from 0,1,..,5
        exploit_action = to.argmax(q_table[s_idx, :6]).item()
        if random.random() > constant_epsilon(m):
            a_star = exploit_action
        else:
            # Choose randomly from 0,1,...,5
            actions = [a for a in range(6)]
            actions.remove(exploit_action)
            a_star = random.choice(actions)
    return a_star


# STEP -1:
# (a) Initialize algorithm parameters...
gamma = 0.95
random.seed(1)
num_reps, num_episodes = 100, 500
# (b) Initialize problem parameters...
# Random starting position across possible starting states...
# Grab the possible initial starting states.
starting_states = []
for i in range(401):
    if S_t[i][1].item() < 4:
        starting_states.append(i)
initial_state_idx = random.choice(starting_states)
initial_state = states_index[initial_state_idx]
terminal_state = to.tensor([25, 5, 5])
terminal_state_idx = grab_idx(states_index, terminal_state.int())

g_rep_m = to.zeros(size=(num_reps, num_episodes))
rmse_rep_m = to.zeros(size=(num_reps, num_episodes))

# Load the v_star...
v_star = to.load('../data/v_star_taxi.pt')

# Perform num_reps replications...
for rep in tqdm(np.arange(num_reps), desc='performing {} replications'.format(num_reps)):
    # Initialize the Q-factors optimistically...
    q_bar = to.zeros(size=(card_s, card_a))
    # Go forward num_episodes episodes...
    for m in np.arange(num_episodes):
        # Select an initial state randomly...
        state_t_idx = random.choice(starting_states)
        state_t = states_index[initial_state_idx]
        # Select action based on randomized policy derived from Q-factors...
        action_t = vfa_randomized_policy(state_t, q_bar, m)
        # Initialize the cumulative reward statistic...
        g_sarsa = 0
        # While not in the terminal state...
        while not to.equal(state_t, terminal_state.long()):
            # Compute the next state...
            state_t_plus_1 = system_model(state_t, action_t)
            # Compute and update reward...
            contribution_t = contribution_function(state_t, action_t, state_t_plus_1)
            g_sarsa += contribution_t
            # Compute the next action...
            action_t_plus_1 = vfa_randomized_policy(state_t_plus_1, q_bar, m)
            # Compute q_hat_t...
            state_t_plus_1_idx = grab_idx(states_index, state_t_plus_1)
            q_hat_t = contribution_t + gamma * q_bar[state_t_plus_1_idx, action_t_plus_1].item()
            # Compute updated Q-value...
            q_bar[state_t_idx, action_t] = (1-constant_alpha(m)) * q_bar[state_t_idx, action_t].item() + \
                                           constant_alpha(m) * q_hat_t
            # Update state, state index, and action...
            state_t, state_t_idx, action_t = state_t_plus_1, state_t_plus_1_idx, action_t_plus_1

        # Record the return earned in episode m of repetition z...
        g_rep_m[rep, m] = g_sarsa
        # Compute the root mean square error...
        v_adp, _ = to.max(q_bar, 1)
        rmse_rep_m[rep, m] = math.sqrt(to.mean((v_adp - v_star) ** 2).item())
