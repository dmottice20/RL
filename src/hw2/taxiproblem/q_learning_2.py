import torch as to
from tqdm import tqdm
import timeit
import random, itertools
import numpy as np
import matplotlib.pyplot as plt
from system_model2 import system_model, contribution_fx, grab_key

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
state_space = to.zeros(size=(401, 3), dtype=to.int)
j = 0
for i in range(card_s):
    if S_ta[i, 1] == S_ta[i, 2]:
        continue
    else:
        state_space[j, :] = S_ta[i, :]
        j += 1

state_space[-1] = to.tensor([25, 5, 5])
card_s = state_space.shape[0]

# Hash table to store the index to state mapping.
indices = dict(zip(np.arange(card_s), state_space))

starting_states = []
for i in range(card_s):
    if state_space[i][1].item() < 4 and state_space[i][1].item() != state_space[i][2].item():
        starting_states.append(i)

beta1 = 0.3
beta2 = 0.3
a_alpha = 25
a_epsilon = 50
constant_alpha = 0.8
constant_epsilon = 0.25


# Initialize algorithm parameters and variables...
def alpha(state_action_pair):
    """
    Generalized harmonic
    :param state_action_pair:
    :return:
    """
    if state_action_pair == 0:
        return 0.8
    else:
        return a_alpha / (a_alpha + state_action_pair - 1)


def epsilon(state_count):
    if state_count == 0:
        return 0.15
    else:
        return a_epsilon / (a_epsilon + state_count - 1)


# Initialize key algorithm parameters...
gamma, num_reps, num_episodes = 0.95, 10, 1000
random.seed(1)
terminal_state = 400

# Storage tensors for graphs on algorithm performance.
g_per_episode = to.empty(size=(num_reps, num_episodes))
rmse_per_episode = to.empty(size=(num_reps, num_episodes))

# Load the v star.
# v_star = to.load('src/hw2/taxiproblem/v_star_taxi.pt')
v_star = to.load('v_star_taxi.pt')
v_star = to.tensor(v_star)

start_time = timeit.default_timer()
# Perform num_reps replications of the algorithm to adequately account for any randomness.
for rep in tqdm(np.arange(num_reps), desc='Q-LEARNING for {} replications'.format(num_reps)):
    # for rep in np.arange(num_reps):
    # Initialize Q-factors optimistically.
    q_bar = to.zeros(size=(card_s, 6))
    # q_bar = to.ones(size=(card_s, 6))*20
    # Train for num_episodes episodes...
    state_action_counters = to.zeros(size=(card_s, 6))
    state_counters = to.zeros(card_s)
    for m in np.arange(num_episodes):
        # print('\n\n======= EPISODE {} =======\n'.format(m))
        # Select initial state randomly.
        state_t = random.choice(starting_states)
        state_counters[state_t] += 1
        # Initialize target reward statistic.
        g = 0
        # Simulate while not in the terminal state...
        while state_t != terminal_state:
            exploiting_action = to.argmax(q_bar[state_t, :]).item()
            if random.random() > constant_epsilon:
            # if random.random() > epsilon(state_counters[state_t]):
                # Exploit, i.e. choose the action using current q-bars.
                action_t = exploiting_action
            else:
                # Explore...
                actions = list(np.arange(6))
                actions.remove(exploiting_action)
                action_t = random.choice(actions)

            # Update state, action pair count.
            state_action_counters[state_t, action_t] += 1
            # Compute the next state...
            state_t_plus_1 = system_model(state_t, action_t, indices)
            # print('\nIn state {}...'.format(indices[state_t]))
            # print('Action {} is taken...'.format(action_t))
            # print('Which transitions to {}'.format(indices[state_t_plus_1]))
            # Compute contribution...
            contribution_t = contribution_fx(state_t, state_t_plus_1, action_t, indices)
            # print('This earned {}'.format(contribution_t))
            g += contribution_t
            # Compute q_hat_t
            q_hat_t = contribution_t + gamma * to.max(q_bar[state_t_plus_1, :]).item()
            # Update Q-bar...
            """q_bar[state_t, action_t] = (1 - alpha(state_action_counters[state_t, action_t])) * \
                                       q_bar[state_t, action_t].item() + alpha(state_action_counters[state_t, action_t]) \
                                       * q_hat_t"""
            q_bar[state_t, action_t] = (1-constant_alpha) * q_bar[state_t, action_t].item() + constant_alpha * q_hat_t
            # Update state and action...
            state_t = state_t_plus_1
            # Update state counters.
            state_counters[state_t] += 1

        # Record return earned in episode m of repetition.
        g_per_episode[rep, m] = g
        v_adp_mod, _ = to.max(q_bar, 1)
        v_adp = v_adp_mod.reshape(card_s, 1)
        # Compute the root mean squared error.
        #rmse_rep_m[rep, m] = math.sqrt(to.mean((v_adp_full - v_star) ** 2).item())
        rmse_per_episode[rep, m] = np.sqrt(to.mean((v_adp_mod - v_star) ** 2).item())

# Average return across all runs for each episode.
g_q_learning = to.mean(g_per_episode, 0)
rmse_q_learning = to.mean(rmse_per_episode, 0)

end_time = timeit.default_timer()

print('Run Time is: {}'.format(end_time - start_time))

# Plot performance...
plt.plot(range(num_episodes), rmse_q_learning)
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance over {} Replications'.format(num_reps, beta1, beta2))
plt.grid()
plt.show()

plt.plot(range(num_episodes), g_q_learning)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.grid()
plt.show()
