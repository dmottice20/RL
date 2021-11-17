import torch as to
from tqdm import tqdm
import random, itertools
import numpy as np
import matplotlib.pyplot as plt
from system_model import system_model, contribution_fx, grab_key


# Step 0: Preprocessing Work.
def alpha(m): return 0.9
def epsilon(m): return 0.45
# beta_alpha = 0.95
# beta_epsilon = 0.50


# def alpha(state_action_pair):
#     if state_action_pair == 0:
#         return 0.8
#     else:
#         return 1/(state_action_pair ** beta_alpha)
#
#
# def epsilon(state_count):
#     if state_count == 0:
#         return 0.8
#     else:
#         return 1/(state_count ** beta_epsilon)


# Define a vfa randomized policy.
def vfa_randomized_policy(s, q_table, m):
    exploit_action = to.argmax(q_table[s, :]).item()
    if random.random() > epsilon(m):
        a_star = exploit_action
    else:
        # Explore...one of the other actions (not exploiting).
        actions = list(np.arange(6))
        actions.remove(exploit_action)
        a_star = random.choice(actions)
    return a_star


# Create the state space.
state_space = to.as_tensor(list(itertools.product(to.arange(25), to.arange(5), to.arange(4))))
state_space = to.vstack((state_space, to.as_tensor([25, 5, 5])))
card_s = state_space.shape[0]

# Hash table to store the index to state mapping.
indices = dict(zip(np.arange(card_s), state_space))

# Also, define the allowable states.
allowable_states = []
for i in range(501):
    if state_space[i][1].item() != state_space[i][2].item():
        allowable_states.append(i)
allowable_states.append(500)
# Define the set of possible starting states.
starting_states = []
for i in range(501):
    if state_space[i][1].item() < 4 and state_space[i][1].item() != state_space[i][2].item():
        starting_states.append(i)

assert 300 == len(starting_states), "{} Does not equal the expected length.".format(len(starting_states))

gamma, num_reps, num_episodes = 0.95, 10, 1000
random.seed(1)
terminal_state = 500

# Storage tensors for graphs on algorithm performance.
g_per_episode = to.empty(size=(num_reps, num_episodes))
rmse_per_episode = to.empty(size=(num_reps, num_episodes))
steps_per_episode = to.empty(size=(num_reps, num_episodes))

# Load the optimal value function.
# v_star = to.load('src/hw2/v_star_taxi.pt')
v_star = to.load('v_star_taxi.pt')
v_star = to.tensor(v_star)

# Perform num_reps replications...
for rep in tqdm(range(num_reps), desc='SARSA - {} replications'.format(num_reps)):
    # Initialize Q-factors optimistically?
    q_bar = to.zeros(size=(card_s, 6))
    # Go forward and train for num_episodes episodes...
    state_action_counters = to.zeros(size=(card_s, 6))
    state_counters = to.zeros(card_s)
    for m in range(num_episodes):
        # Select an initial distributions from the list of
        # starting states with an equal probability.
        state_t = random.choice(starting_states)
        state_counters[state_t] += 1
        # Select action based on state, action pair using...
        # action_t = vfa_randomized_policy(state_t, q_bar, m)
        action_t = vfa_randomized_policy(state_t, q_bar, state_counters[state_t].item())
        # Initialize the reward target to 0 for the episode...
        g = 0
        steps = 0
        # Simulate while not in the terminal state...
        while state_t != terminal_state:
            # Compute the next state...
            state_t_plus_1 = system_model(state_t, action_t, indices)
            # Compute the resulting contribution and update reward...
            contribution_t = contribution_fx(state_t, state_t_plus_1, action_t, indices)
            g += contribution_t
            # Compute next action...
            # action_t_plus_1 = vfa_randomized_policy(state_t_plus_1, q_bar, m)
            action_t_plus_1 = vfa_randomized_policy(state_t, q_bar, state_counters[state_t].item())
            # Compute q_hat_t...
            q_hat_t = contribution_t + gamma * to.max(q_bar[state_t_plus_1, :]).item()
            # Updated the Q-bar values...
            # q_bar[state_t, action_t] = (1 - alpha(m)) * q_bar[state_t, action_t].item() + alpha(m) * q_hat_t
            q_bar[state_t, action_t] = (1 - alpha(state_action_counters[state_t, action_t].item())) * \
                                       q_bar[state_t, action_t].item() + alpha(
                state_action_counters[state_t, action_t].item()) * \
                                       q_hat_t
            # Update our state and action...
            state_t, action_t = state_t_plus_1, action_t_plus_1
            state_counters[state_t] += 1
            state_action_counters[state_t, action_t] += 1
            steps += 1

        # Record the return earned in episode m of repetition z...
        g_per_episode[rep, m] = g
        steps_per_episode[rep, m] = steps
        # Compute root mean squared error...
        v_adp_mod, _ = to.max(q_bar, 1)
        v_adp = v_adp_mod[allowable_states].reshape(401, 1)
        rmse_per_episode[rep, m] = np.sqrt(to.mean((v_adp - v_star) ** 2).item())

# Average returns across all episode...
g_sarsa = to.mean(g_per_episode, 0)
# Average RMSE across all episode...
rmse_sarsa = to.mean(rmse_per_episode, 0)
steps_sarsa = to.mean(steps_per_episode, 0)

# Plot performance.
plt.plot(range(num_episodes), rmse_sarsa)
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance over {} Replications'.format(num_reps))
plt.show()

plt.plot(range(num_episodes), g_sarsa)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.show()

expected_steps = to.ones(num_episodes)*12.53
plt.plot(range(num_episodes), steps_sarsa)
plt.plot(range(num_episodes), expected_steps, color='red')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.grid()
plt.savefig('sarsa-steps.png')
plt.show()
