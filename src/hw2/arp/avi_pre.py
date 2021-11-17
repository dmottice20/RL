import torch as to
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from system_model import approximate_bellman_equation
import matplotlib.pyplot as plt


# Approximate Value Iteration using Pre-Decision State.
# Initialize algorithm parameters and variables.
gamma, num_samples, num_reps, num_episodes, num_steps = 0.9, 2, 10, 1000, 100
random.seed(1)
def alpha(m): return 0.9
# def epsilon(m): return 0.1
# a_alpha = 20
# alpha_beta = 0.31
#
#
# def alpha(count):
#     if count == 0:
#         return 0.9
#     else:
#         return 1/(count**alpha_beta)


# Initialize problem parameters and variables.
problem_size = 3650
states = list(np.arange(3650))
# Read in the data and separate into appropriate numpy arrays.
data = pd.read_csv('ARP_3650_data.csv').values
# data = pd.read_csv('src/hw2/arp/ARP_3650_data.csv').values
cost, value, op_cost, prob = np.hsplit(data, 4)

# Storage tensors, reward and rmse.
g_per_episode = to.zeros(size=(num_reps, num_episodes))
rmse_per_episode = to.zeros(size=(num_reps, num_episodes))

# Load optimal value function.
results = pd.read_csv('hw1problem8results.csv')
v_star = results['V_optimal'].values

# Loop for num_reps replications...
for rep in tqdm(range(num_reps), desc='AVI Pre - {} replications'.format(num_reps)):
    # Optimistic value function initialization...
    v_bar = to.zeros(problem_size)
    state_action_counter = to.zeros(size=(problem_size, problem_size+1))
    # Loop for num_episodes episodes....
    for m in range(num_episodes):
        # Initialize target statistic.
        g = 0
        # Select the initial state randomly.
        state_t = random.choice(states)
        # Simulate for some n-steps.
        for _ in range(num_steps):
            # Solve approximate bellman update...
            solution, action_t = approximate_bellman_equation(state_t, v_bar, gamma, num_samples, prob, op_cost, value, cost)
            # Grab the estimated value of current state and next system state...
            v_hat_t, state_t_plus_1 = solution['value estimate'], solution['next state']
            # Update our VFA and state-action counter.
            count = state_action_counter[state_t, action_t].item()
            v_bar[state_t] = (1-alpha(count)) * v_bar[state_t] + alpha(count) * v_hat_t
            state_action_counter[state_t, action_t] += 1
            # Update the return and state...
            g += solution['reward']
            state_t = state_t_plus_1

        # Record the return earned in the n-steps forward.
        g_per_episode[rep, m] = g
        # Same for root mean square error.
        rmse_per_episode[rep, m] = np.sqrt(to.mean((v_bar - v_star) ** 2).item())


# Average returns and root mean square error across all runs for each episode...
g_avi_pre = to.mean(g_per_episode, 0)
rmse_avi_pre = to.mean(rmse_per_episode, 0)

# Plot the return...
plt.plot(np.arange(num_episodes), rmse_avi_pre)
plt.xlabel('Episode')
plt.ylabel('Root mean Square Error')
plt.title('Performance Over {} Replications'.format(num_reps))
plt.grid()
plt.show()

"""plt.plot(np.arange(num_episodes), g_avi_pre)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Performance over {} Replications'.format(num_rcoseps))
plt.grid()
plt.show()"""

plt.plot(np.arange(3650), v_bar, label='AVI (Pre) Estimate')
plt.plot(np.arange(3650), v_star, label='Optimal Value Fx.')
plt.xlabel('Age of Car (i.e. state in S)')
plt.ylabel('Value of age of Car, V(S)')
plt.legend()
plt.grid()
plt.show()
