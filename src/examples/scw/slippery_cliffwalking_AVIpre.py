import torch as to
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, math
import numpy as np
from avipre_support import approximate_bellman_equation


# STATE SPACE...
#  0  1  2  3  4  5  6  7  8  9 10 11
# 12 13 14 15 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31 32 33 34 35
# 36 -------- cliff = 37 --------- 38

#####################################################
# Approximate Value Iteration (Pre-decision State)  #
#####################################################

# Initialize algorithm parameters and variables.
def alpha(): return 0.5  # learning rate
def epsilon(): return 0.1  # exploration vs. exploring rate


gamma, num_samples, num_reps, num_episodes = 1, 2, 100, 500
random.seed(1)

# Initialize problem parameters and variables.
initial_state = 36
terminal_states = [37, 38]

g_per_episode = to.zeros(size=(num_reps, num_episodes))
rmse_per_episode = to.zeros(size=(num_reps, num_episodes))

# Load the optimal value fx.
v_star = to.load('Vstar_SCW.pt')

# Loop for num_reps replications.
for rep in tqdm(np.arange(num_reps), desc='AVI (Pre) for {} replications'.format(num_reps)):
    # Value function approximation (VFA) initialization.
    v_bar = to.zeros(39)
    # Loop for num_episodes episodes.
    for m in np.arange(num_episodes):
        # Initialize target stat.
        g = 0
        # Select the initial state.
        state_t = initial_state
        # Simulate while not in the terminal state
        while state_t not in terminal_states:
            # Solve approximate Bellman update using current VFA.
            solution = approximate_bellman_equation(state_t, v_bar, gamma, num_samples)
            # Estimated value of the current state.
            v_hat_t = solution['value estimate']
            # Next system state.
            state_tp1 = solution['next state']
            # Update VFA.
            v_bar[state_t] = (1-alpha()) * v_bar[state_t] + alpha() * v_hat_t
            # Update return.
            g += solution['reward']
            # Update state.
            state_t = state_tp1

        # Record the return earned...
        g_per_episode[rep, m] = g
        # Compute root mean square error...
        a = to.reshape(v_bar[:36], (3, 12))
        b = to.hstack((v_bar[37], to.zeros(11)))
        v_adp = to.vstack((a, b.reshape(1, 12)))
        rmse_per_episode[rep, m] = math.sqrt(to.mean((v_adp - v_star) ** 2).item())

# Average returns across all runs for each episode.
g_avi_pre = to.mean(g_per_episode, 0)
rmse_avi_pre = to.mean(rmse_per_episode, 0)

# Save the returns.

plt.plot(np.arange(num_episodes), rmse_avi_pre)
plt.xlabel('Episode')
plt.ylabel('Root mean Square Error')
plt.title('Performance Over {} Replications'.format(num_reps))
plt.grid()
plt.show()

plt.plot(np.arange(num_episodes), g_avi_pre)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.grid()
plt.show()
