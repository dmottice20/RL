import torch as to
import numpy as np
import random, math
import matplotlib.pyplot as plt
from avipost_support import system_model_w, approximate_bellman_equation
from tqdm import tqdm


#####################################################
# Approximate Value Iteration (Post-decision State) #
#####################################################

# Initialize algorithm parameters and variables.
def alpha(): return 0.4
def epsilon(): return 0.1


gamma, num_reps, num_episodes = 1, 100, 500
random.seed(1)

# Initialize problem parameters and variables.
initial_state = 36
terminal_states = [37, 38]

# Record rewards during each episode to measure online performance.
# and root mean square error.
g_per_episode = to.empty(size=(num_reps, num_episodes))
rmse_per_episode = to.empty(size=(num_reps, num_episodes))

# Load the optimal value function (for RMSE computation).
v_star = to.load('Vstar_SCW.pt')

# Loop for num_reps replications.
for rep in tqdm(np.arange(num_reps), desc='perform {} replications'.format(num_reps)):
    # Initialize value function.
    v_bar = to.zeros(39)
    # Loop for num_episodes episodes.
    for m in np.arange(num_episodes):
        # Initialize non-discounted cumulative reward stat (target).
        g = 0
        # Select initial post-decision state randomly.
        if random.random() < epsilon():
            state_t_post = initial_state
        else:
            state_t_post = random.randint(0, 36)
        # Simulate transition to next pre-decision state.
        state_t_pre = system_model_w(state_t_post)
        # Simulate while not in the terminal state.
        while state_t_pre not in terminal_states:
            # Solve approximate bellman update using current Vbar.
            # Returns a hash with value estimate, post-decision state, and reward.
            solution = approximate_bellman_equation(state_t_pre, v_bar, gamma)
            # What is the estimated value of the current state.
            v_hat_t = solution['value estimate']
            # Update our Value Function Approximation.
            v_bar[state_t_post] = (1-alpha())*v_bar[state_t_post].item() + alpha() * v_hat_t
            # Record position-decision state.
            state_tp1_post = solution['post-decision state']
            # Simulate transition to next pre-decision state.
            state_tp1_pre = system_model_w(state_tp1_post)
            # Update the return.
            g += solution['reward']
            # Update the state variables
            state_t_post, state_t_pre = state_tp1_post, state_tp1_pre

        # Record the return earned...
        g_per_episode[rep, m] = g
        # Compute root mean square error...
        a = to.reshape(v_bar[:36], (3, 12))
        b = to.hstack((v_bar[37], to.zeros(11)))
        v_adp = to.vstack((a, b.reshape(1, 12)))
        rmse_per_episode[rep, m] = math.sqrt(to.mean((v_adp - v_star) ** 2).item())


# Average returns across all runs for each episode.
g_avi_post = to.mean(g_per_episode, 0)
rmse_avi_post = to.mean(rmse_per_episode, 0)

# Save the returns.
# Load the pre returns.
g_avi_pre = to.load('g_avi_pre.pt')
rmse_avi_pre = to.load('rmse_avi_pre.pt')


plt.plot(np.arange(num_episodes), rmse_avi_post, label='AVI (Post)')
plt.plot(np.arange(num_episodes), rmse_avi_pre, label='AVI (Pre)')
plt.xlabel('Episode')
plt.ylabel('Root mean Square Error')
plt.title('Performance Over {} Replications'.format(num_reps))
plt.grid()
plt.legend()
plt.show()

plt.plot(np.arange(num_episodes), g_avi_post, label='AVI (Post)')
plt.plot(np.arange(num_episodes), g_avi_pre, label='AVI (Pre)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.grid()
plt.legend()
plt.show()
