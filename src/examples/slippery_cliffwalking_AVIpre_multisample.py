import torch as to
import numpy as np
from scipy.sparse import eye
import random
from tqdm import tqdm
import math
from supporting_functions import approximate_bellman_equation


####################################################
#        Approximate Value Iteration (Pre)         #
####################################################
# Initialize algorithm parameters and variables.
# 1/ Stepsize rule for the learning rate.
def constant_alpha(m):
    return 0.5


def non_constant_alpha(m):
    beta = 0.5
    return 1 / (m ** beta)


# 2/ Random exploration % not used.
def epsilon(m):
    return 0.1


def non_constant_epsilon(m):
    beta = 0.5
    return 1 / (m ** beta)


# 3/ Various other simple parameters.
# Num samples to sample, reps of the algorithm,
# and the num of episodes.
random.seed(1)
num_samples, num_reps, num_episodes = 2, 100, 500
gamma = 1
# Initial state, and terminal states....
S0, Sdelta = 36, [37, 38]

# Table for the sum of rewards during an episode to record online performance...
Grepm = to.empty(num_reps, num_episodes)

# Table for the cumulative Q-values across all algorithm replications (for RMSE computation)...
RMSE = to.empty(num_reps, num_episodes)

# Load the Vstar...
Vstar = to.load('Vstar_SCW.pt')

# Load the optimal value function (for RMSE computation)
# loop over 0, ..., num_reps - 1
for rep in tqdm(range(num_reps), desc='replications...'):
    # VFA initialization....
    # Optimistic initialization...
    Vbar = to.zeros(1, 39)
    # Pessimistic initialization...
    # Vbar = -25 * to.zeros(39, 4)
    # Loop over each episode...
    # 0, ..., num_episodes - 1
    for m in range(num_episodes):
        # initialize discounted cumulative reward statistic (return)...
        GaviPre = 0
        # Select initial state...
        # Static start
        St = S0
        # Random, exploring starts...
        # St = to.randint(37, size=(1,)).item()

        # While not in the terminal state...
        while St not in Sdelta:
            # Solve the approximate Bellman update using current Vbar...
            solution = approximate_bellman_equation(St, Vbar, gamma, num_samples)
            # Estimated value of current state...
            vhatt = solution['vhat']
            # Next system state...
            Stp1 = solution['stp1']
            # Update VFA...
            # Using constant alpha
            Vbar[0, St] = (1-constant_alpha(m))*Vbar[0, St].item() + constant_alpha(m)*vhatt
            # Update return
            GaviPre = GaviPre + solution['reward']
            # Update state...
            St = int(Stp1)

        # Record return earned in episode m of algorithm run z...
        Grepm[rep, m] = GaviPre

        # Compute RMSE... convert to rectangular representation for clarity
        # and to match Vstar representation...
        Vstar_a = to.reshape(Vbar[0, :36], (3, 12))
        Vstar_b = to.hstack((Vbar[0, 37], to.zeros(11)))
        VadpFull = to.vstack((Vstar_a, Vstar_b.reshape(1, 12)))
        RMSE[rep, m] = math.sqrt(to.mean((VadpFull - Vstar) ** 2).item())

# Average returns across all runs for each episode...
# Grab the mean of each column...
GaviPre = to.mean(Grepm, 0)
rmse_aviPre = to.mean(RMSE, 0)

# Plot the results...
print('Look at the results')
x = 4 + 4
