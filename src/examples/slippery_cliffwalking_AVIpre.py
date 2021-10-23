import torch as to
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
from supporting_functions import approximate_bellman_equation


# STATE SPACE...
#  0  1  2  3  4  5  6  7  8  9 10 11
# 12 13 14 15 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31 32 33 34 35
# 36 -------- cliff = 37 --------- 38

# Functions to be used...
def constant_alpha(m): return 0.5


def non_constant_alpha(m, beta=0.5): return 1 / (m ** beta)


# 2/ Random exploration % not used.
def constant_epsilon(m): return 0.1


def polynomial_epsilon(m, beta=0.5): return 1 / (m ** beta)


def generalized_harmonic_epsilon(m, a=1): return a / (a + m)


# PREPROCESSING WORK...
num_samples, num_reps, num_episodes = 2, 100, 500
seed = 1
random.seed(1)
gamma = 1

# Storage for data earned during a repetition...
Grepm = to.empty(size=(num_reps, num_episodes))
RMSE = to.empty(size=(num_reps, num_episodes))

# Load the Vstar...
v_star = to.load('Vstar_SCW.pt')

# Static initial state...
initial_state = 36

# Also, define 37 to be cliff, and 38 to be goal.
terminal_states = [37, 38]

# Set n = 1.
for n in tqdm(range(num_reps), desc='replications for the approximate value iteration (pre)'):
    # STEP 0: Initialization...
    # Initialize Vbar^0(s) for all s in S.
    Vbar = to.zeros(1, 39)
    # Choose an initial state S^1_0.
    # Static or random (exploring) starts.
    state_n = initial_state
    # state_n = random.randint(0, 36)
    # STEP 1: Choose a random sample of outcomes representing possible
    # realizations of the information arriving in between (t, t + 1).

    # Loop over various episodes...
    for m in range(num_episodes):
        # Target for the episode of the AVI pre algorithm on episode m...
        g_aviPre = 0
        # Check if in terminal state...
        while state_n not in terminal_states:
            # Solve the approximate Bellman update using current Vbar...
            vhatt, state_n_plus_1, reward = approximate_bellman_equation(state_n, Vbar, gamma, num_samples)
            # Update our value function approximation....
            Vbar[0, state_n] = (1 - constant_alpha(m)) * Vbar[0, state_n] + constant_alpha(m) * vhatt
            # Update our return...
            g_aviPre += reward
            # Update to new system state...
            state_n = int(state_n_plus_1)

        # Record the return earned in episode m of repetition z...
        Grepm[n, m] = g_aviPre
        # Compute the RMSE...
        # Convert to the rectangular version done w/ the exact algorithm...
        a = to.reshape(Vbar[0, :36], (3, 12))
        b = to.hstack((Vbar[0, 37], to.zeros(11)))
        v_adp = to.vstack((a, b.reshape(1, 12)))
        RMSE[n, m] = math.sqrt(to.mean((v_adp - v_star) ** 2).item())


g_avi_pre = to.mean(Grepm, 0)
rmse_avi_pre = to.mean(RMSE, 0)

plt.plot(range(500), rmse_avi_pre)
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance Over {} Replications'.format(num_reps))
plt.show()

plt.plot(range(500), g_avi_pre)
plt.xlabel('Episode')
plt.ylabel('G_m(S_0)')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.show()
