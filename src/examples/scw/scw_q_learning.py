import torch as to
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
from supporting_functions import exploit_action


# Initialize algorithm parameters and variables...
# Define functions for step-size rule and/or the random exploration probabilities...
def constant_lr(m): return 0.7


def polynomial_lr(m): return 1 / (m ** 0.5)


def constant_er(m): return 0.15


def generalized_harmonic_er(m, a, b): return a / (b + m)


def polynomial_er(m, beta): return 1 / (m ** beta)


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
            elif 25 <= s <= 34:
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

    return stp1


def contribution_function(state_n, state_n_plus_1):
    if state_n_plus_1 == 37:
        c_n = -100
    elif state_n == 38:
        c_n = 0
    else:
        c_n = -1
    return c_n


# Other key algorithm parameters....
gamma = 1
num_reps, num_episodes = 100, 500
initial_state = 36
terminal_states = [37, 38]
card_s, card_a = 39, 4
random.seed(1)

# Storage tensors for graphs on algorithm performance...
g_rep_m = to.empty(size=(num_reps, num_episodes))
rmse_rep_m = to.empty(size=(num_reps, num_episodes))

# Load the v_star from the exact solution...
v_star = to.tensor([[i for i in range(-14, -3 + 1, 1)],
                    [i for i in range(-13, -2 + 1, 1)],
                    [i for i in range(-12, -1 + 1, 1)],
                    [-13 if i == 0 else 0 for i in range(12)]]
                   )

# Do num_reps replications to adequately account for the randomness
# present in the algorithm.
for rep in tqdm(range(num_reps), desc='performing {} replications for q-learning'.format(num_reps)):
    # Initialize the Q-factors...
    # Optimistic initialization...
    q_bar = to.zeros(size=(card_s, card_a))

    if rep == 50 or rep == 75:
        print('===At replication {}==='.format(rep))

    # Go forward num_episodes...
    for m in range(num_episodes):
        # Select the initial state statically...
        state_n = initial_state
        # Select initial state randomly...
        # state_n = random.randint(0, 36)
        g_q_learning = 0

        if rep == 50 or rep == 75:
            print('=====EPISODE = {}====='.format(m))

        # Simulate while not in the terminal state...
        while state_n not in terminal_states:
            if random.random() > constant_er(m):
                # Exploit, i.e. choose the action using Q-bars...
                action_t = exploit_action(state_n, q_bar)
            else:
                # Explore, i.e. choose an action at randomly...
                excluded_action = exploit_action(state_n, q_bar)
                actions = [a for a in range(4)]
                actions.remove(excluded_action)
                action_t = random.choice(actions)

            # Compute the next state...
            state_n_plus_1 = system_model(state_n, action_t)
            # Print results for debugging...
            if rep == 50 or rep == 75:
                print('Current State: {}, Current Action: {}, Next State: {}'.format(state_n, action_t, state_n_plus_1))
            contribution_t = contribution_function(state_n, state_n_plus_1)
            g_q_learning += contribution_t
            # Compute q_hat_t...
            q_hat_t = contribution_t + gamma * to.max(q_bar[state_n_plus_1, :]).item()
            # Compute the updated Q-bar value...
            q_bar[state_n, action_t] = (1 - constant_lr(m)) * q_bar[state_n, action_t].item() + constant_lr(m) * q_hat_t
            # Update the state and action...
            state_n = state_n_plus_1

        # Record the return earned in episode m of repetition z...
        g_rep_m[rep, m] = g_q_learning
        if rep == 50 or rep == 75:
            print('Reward in episode {} is {}'.format(m, g_q_learning))
        # Compute the root mean square error...
        v_adp, _ = to.max(q_bar, 1)
        a = to.reshape(v_adp[:36], (3, 12))
        b = to.hstack((v_adp[37], to.zeros(11)))
        v_adp_full = to.vstack((a, b.reshape(1, 12)))
        # RMSE[rep, m] = math.sqrt(to.mean((VadpFull - Vstar) ** 2).item())
        rmse_rep_m[rep, m] = math.sqrt(to.mean((v_adp_full - v_star) ** 2).item())

# Average returns across all runs for each episode...
g_q_learning = to.mean(g_rep_m, 0)
rmse_q_learning = to.mean(rmse_rep_m, 0)

# Save the data...
to.save(g_q_learning, 'scw_q_learning_g_q.pt')
to.save(rmse_q_learning, 'scw_q_learning_rmse.pt')

# Plot performance...
plt.plot(range(num_episodes), rmse_q_learning)
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance over {} Replications'.format(num_reps))
plt.show()

plt.plot(range(num_episodes), g_q_learning)
plt.xlabel('Episode')
plt.ylabel('G_m(S_0)')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.show()
