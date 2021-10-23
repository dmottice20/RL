import torch as to
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import math
from supporting_functions import contribution_function


############################################
# Step 0: Preprocessing work               #
############################################
def constant_alpha(m): return 0.5
def constant_epsilon(m): return 0.1


def vfa_randomized_policy(s, q_table, m):
    exploit_action = to.argmax(q_table[s, :]).item()
    if random.random() > constant_epsilon(m):
        a_star = exploit_action
    else:
        actions = [a for a in range(4)]
        actions.remove(exploit_action)
        a_star = random.choice(actions)
    return a_star


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

# Discount factor, rng seed, reps, and episodes parameters...
gamma = 1
random.seed(1)
num_reps, num_episodes = 100, 500

# Problem parameters (initial)...
initial_state = 36
terminal_states = [37, 38]
g_rep_m = to.zeros(size=(num_reps, num_episodes))
rmse_rep_m = to.zeros(size=(num_reps, num_episodes))

# Load optimal value fx...
v_star = to.tensor([[i for i in range(-14, -3 + 1, 1)],
                    [i for i in range(-13, -2 + 1, 1)],
                    [i for i in range(-12, -1 + 1, 1)],
                    [-13 if i == 0 else 0 for i in range(12)]]
                   )

# Perform num_reps replications...
for rep in tqdm(range(num_reps), desc='performing {} replications'.format(num_reps)):
    # Initialize the Q-factors...
    # a/ Do it optimistically?
    q_bar = to.zeros(size=(39, 4))
    # b/ Do it pessimistically?
    # q_bar = -25 * to.ones(size=(39,4))
    # c/ Just the terminal states to zero?
    # Go forward num_episodes episodes...
    for m in range(num_episodes):
        # Select an initially state...
        # a/ Statically?
        state_t = initial_state
        # b/ Randomly?
        # state_t = random.randint(0, 36)
        # Select action based on randomized policy derived from Q...
        action_t = vfa_randomized_policy(state_t, q_bar, m)
        # Initialize cumulative reward (return) statistic...
        g_sarsa = 0
        # While not in the terminal state...
        while state_t not in terminal_states:
            # Compute the next state...
            state_t_plus_1 = system_model(state_t, action_t)
            # Compute and updated reward...
            contribution_t = contribution_function(state_t, action_t, state_t_plus_1)
            g_sarsa += contribution_t
            # Compute next action...
            action_t_plus_1 = vfa_randomized_policy(state_t_plus_1, q_bar, m)
            # Compute q_hat...
            q_hat_t = contribution_t + gamma * q_bar[state_t_plus_1, action_t_plus_1].item()
            # Compute updated Q-value...
            q_bar[state_t, action_t] = (1 - constant_alpha(m)) * q_bar[state_t, action_t].item() + constant_alpha(m)*q_hat_t
            # Update state and action...
            state_t, action_t = state_t_plus_1, action_t_plus_1

        # Record and return earned in episode m of repetition z...
        g_rep_m[rep, m] = g_sarsa
        # Compute root mean squared error...
        v_adp, _ = to.max(q_bar, 1)
        a = to.reshape(v_adp[:36], (3, 12))
        b = to.hstack((v_adp[37], to.zeros(11)))
        v_adp_full = to.vstack((a, b.reshape(1, 12)))
        rmse_rep_m[rep, m] = math.sqrt(to.mean((v_adp_full - v_star) ** 2).item())


# Average returns across all episode...
g_sarsa = to.mean(g_rep_m, 0)
# Average RMSE across all episodes...
rmse_sarsa = to.mean(rmse_rep_m, 0)

# Load the values from q-learning...
g_ql = to.load('scw_q_learning_g_q.pt')
rmse_ql = to.load('scw_q_learning_rmse.pt')

# Plot performance...
plt.plot(range(num_episodes), rmse_ql, label='Q-Learning')
plt.plot(range(num_episodes), rmse_sarsa, label='SARSA')
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance over {} Replications'.format(num_reps))
plt.legend()
plt.show()

plt.plot(range(num_episodes), g_ql, label='Q-Learning')
plt.plot(range(num_episodes), g_sarsa, label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Online Performance over {} Replications'.format(num_reps))
plt.legend()
plt.show()
