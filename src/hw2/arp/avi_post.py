import torch as to
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


#####################################################
# Approximate Value Iteration (Post-decision State) #
#####################################################


# Necessary functions.
def system_model_w(post_decision_state):
    """
    Fx to apply the possibility of the car failing (w) and moving to age M.
    :param post_decision_state:
    :return: pre_decision_state
    """
    # If in the state M (or M - 1 in python), I am staying there no matter what.
    if post_decision_state == M - 1:
        pre_decision_state = M - 1
    # Else, if in the state M - 1 (or M - 2 in python), I am moving to M no matter what.
    elif post_decision_state == M - 2:
        pre_decision_state = M - 1
    # Otherwise, if rand() < probability of survival in that state, age by 1 month.
    else:
        if random.random() < prob[post_decision_state + 1]:
            pre_decision_state = post_decision_state
        else:
            pre_decision_state = M - 1

    return pre_decision_state


def system_model_action(pre_decision_state, action):
    """
    Fx to simulate to post_decision state given a state action pair
    before the W_{t+1} comes into account.
    :param pre_decision_state:
    :param action:
    :return: post_decision_state:
    """
    # If action is to do nothing, move to age + 1.
    if action == 0:
        # If in last state, will stay there.
        if pre_decision_state == M - 1:
            post_decision_state = M - 1
        # Otherwise, age by 1 month.
        else:
            post_decision_state = pre_decision_state + 1
    # If action is to purchase M - 1 old car, go to M - 1...
    elif action == M:
        post_decision_state = M - 1
    # Otherwise...  move to action.
    else:
        post_decision_state = action
    return post_decision_state


def contribution_fx(state, action):
    """
    Fx. to calculate contribution earned in epoch given state, action pair.
    :param state: state at time t
    :param action: action taken at time t
    :return: contribution: contribution earned
    """
    # If my action is to do nothing...
    if action == 0:
        # Only incur the operating cost.
        contribution = -op_cost[state - 1]
    # Otherwise...
    else:
        # Earn the trade-in value then subtract the cost of buying car of age (action-1) and its operating cost.
        contribution = value[state - 1] - cost[action - 1] - op_cost[action - 1]
    return contribution[0]


def approximate_bellman_equation(state_t_pre, value, gamma):
    value_hat = to.zeros(size=(1, 4))
    states = to.zeros(size=(1, 4), dtype=to.int)
    for a in np.arange(4):
        states[:, a] = system_model_action(state_t_pre, a)
        # Compute the value.
        value_hat[:, a] = contribution_fx(state_t_pre, a) + gamma * value[states[:, a].item()]

    v_hat, a_star = to.max(value_hat).item(), to.argmax(value_hat).item()

    return {
        'value estimate': v_hat,
        'post-decision state': states[:, a_star].item(),
        'reward': contribution_fx(state_t_pre, a_star)
    }


# Initialize algorithm params and variables.
def alpha(m): return 0.4
def epsilon(m): return 0.1


gamma, num_reps, num_episodes, num_steps = 0.9, 10, 1000, 1000
random.seed(1)

# Initialize storage tensors.
g_per_episode = to.empty(size=(num_reps,  num_episodes))
rmse_per_episode = to.empty(size=(num_reps, num_episodes))

# Load optimal v-star for RMSE computation.
data = pd.read_csv('hw1problem8results.csv')
v_star = data['V_optimal'].values
data = pd.read_csv('ARP_3650_data.csv').values
# data = pd.read_csv('src/hw2/arp/ARP_3650_data.csv').values
cost, value, op_cost, prob = np.hsplit(data, 4)
del data
M = 3650

# Loop for num_reps replications.
for rep in tqdm(range(num_reps), desc=
                'AVI Post - {} replications'.format(num_reps)):
    # Initialize value function approximation (VFA).
    v_bar = to.zeros(M)
    # Loop for num_episodes episodes...
    for m in range(num_episodes):
        # Initialize reward target.
        g = 0
        # Select initial post-decision state randomly.
        state_t_post = random.choice(np.arange(M))
        # Simulate transition to next pre-decision state.
        state_t_pre = system_model_w(state_t_post)
        # Simulate forward some num_steps.
        for _ in range(num_steps):
            # Solve approximate bellman equation using current v_bar.
            solution = approximate_bellman_equation(state_t_pre, v_bar, gamma)
            # What is the estimated value of current state, then update VFA.
            v_hat_t = solution['value estimate']
            v_bar[state_t_post] = (1-alpha(m))*v_bar[state_t_post].item() + alpha(m) * v_hat_t
            # Record post-decision state.
            state_tp1_post = solution['post-decision state']
            # Simulate transition to next pre-decision state.
            state_tp1_pre = system_model_w(state_tp1_post)
            # Update the return.
            g += solution['reward']
            # Update the state variables
            state_t_post, state_t_pre = state_tp1_post, state_tp1_pre

        # Record the return earned in the n-steps forward.
        g_per_episode[rep, m] = g
        # Same for root mean square error.
        rmse_per_episode[rep, m] = np.sqrt(to.mean((v_bar - v_star) ** 2).item())

# Average returns and root mean square error across all episodes.
g_avi_pre = to.mean(g_per_episode, 0)
rmse_avi_pre = to.mean(rmse_per_episode, 0)

# Plot the results.
plt.plot(np.arange(num_episodes), rmse_avi_pre)
plt.xlabel('Episode')
plt.ylabel('Root mean Square Error')
plt.title('Performance Over {} Replications'.format(num_reps))
plt.grid()
plt.show()

plt.plot(np.arange(3650), v_bar, label='AVI (Pre) Estimate')
plt.plot(np.arange(3650), v_star, label='Optimal Value Fx.')
plt.xlabel('Age of Car (i.e. state in S)')
plt.ylabel('Value of age of Car, V(S)')
plt.legend()
plt.grid()
plt.show()
