from system_model import system_model, contribution_fx
import torch as to
import statsmodels.stats.api as sms
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Read in the data and separate into appropriate numpy arrays.
data = pd.read_csv('ARP_3650_data.csv').values
# data = pd.read_csv('src/hw2/arp/ARP_3650_data.csv').values
cost, value, op_cost, prob = np.hsplit(data, 4)
data_results = pd.read_csv('hw1problem8results.csv')
card_s = 3650
card_a = 3650+1
# Define the myopic policy.
# myopic_policy = to.zeros(card_s)
# for s in tqdm(range(card_s), desc='building the myopic policy'):
#     max_a = None
#     best_contribution = -np.inf
#     for a in range(card_a):
#         contribution = contribution_fx(s, a, op_cost, value, cost)
#         if contribution > best_contribution:
#             max_a = a
#             best_contribution = contribution
#
#     myopic_policy[s] = max_a

# Save the myopic policy.
# to.save(myopic_policy, 'myopic-policy.pt')

# Load the myopic policy.
optimal_policy = to.tensor(data_results['pi_optimal'].values)
myopic_policy = to.load('myopic-policy.pt')
print(myopic_policy.shape)
random.seed(1)
num_reps = 10
days = 3650
contribution_data = to.empty(num_reps)
for rep in tqdm(range(num_reps), desc='simulating 100 3650-day trajectories'):
    # Initialize the contributions earned to 0.
    contribution = 0
    state = random.choice(list(np.arange(card_s)))
    # Simulate forward 3650 days.
    for d in range(days):
        action = int(myopic_policy[state].item())
        next_state = system_model(state, action, prob)
        contribution += contribution_fx(state, action, op_cost, value, cost) * (0.9**d)
        state = next_state

    # Store the contribution earned in that replication.
    contribution_data[rep] = contribution

# Save the data.
to.save(contribution_data, 'contribution-myopic-policy.pt')

# Calculate 95% CI.
print('Myopic Policy:', sms.DescrStatsW(contribution_data.numpy()).tconfint_mean())

# # Optimal policy
random.seed(1)
optimal_data = to.empty(num_reps)
for rep in tqdm(range(num_reps), desc='simulating 100 3650-day trajectories'):
    # Initialize the contributions earned to 0.
    contribution = 0
    # Random initial start.
    state = random.choice(list(np.arange(card_s)))
    # Simulate forward 3650 days.
    for d in range(days):
        action = int(optimal_policy[state].item())
        next_state = system_model(state, action, prob)
        contribution += contribution_fx(state, action, op_cost, value, cost) * (0.9**d)
        state = next_state

    # Store the contribution earned in that replication.
    optimal_data[rep] = contribution

print('Optimal Policy:', sms.DescrStatsW(optimal_data.numpy()).tconfint_mean())

# AVI Pre
avi_pre = to.tensor(data_results['pi_early_iteration'].values)
avi_pre_data = to.empty(num_reps)
for rep in tqdm(range(num_reps), desc='simulating 100 3650-day trajectories'):
    # Initialize the contributions earned to 0.
    contribution = 0
    # Random initial start.
    state = random.choice(list(np.arange(card_s)))
    # Simulate forward 3650 days.
    for d in range(days):
        action = int(avi_pre[state].item())
        next_state = system_model(state, action, prob)
        contribution += contribution_fx(state, action, op_cost, value, cost) * (0.9**d)
        state = next_state

    # Store the contribution earned in that replication.
    avi_pre_data[rep] = contribution

print('AVI (Pre) Policy:', sms.DescrStatsW(avi_pre_data.numpy()).tconfint_mean())
