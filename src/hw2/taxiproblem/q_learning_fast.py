import torch as to
import timeit, random, itertools
import numpy as np


# Construct the state space.
state_space = to.as_tensor(list(itertools.product(to.arange(25), to.arange(5), to.arange(4))))
state_space = to.vstack((state_space, to.as_tensor([25, 5, 5])))
card_s = state_space.shape[0]

# Use a matrix / tensor to store indices to state space mapping
# and not a hash.
# Col 0: Index, Cols 1-3: taxi, customer, destination variables.
state_space = to.hstack((to.arange(card_s).reshape(card_s, 1), state_space))

# Define the set of allowable states and starting states.
allowable_states = []
for i in range(501):
    if state_space[i][2].item() != state_space[i][3].item():
        allowable_states.append(i)
allowable_states.append(500)
# Define the set of possible starting states.
starting_states = []
for i in range(501):
    if state_space[i][2].item() < 4 and state_space[i][2].item() != state_space[i][3].item():
        starting_states.append(i)

# Initialize algorithm params and variables.
def alpha(m): return 0.8
def epislon(m): return 0.25


gamma, num_reps, num_episodes = 0.95, 10, 1000
random.seed(1)
terminal_state = 500

# Storage tensors for graphs on algorithm performance.
g_per_episode = to.empty(size=(num_reps, num_episodes))
rmse_per_episode = to.empty(size=(num_reps, num_episodes))

# Load the v star.
v_star = to.load('src/hw2/taxiproblem/v_star_taxi.pt')
# v_star = to.load('v_star_taxi.pt')
v_star = to.tensor(v_star)

start_time = timeit.default_timer()

for rep in range(num_reps):
    q_bar = to.zeros(size=(card_s, 6))
    for m in range(num_episodes):
        state_t = random.choice(starting_states)
        g = 0
        while state_t != terminal_state:
            exploiting_action = to.argmax(q_bar[state_t, :]).item()
            if random.random() > epislon(m):
                action_t = exploiting_action
            else:
                # Explore...
                actions = list(np.arange(6))
                actions.remove(exploiting_action)
                action_t = random.choice(actions)

        state_t_plus_1 = system_model(state_t, action_t, indices)

