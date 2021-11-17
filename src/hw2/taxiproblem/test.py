import matplotlib.pyplot as plt
import numpy as np
import torch as to
import random


# Load the data.
sarsa_first = to.load('rmse-sarsa-top1.pt')
sarsa_second = to.load('rmse-sarsa-top2.pt')
sarsa_third = to.load('rmse-sarsa-top3.pt')
ql_first = to.load('rmse-ql-top1.pt')
ql_second = to.load('rmse-ql-top2.pt')
ql_third = to.load('rmse-ql-top3.pt')

plt.plot(np.arange(1000), sarsa_first, color='blue', linestyle='-', label='SARSA: 1st Setting')
plt.plot(np.arange(1000), sarsa_second, color='blue', linestyle='-.', label='SARSA: 2nd Setting')
plt.plot(np.arange(1000), sarsa_third, color='blue', linestyle=':', label='SARSA: 3rd Setting')
plt.plot(np.arange(1000), ql_first, color='red', linestyle='-', label='QL: 1st Setting')
plt.plot(np.arange(1000), ql_second, color='red', linestyle='-.', label='QL: 2nd Setting')
plt.plot(np.arange(1000), ql_third, color='red', linestyle=':', label='QL: 3rd Setting')
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.grid()
plt.legend()
plt.savefig('compare.png')
plt.show()
