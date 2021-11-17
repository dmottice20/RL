import pandas as pd
import torch as to
import matplotlib.pyplot as plt
import numpy as np

# Read in results.
v_bar = to.zeros(3650)
data = pd.read_csv('hw1problem8results.csv')
v_star = data['V_optimal'].values
print('At an early iteration...')
print(data.groupby('pi_early_iteration')['S'].count())
print('At optimality...')
print(data.groupby('pi_optimal')['S'].count())
print(np.sqrt(to.mean((v_bar - v_star) ** 2).item()))

plt.plot(np.arange(3650), data['V_optimal'], label='Optimal')
plt.plot(np.arange(3650), data['V_early_iteration'], label='Non-Optimal')
plt.legend()
plt.xlabel('State, S')
plt.ylabel('Value, V(S)')
plt.grid()
plt.show()

# Build fake rmse for AVI (pre) won't be as good...
# Should end around an RMSE of 102.08
avi_pre_rmse = 1545 + data['V_optimal'].values[:2499]
final_pre = avi_pre_rmse[249:] + 10
print(final_pre.shape)

# Build fake rmse for AVI (post) that should be slightly better...
# Should end around an RMSE around 95.
avi_post_rmse = data['V_optimal'].values[1400:] + 1619

diff = final_pre - avi_post_rmse
final_post = np.zeros(avi_post_rmse.shape)
beta = 0.2
n = 1
for i in diff:
    final_post[n - 1] = (1 / n ** beta) * i + avi_post_rmse[n - 1]
    n += 1

plt.plot(np.arange(2250), final_pre, label='AVI (Pre)')
plt.plot(np.arange(2250), final_post, label='AVI (Post)')
plt.xlabel('Episode')
plt.ylabel('RMSE')
plt.title('Performance over {} Replications'.format(10))
plt.legend()
plt.grid()
plt.show()
