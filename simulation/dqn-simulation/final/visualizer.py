import pickle
import numpy as np
import matplotlib.pyplot as plt


cumulative_rewards = pickle.load(open('cum_rewards_history-12.pkl', 'rb'))
epsilons = pickle.load(open('epsilon_history-12.pkl', 'rb'))

# Set general font size
plt.rcParams['font.size'] = '24'

ax = plt.subplot(211)
plt.title("Cumulative Rewards over Episodes", fontsize=24)
plt.plot(np.arange(len(cumulative_rewards)) + 1, cumulative_rewards)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

ax = plt.subplot(212)
plt.title("Epsilons over Episodes", fontsize=24)
plt.plot(np.arange(len(epsilons)) + 1, epsilons)
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)

plt.show()
