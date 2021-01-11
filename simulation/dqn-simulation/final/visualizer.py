import pickle
import numpy as np
import matplotlib.pyplot as plt


cumulative_rewards = pickle.load(open('cum_rewards_history-12.pkl', 'rb'))
epsilons = pickle.load(open('epsilon_history-12.pkl', 'rb'))
plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.title("Cumulative Rewards over Episodes")
plt.plot(np.arange(len(cumulative_rewards)) + 1, cumulative_rewards)
plt.subplot(212)
plt.title("Epsilons over Episodes")
plt.plot(np.arange(len(epsilons)) + 1, epsilons)
plt.show()
