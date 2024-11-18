"""
  Run this file at first, in order to see what is it printing. Instead of the print() use the respective log level.
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import csv


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():

    def plot1(self, rewards, algorithm_names):
        """
        Visualize the performance of each bandit over time in linear and log
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for i, reward in enumerate(rewards):
            cumulative_rewards = np.cumsum(reward)
            plt.plot(cumulative_rewards, label=algorithm_names[i])
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards (Linear Scale)')
        plt.legend()

        plt.subplot(1, 2, 2)
        for i, reward in enumerate(rewards):
            cumulative_rewards = np.cumsum(reward)
            plt.plot(cumulative_rewards, label=algorithm_names[i])
        plt.xscale('log')
        plt.xlabel('Trial (Log Scale)')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards (Log Scale)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, e_greedy_rewards, thompson_rewards):

         # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(e_greedy_rewards), label='Epsilon-Greedy')
        plt.plot(np.cumsum(thompson_rewards), label='Thompson Sampling')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Comparison')
        plt.legend()

        plt.subplot(1, 2, 2)
        optimal_rewards = np.maximum(e_greedy_rewards, thompson_rewards)
        e_greedy_regrets = optimal_rewards - e_greedy_rewards
        thompson_regrets = optimal_rewards - thompson_rewards
        plt.plot(np.cumsum(e_greedy_regrets), label='Epsilon-Greedy Regrets')
        plt.plot(np.cumsum(thompson_regrets), label='Thompson Sampling Regrets')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regrets Comparison')
        plt.legend()

        plt.tight_layout()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        self.p = p
        self.epsilon = epsilon
        self.n_arms = len(p)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.rewards = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon}, probabilities={self.p})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n

    def experiment(self, N):
        for _ in range(N):
            chosen_arm = self.pull()
            reward = int(np.random.rand() < self.p[chosen_arm])  # Convert reward to int
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
        return self.rewards

    def report(self):
        average_reward = np.mean(self.rewards)
        optimal = max(self.p)
        regrets = [optimal - r for r in self.rewards]
        average_regret = np.mean(regrets)

        logger.info(f"Epsilon-Greedy Average Reward: {average_reward:.4f}")
        logger.info(f"Epsilon-Greedy Average Regret: {average_regret:.4f}")

        with open('epsilon_greedy_report.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Arm', 'Reward', 'Algorithm'])
            for i, reward in enumerate(self.rewards):
                writer.writerow([i, reward, 'Epsilon-Greedy'])

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p):
        self.p = p
        self.n_arms = len(p)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.rewards = []

    def __repr__(self):
        return f"ThompsonSampling(probabilities={self.p})"

    def pull(self):
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward

    def experiment(self, N):
        for _ in range(N):
            chosen_arm = self.pull()
            reward = int(np.random.rand() < self.p[chosen_arm])
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
        return self.rewards

    def report(self):
        average_reward = np.mean(self.rewards)
        regrets = [max(self.p) - r for r in self.rewards]
        average_regret = np.mean(regrets)

        logger.info(f"Thompson Sampling Average Reward: {average_reward:.4f}")
        logger.info(f"Thompson Sampling Average Regret: {average_regret:.4f}")

        with open('thompson_sampling_report.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Arm', 'Reward', 'Algorithm'])
            for i, reward in enumerate(self.rewards):
                writer.writerow([i, reward, 'Thompson Sampling'])

#--------------------------------------#

def comparison(e_greedy_rewards, thompson_rewards):
    """
    Compare the performances of the two algorithms visually.
    """
    plt.figure(figsize=(12, 6))

    # Cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(e_greedy_rewards), label='Epsilon-Greedy')
    plt.plot(np.cumsum(thompson_rewards), label='Thompson Sampling')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Comparison')
    plt.legend()

    # Cumulative regrets
    plt.subplot(1, 2, 2)
    optimal_rewards = np.maximum(e_greedy_rewards, thompson_rewards)
    e_greedy_regrets = optimal_rewards - e_greedy_rewards
    thompson_regrets = optimal_rewards - thompson_rewards
    plt.plot(np.cumsum(e_greedy_regrets), label='Epsilon-Greedy Regrets')
    plt.plot(np.cumsum(thompson_regrets), label='Thompson Sampling Regrets')
    plt.xlabel('Trial')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regrets Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    bandit_probs = [0.2, 0.4, 0.6, 0.8]
    N = 20000
    epsilon = 0.1

    eg = EpsilonGreedy(bandit_probs, epsilon)
    ts = ThompsonSampling(bandit_probs)

    eg_rewards = eg.experiment(N)
    ts_rewards = ts.experiment(N)

    viz = Visualization()
    viz.plot1([eg_rewards, ts_rewards], ['Epsilon-Greedy', 'Thompson Sampling'])

    eg.report()
    ts.report()
    comparison(eg_rewards, ts_rewards)