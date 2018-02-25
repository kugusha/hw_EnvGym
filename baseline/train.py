#!/usr/bin/env python
import gym
from pathenv.agent import DeepQLearningAgent
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch


episodes_number = 5000


def play_episode(env, agent, t_max=500):
    total_reward = 0.0
    qvalues = [0]
    s = env.reset()
    for t in range(t_max):
        qval, a = agent.get_action(s)

        if qval is not None:
            qvalues.append(qval)
        next_s, r, done, _ = env.step(a)
        agent.update_memory(s, a, r, next_s, done)

        total_reward += r
        if done:
            break

    return total_reward, np.mean(qvalues)


if __name__ == '__main__':
    env = gym.make('PathFindingByPixelWithDistanceMapEnv-v1')
    env._configure()

    agent = DeepQLearningAgent()

    rewards = []
    qvalues = []
    plt.ion()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plt.show()

    ax1.set_xlabel('episode number', size=15)
    ax1.set_ylabel('reward', size=15)
    ax2.set_xlabel('episode number', size=15)
    ax2.set_ylabel(r'mean $\max Q(s, a)$', size=15)
    for i in xrange(1, episodes_number):
        reward, qval = play_episode(env, agent)
        rewards.append(reward)
        qvalues.append(qval)
        agent.train_on_memory()

        if i % 100 == 0:
            print('episode: {}/{}, mean_reward: {}, mean_Qvalue: {}'.format(i, episodes_number, np.mean(rewards[-100:]), np.mean(qvalues[-100:])))
            ax1.plot(rewards, color='#1c6dc9')
            ax2.plot(qvalues, color='#1c6dc9')
            plt.draw()
            plt.pause(1)
            torch.save(agent.model.state_dict(), 'agent_model')