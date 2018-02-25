#!/usr/bin/env python
import gym
from pathenv.agent import DeepQLearningAgent
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch


action_direction = {
    0: 'N',
    1: 'E',
    2: 'S',
    3: 'W'
}

def play_visual(env, agent):
    s = env.reset()
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    env.render(ax)
    while True:
        _, a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        agent.update_memory(s, a, r, next_s, done)
        s = next_s
        print('action: {}  reward: {}'.format(action_direction[a], r))
        if r != -1:
            env.render(ax)
        if done:
            break
        agent.train_on_memory()



if __name__ == '__main__':
    env = gym.make('PathFindingByPixelWithDistanceMapEnv-v1')
    env._configure()

    agent = DeepQLearningAgent(epsilon=0.1, max_memory_size=1, batch_size=1)

    agent.model.load_state_dict(torch.load('agent_model'))
    play_visual(env, agent)
