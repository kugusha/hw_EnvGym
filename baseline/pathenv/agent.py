import collections
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(400 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.linear_layers(conv_out.view(conv_out.size(0), -1))

class DeepQLearningAgent(object):
    def __init__(self,
                 input_shape=(20, 20),
                 number_of_actions=4,
                 max_memory_size=1000,
                 epsilon=1,
                 learning_rate=0.01,
                 discount=0.99,
                 batch_size=128,
                 epsilon_min=0.1,
                 epsilon_decay=0.999):
        self.input_shape = input_shape
        self.number_of_actions = number_of_actions
        self.max_memory_size = max_memory_size

        self.goal = None
        self.memory = []

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self._build_model()


    def __repr__(self):
        return self.__class__.__name__

    def _build_model(self):
        self.model = CNN(self.input_shape, self.number_of_actions)
        # self.model = nn.Sequential(
        #     nn.Linear(self.input_shape[0] * self.input_shape[1], 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 4)
        # )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def get_qvalues(self, state, vol=False):
        if state.ndim < 4:
            input_field = state.reshape(1, *state.shape)
        else:
            input_field = state

        return self.model(Variable(torch.from_numpy(input_field).float(), volatile=vol))

    def get_policy(self, state):
        max_q = self.get_qvalues(state, True).max(1)
        qvalue = max_q[0].data[0]
        action = max_q[1].data[0]
        return qvalue, action

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return None, random.choice(range(self.number_of_actions))
        return self.get_policy(state)

    def train_on_memory(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = np.array(random.sample(self.memory, min(self.batch_size, len(self.memory))))

        updates = torch.from_numpy(np.array(minibatch[:,2], np.float32))

        done_ids = torch.from_numpy(np.array(minibatch[:,4], np.uint8))
        non_final_next_states = np.array([x[3] for x in minibatch if x[4] is False])
        max_new_qvalues = self.get_qvalues(non_final_next_states, vol=True).max(1)[0]
        max_new_qvalues.volatile = False

        updates[done_ids == 0] += self.discount * max_new_qvalues.data

        states_batch = np.stack(minibatch[:,0])
        qvalues = self.get_qvalues(states_batch)

        y = qvalues.data.clone()
        actions = torch.from_numpy(np.array(minibatch[:, 1], np.uint8))
        for i, action in enumerate(actions):
            y[i][action] = updates[i]
        y = Variable(y)


        self.optimizer.zero_grad()
        loss = self.criterion(qvalues, y)
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        # for state, action, reward, next_state, done in minibatch:
        #     update = reward
        #     if not done:
        #         max_new_qvalue = self.get_qvalues(next_state).max(0)[0].data[0]
        #         update += self.discount * max_new_qvalue
        #
        #     qvalues = self.get_qvalues(state)
        #
        #     y = qvalues.data.clone()
        #     y[0][action] = update
        #     y = Variable(y)
        #
        #     self.criterion(qvalues, y).backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        #
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     if self.epsilon < self.epsilon_min:
        #         self.epsilon = self.epsilon_min

    def update_memory(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))
        self.memory = self.memory[-self.max_memory_size:]
