import gym
import numpy as np
import gym.spaces
from scipy.spatial.distance import euclidean
from .utils_compiled import build_distance_map, check_finish_achievable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random as rand

from .tasks import BY_PIXEL_ACTIONS, BY_PIXEL_ACTION_DIFFS, TaskSet


class PathFindingByPixelWithDistanceMapEnv(gym.Env):
    def __init__(self):
        self.task_set = None
        self.cur_task = None
        self.observation_space = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.done_reward = None

        self.distance_map = None

        self.action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))
        self.cur_position_discrete = None
        self.goal_error = None

    def _configure(self,
                   tasks_dir='data/imported/paths',
                   maps_dir='data/imported/maps',
                   obstacle_punishment=1,
                   local_goal_reward=5,
                   done_reward=10,
                   greedy_distance_reward_weight=0.05,
                   absolute_distance_reward_weight=0.15,
                   target_on_border_reward=5):

        self.task_set = TaskSet(tasks_dir, maps_dir)
        self.task_ids = list(self.task_set.keys())
        self.cur_task_i = 0

        self.observation_space = gym.spaces.Box(0, 1,
                                                (3, 20, 20), np.float32)

        self.obstacle_punishment = abs(obstacle_punishment)
        self.local_goal_reward = local_goal_reward
        self.done_reward = done_reward

        self.greedy_distance_reward_weight = greedy_distance_reward_weight
        self.absolute_distance_reward_weight = absolute_distance_reward_weight

        self.target_on_border_reward = target_on_border_reward

    def __repr__(self):
        return self.__class__.__name__
    
    def render(self, ax):
        plt.cla()
        ax.set_facecolor('black')

        ax.set_xlim(0, 20)
        ax.set_ylim(20, 0)
        ticks = np.arange(20)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        for i, (x,y) in enumerate(self.obstacle_points_for_vis):
            if i == 0:
                ax.fill_between([x, x + 1], y, y + 1, color='white', label='wall')
            else:
                ax.fill_between([x, x + 1], y, y + 1, color='white')

        ax.fill_between([self.finish[1], self.finish[1] + 1], self.finish[0], self.finish[0] + 1,
                        color='green', label='finish')

        ax.scatter(self.cur_position_discrete[1] + 0.5, self.cur_position_discrete[0] + 0.5,
                   marker='*', s=500, label='position')
        ax.grid()
        ax.legend()
        plt.draw()
        plt.pause(0.1)



    def reset(self):
        self.cur_task = self.task_set[self.task_ids[self.cur_task_i]]
        self.cur_task_i += 1
        if self.cur_task_i >= len(self.task_ids):
            self.cur_task_i = 0

        if self.cur_task is not None:
            local_map = self.cur_task.local_map  # shortcut
            while True:
                self.start = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                              rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                self.finish = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                               rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                if local_map[self.start] == 0 \
                        and local_map[self.finish] == 0 \
                        and self.start != self.finish \
                        and check_finish_achievable(np.array(local_map, dtype=np.float),
                                                    np.array(self.start, dtype=np.int),
                                                    np.array(self.finish, dtype=np.int)):
                    break
            self.goal_map = np.zeros(self.cur_task.local_map.shape)
            self.goal_map[self.finish] = 1

        return self._init_state()

    def _init_state(self):
        local_map = np.array(self.cur_task.local_map, dtype=np.float)
        self.distance_map = build_distance_map(local_map,
                                               np.array(self.finish, dtype=np.int))

        m = self.cur_task.local_map
        self.obstacle_points_for_vis = [(x, y)
                                        for y in xrange(m.shape[0])
                                        for x in xrange(m.shape[1])
                                        if m[y, x] > 0]

        self.cur_position_discrete = self.start
        return self._get_state()

    def _get_state(self):
        walls = np.zeros(self.cur_task.local_map.shape)
        walls[self.cur_task.local_map == 1] = 1

        player = np.zeros(self.cur_task.local_map.shape)
        player[self.cur_position_discrete] = 1

        return np.stack((walls, player, self.goal_map))

        # state = np.zeros(self.cur_task.local_map.shape, dtype=np.float32)
        # state[self.cur_position_discrete] = 1
        # state[self.cur_task.local_map == 1] = -1
        # state[self.finish] = 5
        #
        # return state.reshape(1, *state.shape)

    def step(self, action):
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]

        done = np.allclose(new_position, self.finish)
        if done:
            reward = self.done_reward
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or self.cur_task.local_map[tuple(new_position)] > 0
            if invalid_step:
                reward = -self.obstacle_punishment
            else:
                local_target = self.finish
                cur_target_dist = euclidean(new_position, local_target)
                if cur_target_dist < 1:
                    reward = self.local_goal_reward
                    done = True
                else:
                    reward = self._get_usual_reward(self.cur_position_discrete, new_position)
                self.cur_position_discrete = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]

        observation = self._get_state()
        return observation, reward, done, None

    def _get_usual_reward(self, old_position, new_position):
        old_height = self.distance_map[tuple(old_position)]
        new_height = self.distance_map[tuple(new_position)]
        true_gain = old_height - new_height

        local_target = self.finish
        old_dist = euclidean(old_position, local_target)
        new_dist = euclidean(new_position, local_target)
        greedy_gain = old_dist - new_dist

        start_height = self.distance_map[tuple(self.start)]
        abs_gain = np.exp(-new_height / start_height)

        total_gain = sum(
            ((1 - self.greedy_distance_reward_weight - self.absolute_distance_reward_weight) * true_gain,
             self.greedy_distance_reward_weight * greedy_gain,
             self.absolute_distance_reward_weight * abs_gain))
        return total_gain - 0.1
