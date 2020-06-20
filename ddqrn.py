# -*- coding: utf-8 -*-
import collections
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam

from env import Env as Drone
from configurationSimple import *

EPISODES = 500


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=[5, self.state_size+4], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(LSTM(64))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Pass state (3x25), action, reward, next_state, ep_done

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def act(self, states, t):
        if np.random.rand() <= self.epsilon or t < 5:
            return random.randrange(self.action_size)
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            target_next = self.model.predict(next_state)
            target_val = self.target_model.predict(next_state)
            if done:
                target[0][action] = reward
            else:
                a = np.argmax(target_next[0])
                target[0][action] = reward + self.gamma * target_val[0][a]
                # a = self.model.predict(next_state)[0]
                # t = self.target_model.predict(next_state)[0]
                # target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_last_t_states(self, t, episode):
        states = []
        for i, transition in enumerate(episode[-t:]):
            states.append(transition.state)

        states = np.asarray(states)
        states = np.reshape(states, [5, 79])
        states = np.expand_dims(states, axis=0)
        return states


if __name__ == "__main__":
    config = ConfigSimple()
    env = Drone(config)
    state_size = 75 # FIXME: make dynamic
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    episode_rewards = []
    episode_covered = []
    average_over = config.num_episodes / 10
    average_reward = []
    average_covered = []
    average_r = 0
    average_c = 0

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for e in range(config.num_episodes):
        episode = []
        total_reward = 0
        episode_rewards.append(0)
        state = env.reset()
        state = np.reshape(state, [1, state_size+4])
        for time in range(config.max_steps):
            states = np.zeros([1,5,79])
            if time > 5:
                states = agent.get_last_t_states(5, episode)
            # env.render()
            action = agent.act(states, time)

            action_taken, next_state, reward, done = env.step(action, time, config.max_steps)
            total_reward += reward
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size+4])

            episode.append(Transition(
                state=state, action=action_taken, reward=reward, next_state=next_state, done=done))

            state = next_state
            if time > 5:
                next_states = agent.get_last_t_states(5, episode)
                agent.memorize(states, action_taken, reward, next_states, done)

            if done:
                agent.update_target_model()
                break

            if len(agent.memory) > batch_size and time > 5:
                agent.replay(batch_size)

        covered = env.calculate_covered()
        episode_rewards[e] = total_reward
        episode_covered.append(covered)
        average_r += total_reward
        average_c += covered
        if e % average_over == 0:
            average_r /= average_over
            average_c /= average_over
            average_reward.append(average_r)
            average_covered.append(average_c)
            average_r = 0
            average_c = 0
        if e == config.num_episodes - 3:
            env.plot_visited('drone_path1_1.jpg')
        if e == config.num_episodes - 2:
            env.plot_visited('drone_path2_1.jpg')
        if e == config.num_episodes - 1:
            env.plot_visited('drone_path3_1.jpg')
        print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}"
              .format(e, config.num_episodes, total_reward, episode_covered[e], env.start_row, env.start_col))

    plt.plot(average_reward)
    plt.ylabel('Averaged Episode reward')
    plt.xlabel('Episode')
    plt.savefig('ddqn_reward_1.png')
    plt.clf()

    plt.plot(average_covered)
    plt.ylabel('Average Percent Covered')
    plt.xlabel('Episode')
    plt.savefig('ddqn_coverage_1.png')
    plt.clf()
