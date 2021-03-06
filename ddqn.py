import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
from env2 import Env as Drone
from configurationSimple import *
from configurationFull import *
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size+8, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
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
                #t = self.target_model.predict(next_state)[0]
                #target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    config = ConfigSimple()
    env = Drone(config)
    state_size = 75
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32

    episode_rewards = []
    episode_covered = []
    average_over = config.num_episodes / 100
    average_reward = []
    average_covered = []
    average_r = 0
    average_c = 0
    max_reward = 0

    for e in range(config.num_episodes):
        total_reward = 0
        episode_rewards.append(0)
        state = env.reset()
        t = 0
        for time in range(config.max_steps):
            # env.render()
            action = agent.act(state)
            action_taken, next_state, reward, done = env.step(action, time, config.max_steps)
            total_reward += reward

            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            t = time

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
            plt.plot(average_reward)
            plt.ylabel('Averaged Episode reward')
            plt.xlabel('Episode')
            plt.savefig('ddqn_average_reward.png')
            plt.clf()
            plt.plot(episode_rewards)
            plt.ylabel('Episode reward')
            plt.xlabel('Episode')
            plt.savefig('ddqn_reward.png')
            plt.clf()
            plt.plot(average_covered)
            plt.ylabel('Average Percent Covered')
            plt.xlabel('Episode')
            plt.savefig('ddqn_average_coverage.png')
            plt.clf()
            plt.plot(episode_covered)
            plt.ylabel('Percent Covered')
            plt.xlabel('Episode')
            plt.savefig('ddqn_coverage.png')
            plt.clf()
        if e == config.num_episodes - 6:
            env.plot_visited('ddqn_drone_path1.jpg')
        if e == config.num_episodes - 5:
            env.plot_visited('ddqn_drone_path2.jpg')
        if e == config.num_episodes - 4:
            env.plot_visited('ddqn_drone_path3.jpg')
        if e == config.num_episodes - 3:
            env.plot_visited('ddqn_drone_path4.jpg')
        if e == config.num_episodes - 2:
            env.plot_visited('ddqn_drone_path5.jpg')
        if e == config.num_episodes - 1:
            env.plot_visited('ddqn_drone_path6.jpg')

        if total_reward >= max_reward:
            max_reward = total_reward
            env.plot_visited('drone_max.jpg')
            agent.save('ddqn_trained_weights.h5')

        print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}, environment: {}"
              .format(e, config.num_episodes, total_reward, episode_covered[e], env.start_row, env.start_col, t, env.image_num))

    plt.plot(average_reward)
    plt.ylabel('Averaged Episode reward')
    plt.xlabel('Episode')
    plt.savefig('ddqn_average_reward.png')
    plt.clf()

    plt.plot(average_covered)
    plt.ylabel('Average Percent Covered')
    plt.xlabel('Episode')
    plt.savefig('ddqn_average_coverage.png')
    plt.clf()
