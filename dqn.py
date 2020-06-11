import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from env import Env as Drone
from configurationSimple import *
from configurationFull import *
EPISODES = 300


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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size+2, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Pass state (3x25), action, reward, next_state, ep_done

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            predicted = self.model.predict(next_state)
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(predicted[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    config = Config()
    env = Drone(config)
    state_size = 75 # FIXME: make dynamic
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)

    done = False
    batch_size = 32
    episode_rewards = []
    episode_covered = []

    for e in range(config.num_episodes):
        total_reward = 0
        actions = []
        episode_rewards.append(0)
        state = env.reset()
        #state = np.append(state, env.row_position)
        #state = np.append(state, env.col_position)
        #state = np.reshape(state, [1, (env.image_size * env.num_classes) + 2])

        for time in range(config.max_steps):
            action = agent.act(state)
            if time == 0:
                last_action = 4
            else:
                last_action = actions[-1]
            action_taken, next_state, reward, done = env.step(action, time, config.max_steps, last_action)
            total_reward += reward
            actions.append(action_taken)
            #next_state = np.append(next_state, env.row_position)
            #next_state = np.append(next_state, env.col_position)
            #next_state = np.reshape(next_state, [1, (env.image_size * env.num_classes) + 2])
            agent.memorize(state, action_taken, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        episode_rewards[e] = total_reward
        episode_covered.append(env.calculate_covered())
        if e > env.num_episodes - 10:
            env.plot_visited()
        print("episode: {}/{}, score: {}, e: {:.2}, percent covered: {}, start position: {},{}"
              .format(e, env.num_episodes, total_reward, agent.epsilon, episode_covered[e], env.start_row, env.start_col))

    plt.plot(episode_rewards)
    plt.ylabel('Episode reward')
    plt.xlabel('Episode')
    plt.show()
    plt.clf()

    plt.plot(episode_covered)
    plt.ylabel('Percent Covered')
    plt.xlabel('Episode')
    plt.show()
    plt.clf()
