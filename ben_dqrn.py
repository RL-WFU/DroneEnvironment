# -*- coding: utf-8 -*-
import collections
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam

from env import Env as Drone
from DRQN.config import *

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

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=[5, self.state_size+2], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(LSTM(16))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Pass state (3x25), action, reward, next_state, ep_done

    def act(self, states, t):
        if np.random.rand() <= self.epsilon or t < 5:
            return random.randrange(self.action_size)
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
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
        states = np.reshape(states, [5, 75])
        states = np.expand_dims(states, axis=0)
        return states


if __name__ == "__main__":
    config = Config()
    env = Drone(config)
    state_size = 75 # FIXME: make dynamic
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for e in range(EPISODES):
        episode = []
        total_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(50):
            states = np.zeros([1,5,75])
            if time > 5:
                states = agent.get_last_t_states(5, episode)
            # env.render()
            action = agent.act(states, time)

            action_taken, next_state, reward, done = env.step(action)
            total_reward += reward
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            episode.append(Transition(
                state=state, action=action_taken, reward=reward, next_state=next_state, done=done))

            state = next_state
            if time > 5:
                next_states = agent.get_last_t_states(5, episode)
                agent.memorize(states, action_taken, reward, next_states, done)

            if len(agent.memory) > batch_size and time > 5:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
        print("episode: {}/{}, score: {}, e: {:.2}"
              .format(e, EPISODES, total_reward, agent.epsilon))