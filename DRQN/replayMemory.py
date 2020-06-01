import numpy as np
import random
import os


class ReplayMemory:

    def __init__(self, config):
        super(ReplayMemory, self).__init__(config)
        self.config = config
        self.actions = np.empty(self.config.mem_size, dtype=np.int32)
        self.rewards = np.empty(self.config.mem_size, dtype=np.int32)
        # Screens are dtype=np.uint8 which saves massive amounts of memory, however the network expects state inputs
        # to be dtype=np.float32. Remember this every time you feed something into the network
        self.images = np.empty((self.config.mem_size, self.config.view_height, self.config.view_width), dtype=np.uint8)
        self.terminals = np.empty((self.config.mem_size,), dtype=np.float16)
        self.count = 0
        self.current = 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

        self.timesteps = np.empty(self.config.mem_size, dtype=np.int32)
        self.states = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1,
                                self.config.view_height, self.config.view_width), dtype=np.uint8)
        self.actions_out = np.empty(
            (self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.rewards_out = np.empty(
            (self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))
        self.terminals_out = np.empty(
            (self.config.batch_size, self.config.min_history + self.config.states_to_update + 1))

    def save(self):
        np.save(self.dir_save + "images.npy", self.screens)
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

    def load(self):
        self.screens = np.load(self.dir_save + "images.npy")
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")

    def add(self, image, reward, action, terminal, t):
        assert image.shape == (self.config.screen_height, self.config.screen_width)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.images[self.current] = image
        self.timesteps[self.current] = t
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size

    def getState(self, index):
        a = self.images[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a

    def get_scalars(self, index):
        t = self.terminals[index - (self.config.min_history + self.config.states_to_update + 1): index]
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, t, r

    def sample_batch(self):
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count - 1)
                if index >= self.current and index - self.config.min_history < self.current:
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    continue
                break
            self.states[len(indices)] = self.getState(index)
            self.actions_out[len(indices)], self.terminals_out[len(indices)], self.rewards_out[
                len(indices)] = self.get_scalars(index)
            indices.append(index)

        return self.states, self.actions_out, self.rewards_out, self.terminals_out