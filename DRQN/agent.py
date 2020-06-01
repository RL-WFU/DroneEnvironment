import tqdm as tqdm
import collections
import matplotlib.pyplot as plt
from DRQN.replayMemory import *
from DRQN.network import *


class Agent:

    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.rewards = 0
        self.lens = 0
        self.epsilon = config.epsilon_start
        self.replay_memory = ReplayMemory(config)
        self.history = None
        self.net = DRQN(self.env.num_actions, config)
        self.net.build()
        self.net.add_summary(
            ["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward", "ep_num_game",
             "learning_rate"], ["ep_rewards", "ep_actions"])

        self.episode_i = 0

    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        np.save(self.config.dir_save+'step.npy', self.i)

    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.episode_i = np.load(self.config.dir_save+'step.npy')

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.rand(0, self.env.num_actions)
        else:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run(
                [self.net.q_action, self.net.state_output_c, self.net.state_output_h], {
                    self.net.state: [[state]],
                    self.net.c_state_train: self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })
            return a[0]

    def train(self, max_steps, num_episodes):
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = []
        episode_lengths = []
        most_common_actions = []
        self.total_loss, self.total_q = 0., 0.
        self.update_count = 0

        for self.episode_i in tqdm(range(num_episodes)):
            # Reset the environment and pick the first action
            self.env.reset()

            episode = []
            episode_rewards.append(0)
            episode_lengths.append(0)

            # Right now the state is just the image -- do we need to pass battery/ start position/ anything as the state
            state = self.env.getClassifiedDroneImage()
            actions = []

            self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

            for t in range(max_steps):
                # Choose an action and take a step in the env
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)

                # Keep track of action and transition
                actions.append(action)
                episode.append(Transition(
                    state=state, action=action, reward=reward, next_state=next_state, done=done))

                episode_rewards[self.episode_i] += reward
                episode_lengths[self.episode_i] = t

                # add to replay memory
                self.replay_memory.add(next_state, reward, action, done, t)

                # decrease epsilon
                if self.episode_i < self.config.epsilon_decay_episodes:
                    self.epsilon -= self.config.epsilon_decay

                # train the DRQN
                if self.episode_i % self.config.train_freq == 0 and self.episode_i > self.config.train_start:
                    states, action, reward, done = self.replay_memory.sample_batch()
                    q, loss = self.net.train_on_batch_target(states, action, reward, done, self.episode_i)
                    self.total_q += q
                    self.total_loss += loss
                    self.update_count += 1

                # update target network
                if self.episode_i % self.config.update_freq == 0:
                    self.net.update_target()

                if done:
                    self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
                    break

                state = next_state

            data = collections.Counter(actions)
            most_common_actions.append(data.most_common(1))
            print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}".format(self.episode_i,
                                                                                              episode_rewards[self.episode_i],
                                                                                              episode_lengths[self.episode_i],
                                                                                              most_common_actions[self.episode_i]))
        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.show()
        plt.clf()




