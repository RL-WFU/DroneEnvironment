import collections
import matplotlib.pyplot as plt
from replayMemory import *
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
        self.net = DRQN(config, self.env.num_actions)
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
        if np.random.rand() < self.epsilon or self.episode_i < self.config.train_start:
            a = np.random.randint(0, self.env.num_actions)
            return a
        else:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run(
                [self.net.q_action, self.net.state_output_c, self.net.state_output_h], {
                    self.net.sequence_size: 1,
                    self.net.state: state,
                    self.net.c_state_train: self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })

            action = a[0]
            return action

    def train(self, max_steps, num_episodes):
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = []
        episode_lengths = []
        most_common_actions = []
        self.total_loss, self.total_q = 0., 0.
        self.update_count = 0

        for self.episode_i in range(num_episodes):

            # Reset the environment and pick the first action
            state = self.env.reset()

            episode = []
            episode_rewards.append(0)
            episode_lengths.append(0)

            actions = []

            self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

            for t in range(max_steps):

                state = np.reshape(state, [1, self.config.image_size* self.config.num_classes])
                action = self.get_action(state)

                action_taken, next_state, reward, ep_done = self.env.step(action)

                # Keep track of action and transition
                actions.append(action_taken)
                episode.append(Transition(
                    state=state, action=action_taken, reward=reward, next_state=next_state, done=ep_done))

                episode_rewards[self.episode_i] += reward
                episode_lengths[self.episode_i] = t

                # add to replay memory
                # look at next_state vs states size
                self.replay_memory.add(next_state, reward, action_taken, ep_done, t)

                # decrease epsilon
                if self.episode_i < self.config.epsilon_decay_episodes and (self.epsilon >= self.config.epsilon_decay + self.config.epsilon_end) and self.episode_i > self.config.train_start:
                    self.epsilon -= self.config.epsilon_decay

                # train the DRQN
                if self.episode_i % self.config.train_freq == 0 and self.episode_i > self.config.train_start:
                    states, action, reward, done = self.replay_memory.sample_batch()
                    q, loss = self.net.train_on_batch_target(states, action, reward, done, self.episode_i)
                    self.total_q += q
                    self.total_loss += loss
                    self.update_count += 1


                if ep_done:
                    self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
                    break

                state = next_state

            # update target network
            if self.episode_i % self.config.update_freq == 0:
                self.net.update_target()

            covered = self.env.calculate_covered()
            data = collections.Counter(actions)
            most_common_actions.append(data.most_common(1))
            print("Episode {} finished. Reward: {}. Steps: {}. Covered: {}. Most Common Action: {}".format(self.episode_i,
                                                                                              episode_rewards[self.episode_i],
                                                                                              episode_lengths[self.episode_i],
                                                                                              covered,
                                                                                              most_common_actions[self.episode_i]))

            if self.episode_i % 50 == 0:
                self.env.plot_visited()

                plt.plot(episode_rewards)
                plt.ylabel('Episode reward')
                plt.xlabel('Episode')
                plt.show()
                plt.clf()

        plt.plot(episode_rewards)
        plt.ylabel('Episode reward')
        plt.xlabel('Episode')
        plt.show()
        plt.clf()
                





