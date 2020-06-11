import tensorflow as tf
from env import *
from configurationSimple import *
from configurationFull import *
import argparse

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import collections
import matplotlib.pyplot as plt


def get_action(model, inputs):
    output = model(inputs)
    probs = tf.nn.softmax(output)

    # 4 is num_actions
    action = np.random.choice(5, p=probs.numpy()[0])

    return action



class PolicyEstimator_RNN:
    def __init__(self, sight_dim=2, batch_size = 1, train_length = 6, lr=0.0001, scope='RNN_model_policy'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, self.train_length, ((self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3) + 2], dtype=tf.float32, name='state')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=256)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=64)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(32)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2, initial_state=self.initial_state)

            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * 32])

            #Goes from [1, 192] to [1, 4], may need another fully connected layer
            if args.extra_layer:
                self.dense3 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=16)
                self.output = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=5)
            else:
                self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=5)

            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            self.picked_action_prob = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state})

    def update(self, states, target, action, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.state: states, self.action: action, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

class ValueEstimator_RNN:
    def __init__(self, sight_dim = 2, batch_size = 1, train_length = 6, lr=0.0001, scope='RNN_model_value'):
        self.batch_size = batch_size
        self.train_length = train_length
        self.sight_dim = sight_dim
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None, self.train_length, ((self.sight_dim * 2 + 1) * (self.sight_dim * 2 + 1) * 3) + 2],
                                        dtype=tf.float32, name='state')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=256)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=64)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(32)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.dense2, initial_state=self.initial_state)

            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.train_length * 32])

            # Goes from [1, 192] to [1, 4], may need another fully connected layer

            self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            self.value_estimate = tf.squeeze(self.output)

            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, states, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: states})

    def update(self, states, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: states, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def get_last_t_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t:]):
        states.append(transition.state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, ((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3) + 2])
    return states

def get_last_t_minus_one_states(t, episode):
    states = []
    for i, transition in enumerate(episode[-t + 1:]):
        states.append(transition.state)

    states.append(episode[-1].next_state)

    states = np.asarray(states)
    states = np.reshape(states, [1, t, ((args.sight_dim * 2 + 1) * (args.sight_dim * 2 + 1) * 3) + 2])
    return states


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=0.99):
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = []
    episode_lengths = []
    episode_covered = []
    most_common_actions = []
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()

        episode = []
        episode_rewards.append(0)
        episode_lengths.append(0)

        done = False
        reward = 0
        actions = []

        for t in range(50):

            # Take a step
            if t < args.seq_length:
                action = np.random.randint(0, 5)
            else:
                states = get_last_t_states(args.seq_length, episode)
                action_probs = estimator_policy.predict(state=states)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            if t == 0:
                last_action = 4
            else:
                last_action = actions[-1]
            action_taken, next_state, reward, ep_done = env.step(action, t, 50, last_action)

            actions.append(action)
            # Keep track of the transition
            episode.append(Transition(
                state=state, action=action_taken, reward=reward, next_state=next_state, done=done))

            # Update statistics
            episode_rewards[i_episode] += reward
            episode_lengths[i_episode] = t

            #UPDATE EVERY TIMESTEP AFTER seq_length TIMESTEPS
            if t >= args.seq_length:
                states = get_last_t_states(args.seq_length, episode)

                states1 = get_last_t_minus_one_states(args.seq_length, episode)
                value_next = estimator_value.predict(states1)

                td_target = reward + args.gamma * value_next
                td_error = td_target - estimator_value.predict(states)

                estimator_value.update(states, td_target)
                estimator_policy.update(states, td_error, action)

            if done:
                break

            state = next_state

        episode_covered.append(env.calculate_covered())
        data = collections.Counter(actions)
        most_common_actions.append(data.most_common(1))
        print("Episode {} finished. Reward: {}. Steps: {}. Most Common Action: {}. Percent Covered: {}".format(i_episode,
                                                                                          episode_rewards[i_episode],
                                                                                          episode_lengths[i_episode],
                                                                                          most_common_actions[
                                                                                              i_episode],
                                                                                          episode_covered[i_episode]))
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


parser = argparse.ArgumentParser(description='Drone A2C trainer')

parser.add_argument('--seq_length', default=6, type=int, help='number of stacked images')
parser.add_argument('--num_steps', default=1000, type=int, help='episode length')
parser.add_argument('--sight_dim', default=2, type=int, help='Drone sight distance')
parser.add_argument('--num_episodes', default=5000, type=int, help='number of episodes')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument('--extra_layer', default=False, help='extra hidden layer between recurrent and output')
args = parser.parse_args()

conf = Config()
environment = Env(conf)
policy_estimator = PolicyEstimator_RNN(args.sight_dim, lr=args.lr)
value_estimator = ValueEstimator_RNN(args.sight_dim, lr=args.lr)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    actor_critic(environment, policy_estimator, value_estimator, conf.num_episodes)

environment.plot_visited()

# TO DO:
# Currently, reward is based on only the current state. Try to import only the current state,
# and possible successors

# implement gru layer

# Using RNN, we can pass the agent single frames of the environment
# Store entire episodes, and randomly draw traces of 8ish steps from a random batch of episodes
# Keeps random sampling but also retains sequences
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc
# Another possible trick - sending the last half of the gradients back for a given trace


# NOTES
# Tends to favor one action heavily after it has success with mostly that action
# Meaning, it can only tell that action x is good, not why it's good
# RNN will probably fix this issue
