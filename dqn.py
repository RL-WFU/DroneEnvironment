"""Core classes."""

import numpy as np
from PIL import Image
import tensorflow as tf

class Sample:
    """Represents a reinforcement learning sample.
    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.
    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal

class Preprocessor:
    """Preprocessor base class.
    This is a suggested interface for the preprocessing steps.
    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.
    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.
    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.
        Should be called just before the action is selected.
        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.
        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.
        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.
        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.
        Should be called just before appending this to the replay memory.
        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.
        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.
        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.
        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.
        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.
        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process
        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.
        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.
        Parameters
        ----------
        reward: float
          Reward to process
        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.
        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass

class ReplayMemory:
    """Interface for replay memories.
    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, args):
        """Setup memory.
        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.
        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.memory_size = args.replay_memory_size
        self.history_length = args.num_frames
        self.actions = np.zeros(self.memory_size, dtype = np.int8)
        self.rewards = np.zeros(self.memory_size, dtype = np.int8)
        self.screens = np.zeros((self.memory_size, args.frame_height, args.frame_width), dtype = np.uint8)
        self.terminals = np.zeros(self.memory_size, dtype = np.bool)
        self.current = 0

    def append(self, state, action, reward, is_terminal):
        self.actions[self.current % self.memory_size] = action
        self.rewards[self.current % self.memory_size] = reward
        self.screens[self.current % self.memory_size] = state
        self.terminals[self.current % self.memory_size] = is_terminal
        # img = Image.fromarray(state, mode = 'L')
        # path = "./tmp/%05d-%s.png" % (self.current, is_terminal)
        # img.save(path)
        self.current += 1

    def get_state(self, index):
        state = self.screens[index - self.history_length + 1:index + 1, :, :]
        # history dimention last
        return np.transpose(state, (1, 2, 0))

    def sample(self, batch_size):
        samples = []
        indexes = []
        # ensure enough frames to sample
        assert self.current > self.history_length
        # -1 because still need next frame
        end = min(self.current, self.memory_size) - 1

        while len(indexes) < batch_size:
            index = np.random.randint(self.history_length - 1, end)
            # sampled state shouldn't contain episode end
            if self.terminals[index - self.history_length + 1: index + 1].any():
                continue
            indexes.append(index)

        for idx in indexes:
            new_sample = Sample(self.get_state(idx), self.actions[idx],
                self.rewards[idx], self.get_state(idx + 1), self.terminals[idx])
            samples.append(new_sample)
        return samples

    def clear(self):
        self.current = 0


"""Main DQN agent."""


class Qnetwork():
    def __init__(self, args, h_size, num_frames, num_actions, rnn_cell_1, myScope, rnn_cell_2=None):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.imageIn = tf.placeholder(shape=[None, 84, 84, num_frames], dtype=tf.float32)
        self.image_permute = tf.transpose(self.imageIn, perm=[0, 3, 1, 2])
        self.image_reshape = tf.reshape(self.image_permute, [-1, 84, 84, 1])
        self.image_reshape_recoverd = tf.squeeze(
            tf.gather(tf.reshape(self.image_reshape, [-1, num_frames, 84, 84, 1]), [0]), [0])
        self.summary_merged = tf.summary.merge(
            [tf.summary.image('image_reshape_recoverd', self.image_reshape_recoverd, max_outputs=num_frames)])
        # self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,1])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.image_reshape, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv1')
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv2')
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            activation_fn=tf.nn.relu, biases_initializer=None, scope=myScope + '_conv3')
        self.conv4 = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.conv3), h_size,
                                                       activation_fn=tf.nn.relu)

        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levels.
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.convFlat = tf.reshape(self.conv4, [self.batch_size, num_frames, h_size])
        self.state_in_1 = rnn_cell_1.zero_state(self.batch_size, tf.float32)

        if args.bidir:
            self.state_in_2 = rnn_cell_2.zero_state(self.batch_size, tf.float32)
            self.rnn_outputs_tuple, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_1, cell_bw=rnn_cell_2, inputs=self.convFlat, dtype=tf.float32,
                initial_state_fw=self.state_in_1, initial_state_bw=self.state_in_2, scope=myScope + '_rnn')
            # print "====== len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0] ", len(self.rnn_outputs_tuple), self.rnn_outputs_tuple[0].get_shape().as_list(), self.rnn_outputs_tuple[1].get_shape().as_list() # [None, 10, 512]
            # As we have Bi-LSTM, we have two output, which are not connected. So merge them
            self.rnn_outputs = tf.concat([self.rnn_outputs_tuple[0], self.rnn_outputs_tuple[1]], axis=2)
            # self.rnn_outputs = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(self.rnn_outputs_double), h_size, activation_fn=None)
            self.rnn_output_dim = h_size * 2
        else:
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.convFlat, cell=rnn_cell_1, dtype=tf.float32,
                initial_state=self.state_in_1, scope=myScope + '_rnn')
            # print "====== self.rnn_outputs ", self.rnn_outputs.get_shape().as_list() # [None, 10, 512]
            self.rnn_output_dim = h_size

        # attention machanism
        if not (args.a_t):
            self.rnn_last_output = tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1])
            self.rnn = tf.squeeze(self.rnn_last_output, [1])
        else:
            if args.global_a_t:
                self.rnn_outputs_before = tf.slice(self.rnn_outputs, [0, 0, 0], [-1, num_frames - 1, -1])
                self.attention_v = tf.reshape(tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1]),
                                              [-1, self.rnn_output_dim, 1])
                self.attention_va = tf.tanh(tf.matmul(self.rnn_outputs_before, self.attention_v))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs_before, self.attention_a), axis=1)
                self.rnn = tf.concat(
                    [self.rnn, tf.squeeze(tf.slice(self.rnn_outputs, [0, num_frames - 1, 0], [-1, 1, -1]), [1])],
                    axis=1)
            else:
                with tf.variable_scope(myScope + '_attention'):
                    self.attention_v = tf.get_variable(name='atten_v', shape=[self.rnn_output_dim, 1],
                                                       initializer=tf.contrib.layers.xavier_initializer())
                self.attention_va = tf.tanh(tf.map_fn(lambda x: tf.matmul(x, self.attention_v), self.rnn_outputs))
                self.attention_a = tf.nn.softmax(self.attention_va, dim=1)
                self.rnn = tf.reduce_sum(tf.multiply(self.rnn_outputs, self.attention_a), axis=1)
        # print "========== self.rnn ", self.rnn.get_shape().as_list() #[None, 1024]

        if args.net_mode == "duel":
            # The output from the recurrent player is then split into separate Value and Advantage streams
            self.ad_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu,
                                                               scope=myScope + '_fc_advantage_hidden')
            self.Advantage = tf.contrib.layers.fully_connected(self.ad_hidden, num_actions, activation_fn=None,
                                                               scope=myScope + '_fc_advantage')
            self.value_hidden = tf.contrib.layers.fully_connected(self.rnn, h_size, activation_fn=tf.nn.relu,
                                                                  scope=myScope + '_fc_value_hidden')
            self.Value = tf.contrib.layers.fully_connected(self.value_hidden, 1, activation_fn=None,
                                                           scope=myScope + '_fc_value')
            # Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        else:
            self.Qout = tf.contrib.layers.fully_connected(self.rnn, num_actions, activation_fn=None)

        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


def save_scalar(step, name, value, writer):
    """Save a scalar value to tensorboard.
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = float(value)
    summary_value.tag = name
    writer.add_summary(summary, step)