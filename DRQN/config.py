class Config(object):

# There's stuff in here we don't need so after implementing the rest FIX THIS
    train_steps = 50000000
    batch_size = 64
    history_len = 4
    frame_skip = 4
    epsilon_start = 1.0
    epsilon_end = 0.02
    max_steps = 10000
    epsilon_decay_episodes = 1000000
    train_freq = 8
    update_freq = 10000
    train_start = 20000
    dir_save = "saved_session/"
    epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    random_start = 10
    test_step = 5000
    network_type = "dqn"

    gamma = 0.99
    learning_rate_minimum = 0.00025
    lr_method = "rmsprop"
    learning_rate = 0.00025
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 512
    min_history = 4
    states_to_update = 4

    env_image = 'sample12x12km2.jpg'
    image_size = 25
    num_classes = 3
    mem_size = 32
