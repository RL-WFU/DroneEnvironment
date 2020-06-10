class Config(object):

    batch_size = 1
    epsilon_start = 1.0
    epsilon_end = 0.02
    max_steps = 10000
    epsilon_decay_episodes = 10000
    train_freq = 4
    update_freq = 75
    train_start = 10
    dir_save = "saved_session/"
    epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
    test_step = 5000
    network_type = "dqn"

    gamma = 0.99
    learning_rate_minimum = 0.0001
    lr_method = "rmsprop"
    learning_rate = 0.0001
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 32
    min_history = 10
    states_to_update = 1

    env_image = 'simpleTest.jpg'
    image_size = 25
    num_classes = 3
    mem_size = 32
