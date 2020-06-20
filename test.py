from ddqn import DQNAgent
from env import Env as Drone
from configurationSimple import *
from configurationFull import *
from configurationTest import *
import matplotlib.pyplot as plt

config = ConfigTest()
test_env = Drone(config)
state_size = 75
action_size = test_env.num_actions
test = DQNAgent(state_size, action_size)
test.load('trained_weights.h5')

episode_rewards = []
episode_covered = []
for e in range(config.num_episodes):
    total_reward = 0
    state = test_env.reset()
    for time in range(config.max_steps):
        # env.render()
        action = test.act(state)
        action_taken, next_state, reward, done = test_env.step(action, time, config.max_steps)
        total_reward += reward
        state = next_state

    covered = test_env.calculate_covered()
    episode_rewards.append(total_reward)
    episode_covered.append(covered)
    print("episode: {}, reward: {}, percent covered: {}, start position: {},{}"
        .format('test', total_reward, covered, test_env.start_row, test_env.start_col))

    if e == config.num_episodes - 5:
        test_env.plot_visited('test_path1.jpg')
    if e == config.num_episodes - 4:
        test_env.plot_visited('test_path2.jpg')
    if e == config.num_episodes - 3:
        test_env.plot_visited('test_path3.jpg')
    if e == config.num_episodes - 2:
        test_env.plot_visited('test_path4.jpg')
    if e == config.num_episodes - 1:
        test_env.plot_visited('test_path5.jpg')

plt.plot(episode_rewards)
plt.ylabel('Episode reward')
plt.xlabel('Episode')
plt.savefig('test_reward.png')
plt.clf()

plt.plot(episode_covered)
plt.ylabel('Percent Covered')
plt.xlabel('Episode')
plt.savefig('test_coverage.png')
plt.clf()
