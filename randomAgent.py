import random

from env import *

env = Env()

# generate a random policy from the 5 possible actions (left, right, up, down, don't move)
# we can set the number of steps/ terminal state based on how we set up the problem
actions = [1]
for i in range(1000):
    a = random.randint(0, 4)
    actions.append(a)

print(actions)

# follow an episode under this policy
total_reward = 0
for a in actions:
    classifiedImage, next_row, next_col, reward = env.step(action=a)
    total_reward += reward
    print("Action:", a)
    print("New Position:", next_row, ",", next_col)
    print("Reward:", reward)

    print("==========================")

print("Total Episode Reward:", total_reward)
env.plot_visited()
