from DRQN.config import *

from env import *
config = Config
env = Env(config)

# generate a random policy from the 5 possible actions (left, right, up, down, don't move)
# we can set the number of steps/ terminal state based on how we set up the problem
actions = [1]
for i in range(5000):
    a = random.randint(0, 4)
    actions.append(a)

# follow an episode under this policy
total_reward = 0
done = False
i = 0
state = env.getClassifiedDroneImage()
while not done and i < 5000:
    a = actions[i]
    classifiedImage, reward, done = env.step(action=a)
    total_reward += reward
    print("Action:", a)
    print("Reward:", reward)

    print("==========================")
    i += 1

print("Total Episode Reward:", total_reward)
env.plot_visited()
