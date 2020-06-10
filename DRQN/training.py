from env import *
from DRQN.config import *
from DRQN.agent import *


num_episodes = 300
max_steps = 50

conf = Config()
environment = Env(conf)

agent = Agent(conf, environment)

agent.train(max_steps, num_episodes)

# environment.plot_visited()
