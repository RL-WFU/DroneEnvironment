from env import *
from config import *
from agent import *


num_episodes = 2000
max_steps = 5000

conf = Config()
environment = Env(conf)

agent = Agent(conf, environment)

agent.train(max_steps, num_episodes)

# environment.plot_visited()
