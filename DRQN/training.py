from env import *
from DRQN.config import *
from DRQN.agent import *


num_episodes = 1000
max_steps = 5000

conf = Config()
conf.env_image = '../sample12x12km2.jpg'
environment = Env(conf)

agent = Agent(conf, environment)

agent.train(max_steps, num_episodes)

environment.plot_visited()
