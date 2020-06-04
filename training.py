from env import *
from config import *
from agent import *


num_episodes = 50000
max_steps = 1000

conf = Config()
conf.env_image = '../sample12x12km2.jpg'
environment = Env(conf)

agent = Agent(conf, environment)

agent.train(max_steps, num_episodes)

# environment.plot_visited()
