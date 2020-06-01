from env import *
from DRQN.config import *
from DRQN.agent import *

environment = Env()
num_episodes = 100
max_steps = 5000

conf = Config()
conf.env_image = 'sample12x12km2.jpg'

agent = Agent(conf, environment)

agent.train(max_steps, num_episodes)

environment.plot_visited()
