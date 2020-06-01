import random
import sys
from ICRSsimulator import *
import numpy as np


class Env:
    def __init__(self, config):
        # Create simulator object and load the image
        self.sim = ICRSsimulator(config.env_image)
        if self.sim.loadImage() == False:
            print("Error: could not load image")
            sys.exit(0)

        # Simulate classification of mining areas
        lower = np.array([50, 80, 70])
        upper = np.array([100, 115, 110])
        interestValue = 1  # Mark these areas as being of highest interest
        self.sim.classify('Mining', lower, upper, interestValue)

        # Simulate classification of forest areas
        lower = np.array([0, 49, 0])
        upper = np.array([90, 157, 138])
        interestValue = 0  # Mark these areas as being of no interest
        self.sim.classify('Forest', lower, upper, interestValue)

        # Simulate classification of water
        lower = np.array([40, 70, 47])
        upper = np.array([70, 100, 80])
        interestValue = 0  # Mark these areas as being of no interest
        self.sim.classify('Water', lower, upper, interestValue)

        # Number of rows and colums of the map at the finest scale of classification
        # Each (i,j) position in the map is a 1-D array of classification likelihoods
        # of length given by the number of classes
        self.totalRows = 268
        self.totalCols = 250
        self.sim.setMapSize(self.totalRows, self.totalCols)
        self.sim.createMap()

        self.num_actions = 4

        self.battery = 100
        self.visited = np.ones([self.totalRows, self.totalCols])
        self.cached_points = []

        self.start_row = 2
        self.start_col = 2

        self.row_position = self.start_row
        self.col_position = self.start_col

        # for clarity maybe set constants as reward weights up here?
        # if keeping track of exact rotation, a sharp turn could also have a penalty
        self.MINING_REWARD = 100
        self.FOREST_REWARD = 0
        self.WATER_REWARD = 30
        self.RETURN_REWARD = 100
        self.TIMESTEP_PENALTY = -1
        self.BATTERY_PENALTY = -1000

    def reset(self):
        # reset visited states
        self.visited = np.ones_like(self.visited)

        # reset battery to 100
        self.battery = 100

        # clear cached points however you would do that

        # randomize starting position
        self.start_row = random.randint(2,266)
        self.start_col = random.randint(2,248)

        self.row_position = self.start_row
        self.col_position = self.start_col

    def step(self, action):
        self.done = False
        # need to add controls for reaching the edge of the region
        # These are overly simplified discrete actions, will want to make this continuous at some point
        if action == 0 and self.row_position < (self.totalRows - self.sim.droneImgSize['rows'] - 1):  # Forward one grid
            next_row = self.row_position + 1
            next_col = self.col_position
        elif action == 1 and self.col_position < (self.totalCols - self.sim.droneImgSize['cols'] - 1):  # right one grid
            next_row = self.row_position
            next_col = self.col_position + 1
        elif action == 2 and self.row_position > self.sim.droneImgSize['rows']:  # back one grid
            next_row = self.row_position - 1
            next_col = self.col_position
        elif action == 3 and self.col_position > self.sim.droneImgSize['cols']:  # left one grid
            next_row = self.row_position
            next_col = self.col_position - 1
        else:
            next_row = self.row_position
            next_col = self.col_position

        self.sim.setDroneImgSize(2, 2)
        navMapSize = self.sim.setNavigationMap()

        classifiedImage = self.sim.getClassifiedDroneImageAt(next_row, next_col)
        print(classifiedImage.shape)

        self.row_position = next_row
        self.col_position = next_col

        self.battery_loss(action)
        print("Battery:", self.battery)

        reward = self.get_reward(classifiedImage)

        self.visited_position()

        # End if battery dies
        # To end if returns to start position after so many steps add:
        # or (self.row_position == self.start_row and self.col_position == self.start_col and self.battery < 80)
        # (right now I am just using battery level since it decreases per time step but we can make this more sophisticated)
        if self.battery <= 0 or (self.row_position == self.start_row and self.col_position == self.start_col and self.battery < 80):
            self.done = True
        print(classifiedImage)
        # print("Old Shape:", classifiedImage.shape)
        classifiedImage = self.flatten_state(classifiedImage)
        # print("New Shape:", classifiedImage.shape)
        print(classifiedImage)

        return classifiedImage, reward, self.done

    def get_reward(self, classifiedImage):

        battery_dead = False
        returned_to_start = False
        if self.done:
            if self.battery <= 0:
                battery_dead = True
            else:
                returned_to_start = True

        mining_prob = classifiedImage[2,2,0]
        forest_prob = classifiedImage[2,2,1]
        water_prob = classifiedImage[2,2,2]

        print(mining_prob, forest_prob, water_prob)

        reward = mining_prob*self.MINING_REWARD + forest_prob*self.FOREST_REWARD + water_prob*self.WATER_REWARD
        reward = reward*self.visited[self.row_position, self.col_position]
        reward += self.BATTERY_PENALTY*battery_dead + self.TIMESTEP_PENALTY + self.RETURN_REWARD*returned_to_start
        return reward

    def battery_loss(self, action):
        # can set this however we want the battery to deplete
        # auxiliary neural net to optimize battery function

        # battery depletes less when hovering
        if action == 4:
            self.battery -= .01
        else:
            self.battery -= .05

    def visited_position(self):
        # Two options: either count just the current, or count everything in it's field of vision
        self.visited[self.row_position, self.col_position] = 0

        for i in range(5):
            for j in range(5):
                self.visited[self.row_position + i, self.col_position + j] *= .9

    def plot_visited(self):
        plt.imshow(self.visited[:, :], cmap='gray', interpolation='none')
        plt.title("Drone Path")
        plt.show()

    def getClassifiedDroneImage(self):
        image = self.sim.getClassifiedDroneImageAt(self.row_position, self.col_position)
        image = self.flatten_state(image)
        return image

    def flatten_state(self, state):
        flat_state = state.reshape(25, 3)
        return flat_state

    '''
    def save_cached_point(self, row, col):
        cached_point = (row, col)
        self.cached_points.append(cached_point)

    def get_cached_point(self):
        # calculate distance from current position to all current cached points (separate function?) and return the closest one
        # I think it will be better if it can learn when to use cached points rather than hard coding,
        # but I'm not exactly sure how to set that up at this point
        # Consider comparing to just using GRU
        cached_point = self.cached_points[0]
        return cached_point
    '''





