import random
import sys
from ICRSsimulator import *
import numpy as np


class Env:
    def __init__(self, config):
        # Create simulator object and load the image
        self.config = config
        self.sim = []

        for i in range(config.num_images):
            self.sim.append(ICRSsimulator(config.images[i]))
            if self.sim[i].loadImage() == False:
                print("Error: could not load image")
                sys.exit(0)
        '''
        self.sim2 = ICRSsimulator(config.image_2)
        if self.sim2.loadImage() == False:
            print("Error: could not load image")
            sys.exit(0)

        self.sim3 = ICRSsimulator(config.image_3)
        if self.sim3.loadImage() == False:
            print("Error: could not load image")
            sys.exit(0)
        '''

        self.image_num = random.randint(1, config.num_images)

        # Initialize map of size totalRows x totalCols from the loaded image
        self.totalRows = config.total_rows
        self.totalCols = config.total_cols

        # Set size of the image seen by the drone
        self.sight_distance = config.sight_distance
        self.image_size = (self.sight_distance * 2 + 1) * (self.sight_distance * 2 + 1)

        self.mining = []
        for i in range(config.num_images):
            self.set_simulation_map(self.sim[i])
            self.mining.append(self.total_mining(self.sim[i]))
        # self.set_simulation_map(self.sim2)
        # self.set_simulation_map(self.sim3)
        # self.mining2 = self.total_mining(self.sim2)
        # self.mining3 = self.total_mining(self.sim3)

        # Set drone position on map
        '''
        if self.image_num == 1:
            start = random.sample(self.mining, 1)
        elif self.image_num == 2:
            start = random.sample(self.mining2, 1)
        elif self.image_num == 3:
            start = random.sample(self.mining3, 1)
            
        '''
        start = random.sample(self.mining[self.image_num-1], 1)
        self.start_row = start[0][0]
        self.start_col = start[0][1]
        self.row_position = self.start_row
        self.col_position = self.start_col

        # Set array to keep track of locations already visited
        self.visited = np.ones([self.totalRows, self.totalCols])
        self.mining_seen = 0

        # Set number of possible actions and simulation classes
        self.num_actions = 5
        self.num_classes = 3

        # Set reward constants
        self.MINING_REWARD = 100
        self.FOREST_REWARD = 0
        self.WATER_REWARD = 30
        self.TIMESTEP_PENALTY = -1
        self.HOVER_PENALTY = -5
        self.COVERED_REWARD = 1500

    def reset(self):
        # reset visited states
        self.visited = np.ones_like(self.visited)
        self.mining_seen = 0

        self.image_num = random.randint(1, self.config.num_images)
        # print(self.image_num)

        # randomize starting position
        #self.start_col = 2
        #self.start_col = 2
        start = random.sample(self.mining[self.image_num-1],1)
        self.start_row = start[0][0]
        self.start_col = start[0][1]
        #self.start_row = random.randint(self.sight_distance, self.totalRows-self.sight_distance-1)
        #self.start_col = random.randint(self.sight_distance, self.totalCols-self.sight_distance-1)
        self.row_position = self.start_row
        self.col_position = self.start_col

        # get drone image and append current position to set new state
        state = self.get_classified_drone_image()
        state = np.reshape(state, [1, self.image_size * self.num_classes])
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.append(state, 1)
        state = np.reshape(state, [1, (self.image_size * self.num_classes) + 4])
        return state

    def step(self, action, time, max_steps):
        self.done = False

        next_row = self.row_position
        next_col = self.col_position

        if action == 0:
            if self.row_position < (self.totalRows - self.sight_distance - 1):  # Forward one grid
                next_row = self.row_position + 1
                next_col = self.col_position
            else:
                action = 4
        elif action == 1:
            if self.col_position < (self.totalCols - self.sight_distance - 1):  # right one grid
                next_row = self.row_position
                next_col = self.col_position + 1
            else:
                action = 4
        elif action == 2:
            if self.row_position > self.sight_distance + 1:  # back one grid
                next_row = self.row_position - 1
                next_col = self.col_position
            else:
                action = 4
        elif action == 3:
            if self.col_position > self.sight_distance + 1:  # left one grid
                next_row = self.row_position
                next_col = self.col_position - 1
            else:
                action = 4

        self.sim[self.image_num-1].setDroneImgSize(self.sight_distance, self.sight_distance)
        self.sim[self.image_num-1].setNavigationMap()

        classified_image = self.sim[self.image_num-1].getClassifiedDroneImageAt(next_row, next_col)

        self.row_position = next_row
        self.col_position = next_col

        if time > max_steps - 2:
            self.done = True

        reward = self.get_reward(classified_image, action)

        state = classified_image[:,:,:]
        state = self.flatten_state(state)
        state = np.append(state, self.visited[self.row_position+1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position+1])
        state = np.append(state, self.visited[self.row_position-1, self.col_position])
        state = np.append(state, self.visited[self.row_position, self.col_position+1])
        state = np.reshape(state, [1, (self.image_size * self.num_classes)+4])

        self.visited_position()

        return action, state, reward, self.done

    def get_reward(self, classified_image, action):

        if self.done:
            covered = self.calculate_covered()
            #mining = self.mining_seen/(covered*self.totalRows*self.totalCols)
            #reward = self.COVERED_REWARD*covered + mining*self.MINING_REWARD
            #print(self.mining_seen)
        else:
            #reward = 0
            covered = 0

        '''
        for i in range(4):
            for j in range(4):

                if classified_image[i, j, 0] * self.visited[self.row_position, self.col_position] > 0:
                    self.mining_seen += 1

        '''
        mining_prob = classified_image[self.sight_distance, self.sight_distance, 0]
        forest_prob = classified_image[self.sight_distance, self.sight_distance, 1]
        water_prob = classified_image[self.sight_distance, self.sight_distance, 2]

        hover = False
        if action == 4:
            hover = True

        reward = mining_prob*self.MINING_REWARD + forest_prob*self.FOREST_REWARD + water_prob*self.WATER_REWARD
        reward = reward*self.visited[self.row_position, self.col_position]
        reward += self.TIMESTEP_PENALTY + self.HOVER_PENALTY*hover + covered*self.COVERED_REWARD

        return reward

    def calculate_covered(self):
        covered = 0
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                if self.visited[i][j] < 1:
                    covered += 1

        percent_covered = covered / (self.totalCols * self.totalRows)

        return percent_covered

    def visited_position(self):
        self.visited[self.row_position, self.col_position] = 0

        for i in range(4):
            for j in range(4):
                self.visited[self.row_position + i - self.sight_distance, self.col_position + j - self.sight_distance] *= .7

    def plot_visited(self, fname):
        plt.imshow(self.visited[:, :], cmap='gray', interpolation='none')
        plt.title("Drone Path")
        plt.savefig(fname)
        plt.clf()

    def get_classified_drone_image(self):
        self.sim[self.image_num-1].setDroneImgSize(self.sight_distance, self.sight_distance)
        self.sim[self.image_num-1].setNavigationMap()
        image = self.sim[self.image_num-1].getClassifiedDroneImageAt(self.row_position, self.col_position)
        image = self.flatten_state(image)
        return image

    def flatten_state(self, state):
        flat_state = state.reshape(1, self.image_size * self.num_classes)
        return flat_state

    def set_simulation_map(self, sim):
        # Simulate classification of mining areas
        lower = np.array([50, 80, 70])
        upper = np.array([100, 115, 110])
        interest_value = 1  # Mark these areas as being of highest interest
        sim.classify('Mining', lower, upper, interest_value)

        # Simulate classification of forest areas
        lower = np.array([0, 49, 0])
        upper = np.array([90, 157, 138])
        interest_value = 0  # Mark these areas as being of no interest
        sim.classify('Forest', lower, upper, interest_value)

        # Simulate classification of water
        lower = np.array([40, 70, 47])
        upper = np.array([70, 100, 80])
        interest_value = 0  # Mark these areas as being of no interest
        sim.classify('Water', lower, upper, interest_value)

        sim.setMapSize(self.totalRows, self.totalCols)
        sim.createMap()

    def total_mining(self, sim):
        mining = sim.ICRSmap[:, :, 0]
        mining_positions = []
        for i in range(self.totalRows - self.sight_distance):
            for j in range(self.totalCols - self.sight_distance):
                if mining[i][j] > 0 and i >= self.sight_distance and j >= self.sight_distance:
                    mining_positions.append([i,j])

        return mining_positions

    def highest_mining(self, sim):
        mining = sim.ICRSmap[:, :, 0]
        highest_prob = 0
        start_position = [2,2]
        for i in range(self.totalRows - self.sight_distance):
            for j in range(self.totalCols - self.sight_distance):
                if mining[i][j] > highest_prob and i >= self.sight_distance and j >= self.sight_distance:
                    highest_prob = mining[i][j]
                    start_position = [i,j]

        return start_position






