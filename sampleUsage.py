
import sys
from ICRSsimulator import *
import numpy as np
import matplotlib

# Create simulator object and load the image
sim = ICRSsimulator('env_images/image_d.png')
if sim.loadImage() == False:
	print("Error: could not load image")
	sys.exit(0)

lower = np.array([80, 90, 70])
upper = np.array([100, 115, 150])
interest_value = 1  # Mark these areas as being of highest interest
sim.classify('Mining', lower, upper, interest_value)

# Simulate classification of forest areas
lower = np.array([0, 49, 0])
upper = np.array([80, 157, 138])
interest_value = 0  # Mark these areas as being of no interest
sim.classify('Forest', lower, upper, interest_value)

# Simulate classification of water
lower = np.array([92, 100, 90])
upper = np.array([200, 190, 200])
interestValue = 0	# Mark these areas as being of no interest
sim.classify('Water', lower, upper, interestValue)

# Number of rows and colums of the map at the finest scale of classification
# Each (i,j) position in the map is a 1-D array of classification likelihoods
# of length given by the number of classes
rows = 200
cols = 200
sim.setMapSize(rows, cols)

# The map will contain 3 values per map element, simulating the
# likelihood of finding Mining, Forest, and Water, in that order
sim.createMap()

# Show the map with the classification results in separate images
sim.showMap()

# Set the size of the drone image in terms of number of elements in
# the map.
sim.setDroneImgSize(2, 2)

# Create the navigation map for the drone
navMapSize = sim.setNavigationMap()
print(navMapSize)

# Get sample drone image classifications
classifiedImage = sim.getClassifiedDroneImageAt(3,3)
print(classifiedImage.shape)
fig, axs = plt.subplots(1, len(classifiedImage.shape))
for k in range(0, len(classifiedImage.shape)):
	axs[k].imshow(classifiedImage[:,:,k], cmap = 'gray', interpolation = 'none')
plt.show()


classifiedImage = sim.getClassifiedDroneImageAt(navMapSize[0] - 1, navMapSize[1] - 1)
print(classifiedImage.shape)
fig, axs = plt.subplots(1, len(classifiedImage.shape))
for k in range(0, len(classifiedImage.shape)):
	axs[k].imshow(classifiedImage[:,:,k], cmap = 'gray', interpolation = 'none')
plt.show()

