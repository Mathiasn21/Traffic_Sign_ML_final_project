import concurrent.futures
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loading import signs

# Load training data in grayscale and corresponding labels
x_train, y_train = signs.training_data()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)  # Total images
# Flatten image values
x_train = x_train.reshape((n, -1))

# Load test data in grayscale and corresponding labels
x_test, y_test = signs.test_data()
x_test = x_test[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_test)  # Total images
# Flatten image values
x_test = x_test.reshape((n, -1))

# Create a scalar used for scaling the image values
scalar = StandardScaler()

# Computes mean and std - used later for scaling the training dataset
scalar.fit(x_train)
# Scale the training dataset by centering and scaling
x_train = scalar.transform(x_train)

# Computes mean and std - used later for scaling the test dataset
scalar.fit(x_test)
# Scale the test dataset by centering and scaling
x_test = scalar.transform(x_test)

# Array for storing results from training each MLP model
results_3D = []
test_size = len(y_test)  # Size of test data


def execute_mlp(args):
    # Retrieve config from arguments
    hidden_layer = args.get("hidden_layer")
    lay_size = args.get("lay_size")
    x_nodes = args.get("x_nodes")

    time = process_time()

    # Build MLP classifier model
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=1000)
    mlp.fit(x_train, y_train.ravel())
    predictions = mlp.predict(x_test)
    time = process_time() - time  # Calculate end time after model being fitted and executed predict on test data

    hits = 0  # Number of correct classifications
    index = 0  # Current prediction index
    for num in y_test:
        # If prediction matches ground truth, then increment hits
        if num == predictions[index]:
            hits += 1
        index += 1

    avg = hits / test_size  # Calculate average of hits - Accuracy rating

    # Append results from this MLP configuration onto final result array
    results_3D.append([lay_size, x_nodes, avg, time])


# Specify configuration to generate thread args
tot_layers = 10
tot_nodes = 200

# List - Holds thread arguments
arg_list = list()

# Generate work arguments with varying neurons and layers.
for y in range(1, tot_nodes, 5):
    layers = []
    for x in range(0, tot_layers, 1):
        layers.append(y)
        arg_list.append({"hidden_layer": layers.copy(), "lay_size": x + 1, "x_nodes": y})

# Map work to threads, where work is to execute a method with arguments present in arg_arr
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    executor.map(execute_mlp, arg_list)

# Create 3D scatter plot figure with heat map for plotting the results
fig = plt.figure()
ax: Axes3D = fig.gca(projection='3d')
# Convert to numpy array for easier handling
results_3D = np.array(results_3D)

normal__time_res = (results_3D[:, 3] / results_3D[:, 3].max(axis=0)) * 100
img = ax.scatter(results_3D[:, 0], results_3D[:, 1], results_3D[:, 2],
                 c=normal__time_res, cmap=plt.hot())
color_bar = fig.colorbar(img)

# Set information labels on axes
ax.set_xlabel('Layers')
ax.set_ylabel('Neurons')
ax.set_zlabel('Accuracy')

# Set information label on colorbar
color_bar.set_label('Time')

# Render figure
plt.show()
