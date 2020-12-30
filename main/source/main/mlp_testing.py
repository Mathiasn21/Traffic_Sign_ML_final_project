import concurrent.futures
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loading import signs

x_train, y_train = signs.training_data_grayscale()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)
x_train = x_train.reshape((n, -1))

x_test, y_test = signs.test_data_greyscale()
x_test = x_test[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_test)
x_test = x_test.reshape((n, -1))

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)

scalar.fit(x_test)
x_test = scalar.transform(x_test)

tot_layers = 10
tot_nodes = 200

best_avg = 0
best_nodes_pr_layer = []

results_3D = []
test_size = len(y_test)


def execute_mlp(args):
    global best_avg
    hidden_layer = args.get("hidden_layer")
    lay_size = args.get("lay_size")
    x_nodes = args.get("x_nodes")

    time = process_time()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=1000)
    mlp.fit(x_train, y_train.ravel())
    predictions = mlp.predict(x_test)
    time = process_time() - time

    hits = 0
    index = 0
    for num in y_test:
        if num == predictions[index]:
            hits += 1
        index += 1

    avg = hits / test_size
    if best_avg < avg:
        best_avg = avg

    results_3D.append([lay_size, x_nodes, avg, time])


# Generate thread arguments
arr = list()
for y in range(1, tot_nodes, 5):
    layers = []
    for x in range(0, tot_layers, 1):
        layers.append(y)
        arr.append({"hidden_layer": layers.copy(), "lay_size": x + 1, "x_nodes": y})

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    executor.map(execute_mlp, arr)

fig = plt.figure()
ax: Axes3D = fig.gca(projection='3d')
results_3D = np.array(results_3D)

normal__time_res = (results_3D[:, 3] / results_3D[:, 3].max(axis=0)) * 100
img = ax.scatter(results_3D[:, 0], results_3D[:, 1], results_3D[:, 2],
                 c=normal__time_res, cmap=plt.hot())
color_bar = fig.colorbar(img)
ax.set_xlabel('Layers')
ax.set_ylabel('Neurons')
ax.set_zlabel('Accuracy')
color_bar.set_label('Time')
plt.show()
