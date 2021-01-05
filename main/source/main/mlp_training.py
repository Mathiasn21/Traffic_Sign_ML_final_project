import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_loading import signs

x_train, y_train = signs.training_data()
x_train = x_train[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_train)  # Total images
# Flatten image values
x_train = x_train.reshape((n, -1))

# Create a scalar used for scaling the image values - By removing the mean and scaling to unit variance
scalar = StandardScaler()

# Computes mean and std - used later for scaling the dataset
scalar.fit(x_train)
# Scale the dataset by centering and scaling
x_train = scalar.transform(x_train)

# Build MLP classifier model 1 - With solver adam and learning rate .0001
mlp_adam_best = MLPClassifier(solver='adam',
                              hidden_layer_sizes=(150, 110, 43),
                              max_iter=1000,
                              learning_rate_init=.0001,
                              activation='relu')

# Fit model to training data
mlp_adam_best.fit(x_train, y_train.ravel())

# Dump model onto disk
joblib.dump(mlp_adam_best, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_adam_best.joblib')

# Build MLP classifier model 2 - With solver sgd and learning rate .01
mlp_sgd = MLPClassifier(solver='sgd',
                        hidden_layer_sizes=(150, 110, 43),
                        max_iter=1000,
                        learning_rate_init=.01,
                        activation='relu')

# Fit model to training data
mlp_sgd.fit(x_train, y_train.ravel())

# Dump model onto disk
joblib.dump(mlp_sgd, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_sgd.joblib')

# Build MLP classifier model 3 - With solver adam and learning rate .01
mlp_adam_worst = MLPClassifier(solver='adam',
                               hidden_layer_sizes=(150, 110, 43),
                               max_iter=1000,
                               learning_rate_init=.01,
                               activation='relu')

# Fit model to training data
mlp_adam_worst.fit(x_train, y_train.ravel())

# Dump model onto disk
joblib.dump(mlp_adam_worst, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_adam_worst.joblib')
