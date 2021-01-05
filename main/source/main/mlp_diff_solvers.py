import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Load previously saved models, with different MLP solvers
mlp_adam_best: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_adam_best.joblib')
mlp_sgd: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_sgd.joblib')
mlp_adam_worst: MLPClassifier = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\mlp_adam_worst.joblib')

# Plot various loss curves from models with different solvers and learning rate.
plt.plot(mlp_adam_best.loss_curve_, label='Adam - Learning rate 0.0001')
plt.plot(mlp_sgd.loss_curve_, label='SGD - Learning rate 0.01')
plt.plot(mlp_adam_worst.loss_curve_, label='Adam - Learning rate 0.01')
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
