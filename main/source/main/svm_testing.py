import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from data_loading import signs

x_features, y_targets = signs.training_data_grayscale()
x_features = x_features[:, :, :, 0]  # Selects only the grey column for each x => Decreased training time and less data
n = len(x_features)
x_features = x_features.reshape((n, -1))

x_train, x_test, y_train, y_test = train_test_split(x_features, y_targets, test_size=0.2)

best_svc = SVC(C=2.1, gamma=0.00001, kernel='rbf', verbose=True)
best_svc.fit(x_train, y_train)
svc_predictions = best_svc.predict(x_test)


joblib.dump(best_svc, 'D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\svm_1.joblib')
# best_svc = joblib.load('D:\\group_projects\\Sign-machine-learning\\main\\source\\models\\svm_1.joblib')

print('Classification report: ', classification_report(y_test, svc_predictions), '\n')
