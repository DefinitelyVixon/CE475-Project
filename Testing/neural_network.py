from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

# Configurations
ecm = {'mae': make_scorer(mean_absolute_error),
       'r2': make_scorer(r2_score)}

predictors = ['x1', 'x3', 'x5']
pd.set_option("display.precision", 1)
np.set_printoptions(precision=1, suppress=True)

# Dataset
raw_dataset = pd.read_csv('../dataset.csv', sep=',', skipinitialspace=True)
dataset = raw_dataset.copy()

# Standardization
scaler = StandardScaler()

# train_X, train_y = pd.DataFrame(scaler.fit_transform(dataset[predictors]), columns=predictors), dataset['Y']
# predict_set = train_X.tail(20)
# train_X, train_y = train_X.drop(predict_set.index), train_y.drop(predict_set.index)
predict_set = dataset.tail(20)
dataset = dataset.drop(predict_set.index)
train_X, train_y = pd.DataFrame(scaler.fit_transform(dataset[predictors]), columns=predictors), dataset['Y']

params = {'hidden_layer_sizes': [(100*n,) for n in range(10)],
          'alpha': []}

# Neural Network
# for layer in range(100, 1000, 100):
#     regressor = MLPRegressor(random_state=1, max_iter=50000,
#                              solver='lbfgs',
#                              hidden_layer_sizes=(layer,),
#                              )
#     cv_score = cross_val_score(regressor, X=train_X, y=train_y, scoring=ecm['mae'], cv=5)
#     print(layer, cv_score.mean())
clf = GridSearchCV(MLPRegressor, params)
clf_score = clf.score(train_X, train_y)


def predict():
    # Train Estimator
    final_estimator = MLPRegressor(random_state=1, max_iter=10000, solver='lbfgs')
    final_estimator.fit(X=train_X, y=train_y)
    # Predict
    predictions = final_estimator.predict(predict_set)
    print(predictions)
