from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, make_scorer


ecm = make_scorer(mean_absolute_error)
ecm_name = 'mae'
cv_value = 5


# noinspection PyUnusedLocal
def single_linear_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    import pandas as pd
    estimator = None
    if type(x) == pd.Series:
        x = x.values.reshape(-1, 1)
    else:
        x = x[inputs].values.reshape(-1, 1)
    cv_scores = cross_val_score(estimator=LinearRegression(),
                                X=x,
                                y=y,
                                cv=cv_value,
                                scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=x,
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator = LinearRegression()
        estimator.fit(x, y)
    return {'model': 'single_linear_regression',
            'arg': '',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


# noinspection PyUnusedLocal
def multiple_linear_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    estimator = LinearRegression()
    cv_scores = cross_val_score(estimator=estimator,
                                X=x[inputs],
                                y=y,
                                cv=cv_value, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=x[inputs],
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator.fit(x, y)
    return {'model': 'multiple_linear_regression',
            'arg': '',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


def polynomial_regression(x, y, arg=None, inputs=None, keep_estimator=False):
    estimator = None
    poly_features = PolynomialFeatures(degree=arg, include_bias=False).fit_transform(x[inputs])
    cv_scores = cross_val_score(estimator=LinearRegression(),
                                X=poly_features,
                                y=y,
                                cv=cv_value, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=LinearRegression(),
                                       X=poly_features,
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator = LinearRegression()
        estimator.fit(poly_features, y)
    return {'model': 'polynomial_regression',
            'arg': f'order={arg}',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


def k_nearest_regression(x, y, arg=5, inputs=None, keep_estimator=False):
    estimator = None
    knn_regressor = KNeighborsRegressor(n_neighbors=arg)
    cv_scores = cross_val_score(estimator=knn_regressor,
                                X=x[inputs],
                                y=y,
                                cv=cv_value, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=knn_regressor,
                                       X=x[inputs],
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator = KNeighborsRegressor(n_neighbors=arg)
        estimator.fit(x[inputs], y)
    return {'model': 'k_nearest_regression',
            'arg': f'k={arg}',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


def decision_tree_regression(x, y, arg=5, inputs=None, keep_estimator=False):
    estimator = None
    tree_regressor = DecisionTreeRegressor(random_state=0, max_depth=arg)
    cv_scores = cross_val_score(estimator=tree_regressor,
                                X=x[inputs],
                                y=y,
                                cv=cv_value, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=tree_regressor,
                                       X=x[inputs],
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator = DecisionTreeRegressor(random_state=0, max_depth=arg)
        estimator.fit(x[inputs], y)
    return {'model': 'decision_tree_regression',
            'arg': f'max_depth={arg}',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


def random_forest_regression(x, y, arg=5, inputs=None, keep_estimator=False):
    estimator = None
    tree_regressor = RandomForestRegressor(random_state=0, max_depth=arg)
    cv_scores = cross_val_score(estimator=tree_regressor,
                                X=x[inputs],
                                y=y,
                                cv=cv_value, scoring=ecm)
    cv_predictions = cross_val_predict(estimator=tree_regressor,
                                       X=x[inputs],
                                       y=y,
                                       cv=cv_value)
    if keep_estimator:
        estimator = DecisionTreeRegressor(random_state=0, max_depth=arg)
        estimator.fit(x[inputs], y)
    return {'model': 'random_forest_regression',
            'arg': f'max_depth={arg}',
            'predictors': inputs,
            'predicted': cv_predictions,
            'error': cv_scores.sum() / len(cv_scores),
            'estimator': estimator}


def return_regression_functions():
    return {'single_linear_regression': single_linear_regression,
            'multiple_linear_regression': multiple_linear_regression,
            'polynomial_regression': polynomial_regression,
            'k_nearest_regression': k_nearest_regression,
            'decision_tree_regression': decision_tree_regression,
            'random_forest_regression': random_forest_regression}
