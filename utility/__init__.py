import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_absolute_error, make_scorer
from IPython.display import display
from sklearn.preprocessing import PolynomialFeatures
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
ecm = make_scorer(mean_absolute_error)
ecm_name = 'mae'
cv_value = 5


class ModelResult:
    def __init__(self, info_dict: dict):
        self.model = info_dict['model']
        self.arg = info_dict['arg']
        self.predictors = info_dict['predictors']
        self.predicted = info_dict['predicted']
        self.error = info_dict['error']
        if 'estimator' in info_dict.keys():
            self.estimator = info_dict['estimator']


class ModelResultsTable:
    def __init__(self):
        self.all_model_results = {}
        self.model_results_with_best_params = []

    def add_model(self, model_results: list):
        sorted_models = list(sorted(model_results, key=lambda item: item.error))
        best_performing_var = sorted_models[0]
        self.all_model_results[best_performing_var.model] = sorted_models
        self.model_results_with_best_params.append(best_performing_var)
        self.model_results_with_best_params = list(sorted(self.model_results_with_best_params,
                                                          key=lambda item: item.error))

    def error_table(self, transpose=False, model='best', top=1000, return_table=False, display_table=True):
        if model == 'all':
            error_table = []
            for model_result in self.all_model_results.values():
                error_table.extend([[result.error, result.model, result.predictors, result.arg]
                                    for result in model_result[:top]])
            error_table = pd.DataFrame(error_table, columns=[ecm_name, 'model', 'predictors', 'arg'])
            error_table.sort_values(by=[ecm_name], inplace=True, ignore_index=True)
        elif model == 'best':
            model_indexes = [model_result.model for model_result in self.model_results_with_best_params[:top]]
            error_table = pd.DataFrame([[result.arg, result.predictors, result.error]
                                        for result in self.model_results_with_best_params[:top]],
                                       columns=['args', 'predictors', ecm_name], index=model_indexes)
        else:
            arg_indexes = [model_result.arg for model_result in self.all_model_results[model][:top]]
            error_table = pd.DataFrame([[result.predictors, result.error]
                                        for result in self.all_model_results[model][:top]],
                                       columns=['predictors', ecm_name], index=arg_indexes)
        if transpose:
            error_table = error_table.transpose()
        if display_table:
            display(error_table)
        if return_table:
            return error_table


class ModelTree:
    def __init__(self, source_set, inputs, model_grid):
        self.inputs = inputs
        self.model_grid = model_grid
        self.source_set = source_set
        self.node_conditions = None
        self.leaf_models = None
        self.conditional_sets = None
        self.conditional_fits_table = []

    def fit_conditions(self, node_conditions):
        self.node_conditions = node_conditions
        self.leaf_models = [None] * len(node_conditions)
        self.conditional_sets = self.get_decision_sets()
        for cond_set in self.conditional_sets:
            fit_result_table = ModelResultsTable()
            for model in self.model_grid:
                fit_result_models = [ModelResult(model['model'](x=cond_set, y=cond_set['Y'], inputs=p, arg=a))
                                     for p in model['predictors']
                                     for a in model['arg']]
                fit_result_table.add_model(fit_result_models)
            self.conditional_fits_table.append(fit_result_table)

    def select_leaf_models(self, model_results: list):
        from models import return_regression_functions

        reg_funcs = return_regression_functions()

        for m_i in range(len(model_results)):
            m_r = model_results[m_i]
            if m_r.arg != '':
                arg = int(m_r.arg.split('=')[1])
            else:
                arg = None

            model_result = ModelResult(reg_funcs[m_r.model](x=self.conditional_sets[m_i][self.inputs],
                                                            y=self.conditional_sets[m_i]['Y'],
                                                            inputs=m_r.predictors, arg=arg, keep_estimator=True))
            self.leaf_models[m_i] = model_result

    def get_decision_sets(self, source_set=None):
        return_set = []
        if source_set is None:
            sample_set = self.source_set.copy()
        else:
            sample_set = source_set.copy()

        for condition in self.node_conditions:
            if condition == self.node_conditions[-1]:
                return_set.append(sample_set)
                break
            predictor, operator, threshold = condition[:2], condition[2], int(condition[3:])
            lt, gt = split_by_threshold(sample_set, predictor, threshold)
            if operator == '<':
                return_set.append(lt)
                sample_set = gt
            else:
                return_set.append(gt)
                sample_set = lt
        return return_set

    def predict(self, x):
        predictions = pd.DataFrame()
        p_sets = self.get_decision_sets(x)

        for c_model, p_set in zip(self.leaf_models, p_sets):
            estimator = c_model.estimator
            inputs = c_model.predictors
            test_set = p_set[inputs]

            if c_model.model == 'polynomial_regression':
                degree = int(c_model.arg.split('=')[-1])
                test_set = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(test_set)

            y_predicted = estimator.predict(test_set)
            predictions = pd.concat([predictions,
                                     pd.DataFrame({'Predictions': y_predicted}, index=p_set.index)])

        return predictions.sort_index()

    def plot_models(self, plot_titles, angle_3d=-96, cmap='r'):
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=(18, 5))
        cv_error = 0
        plot_i = 1
        decision_df = pd.DataFrame(columns=['Model', 'Predictors', 'Args', 'Error'])

        for cond_model, cond_set, plot_title in zip(self.leaf_models, self.conditional_sets, plot_titles):
            decision_df.loc[plot_title] = [cond_model.model, cond_model.predictors,
                                           cond_model.arg, cond_model.error]
            if len(cond_model.predictors) == 1:
                ax = fig.add_subplot(1, 3, plot_i)
                sns.scatterplot(ax=ax, x=cond_set[cond_model.predictors[0]], y=cond_set['Y'],
                                color='b', label='Actual Data')
                sns.scatterplot(ax=ax, x=cond_set[cond_model.predictors[0]], y=cond_model.predicted,
                                color='r', label='Predicted')
                ax.set_title(plot_title)
            elif len(cond_model.predictors) == 2:
                ax = fig.add_subplot(1, 3, plot_i, projection='3d')
                ax.view_init(elev=2., azim=angle_3d)
                ax.set_box_aspect(aspect=None, zoom=1.35)

                x_axis = cond_model.predictors[0]
                y_axis = cond_model.predictors[1]
                z_axis = 'Y'

                ax.scatter(cond_set[x_axis],
                           cond_set[y_axis],
                           cond_set[z_axis],
                           c=cond_set[z_axis])
                ax.plot_trisurf(cond_set[x_axis],
                                cond_set[y_axis],
                                cond_model.predicted,
                                linewidth=0, antialiased=True, alpha=0.3, cmap=cmap)
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_zlabel(z_axis)
                ax.set_title(plot_title)
            else:
                pass

            cv_error += cond_model.error
            plot_i += 1

        print('Total CV Score (MAE) =', round(cv_error, 2))
        display(decision_df)


def get_combinations(inputs, min_element=2):
    p_comb = []
    for L in range(min_element, len(inputs) + 1):
        p_comb.extend(combinations(inputs, L))
    return list(map(list, p_comb))


def subplot_for_five(data, predictors, title=None, highlight=None, reg_plot=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    point_colors = ['b', 'b', 'b', 'b', 'b']
    if highlight is not None:
        point_colors[highlight] = 'r'
    fig_, axes_ = plt.subplots(2, 3, figsize=(18, 10))
    fig_.suptitle(title, fontsize=20)

    axes_[1][2].set_visible(False)
    axes_[1][0].set_position([0.24, 0.125, 0.228, 0.343])
    axes_[1][1].set_position([0.55, 0.125, 0.228, 0.343])

    if reg_plot is not None:
        for i in range(len(predictors)):
            sns.scatterplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=data['Y'],
                            alpha=0.5, color=point_colors[i])
            sns.regplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=reg_plot[i].predicted,
                        color='r', scatter=False, label='Predicted Y')
    else:
        for i in range(len(predictors)):
            sns.scatterplot(ax=axes_[i // 3, i % 3], x=data[predictors[i]], y=data['Y'],
                            alpha=0.5, color=point_colors[i])


def split_by_threshold(ss, predictor, threshold):
    return ss[ss[predictor] < threshold], ss[ss[predictor] >= threshold]
