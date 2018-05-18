import numpy as np
import pandas as pd
import timeit
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


class CallForest(object):
    def __init__(self, rf_type, n_trees, n_predictors, oob_score, feature_importance, **kwargs):
        self.n_trees = n_trees
        self.n_predictors = n_predictors
        self.oob_score = oob_score
        self.feature_importance = feature_importance
        self.isclassifier = rf_type is 'classifier'
        if self.isclassifier:
            self.called_method = test_class_tree_bags
            self.class_weight = kwargs['class_weight']
        else:
            self.called_method = test_regress_tree_bags

    def train_method(self, all_data, X_train, y_train, X_test, y_test):
        if self.isclassifier:
            return self.called_method(all_data=all_data, training_data=X_train, training_groups=y_train,
                                      testing_data=X_test,
                                      testing_groups=y_test,
                                      n_trees=self.n_trees,
                                      n_predictors=self.n_predictors,
                                      oob_score=self.oob_score,
                                      feature_importance=self.feature_importance,
                                      class_weight=self.class_weight)
        else:
            return self.called_method(all_data=all_data, training_data=X_train, training_groups=y_train,
                                      testing_data=X_test,
                                      testing_groups=y_test,
                                      n_trees=self.n_trees,
                                      n_predictors=self.n_predictors,
                                      oob_score=self.oob_score,
                                      feature_importance=self.feature_importance)


def build_kfolds(df, data_cols, target_cols, forest_params, n_splits=10, n_repeats=3):
    """
    This takes a pandas df with headers and two lists denoting the data columns and the target columns.
    It also inherits user specified parameters for the rf model.

    This sets up the RepeatedStratifiedKFold func, and makes a test and train split based on the parameters provided.
    For each fold it runs the model then takes the rf models:
        a) individual level outputs:
            inserts them into new columns of the df at the specific indices used for testing that fold.
            e.g. all indices will have a value for predicted_class and predicted_score per repetition.
        b) overall summary outputs:
            Returns an average of the summary output. e.g. overall_accuracy
    """
    # Initializing outputs
    out_df = pd.DataFrame(index=range(len(df)))
    out_df['predicted_classes'] = np.empty((len(df), 0)).tolist()
    out_df['predicted_scores'] = np.empty((len(df), 0)).tolist()
    out_df['individual_accuracy'] = np.empty((len(df), 0)).tolist()
    oob_score_sum = 0
    all_accuracies = []
    all_prox_mat = []
    all_group_accuracies = []
    feature_importance_sum = {}

    X = df[data_cols]
    y = df[target_cols]
    rskf = RepeatedStratifiedKFold(n_splits, n_repeats, random_state=int(timeit.default_timer()))
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if forest_params.oob_score and forest_params.feature_importance:
            predicted_classes, predicted_scores, overall_accuracy, individual_accuracy, prox_mat, group_accuracy, \
            oob_score, feature_importance = forest_params.train_method(X, X_train, y_train, X_test, y_test)

            oob_score_sum += oob_score
            feature_importance_sum = {k: feature_importance_sum.get(k, 0) + feature_importance.get(k, 0) for k in
                                      set(feature_importance)}

        elif forest_params.oob_score and not forest_params.feature_importance:
            predicted_classes, predicted_scores, overall_accuracy, individual_accuracy, prox_mat, group_accuracy, \
            oob_score = forest_params.train_method(X, X_train, y_train, X_test, y_test)

            oob_score_sum += oob_score

        elif forest_params.feature_importance and not forest_params.oob_score:
            predicted_classes, predicted_scores, overall_accuracy, individual_accuracy, prox_mat, group_accuracy, \
            feature_importance = forest_params.train_method(X, X_train, y_train, X_test, y_test)

            feature_importance_sum = {k: feature_importance_sum.get(k, 0) + feature_importance.get(k, 0) for k in
                                      set(feature_importance)}

        else:
            predicted_classes, predicted_scores, overall_accuracy, individual_accuracy, prox_mat, group_accuracy = \
                forest_params.train_method(X, X_train, y_train, X_test, y_test)

        list_loc = 0
        for idx in test_index:
            out_df['predicted_classes'].iloc[idx].append(predicted_classes[list_loc])
            out_df['predicted_scores'].iloc[idx].append(predicted_scores[list_loc].max())
            out_df['individual_accuracy'].iloc[idx].append(individual_accuracy[0][list_loc])
            # TODO: Check this for multiple target_cols
            list_loc += 1

        all_accuracies.append(overall_accuracy)
        all_prox_mat.append(prox_mat)
        all_group_accuracies.append(group_accuracy)

    # Averaging across reps
    out_df['individual_accuracy'] = out_df['individual_accuracy'].apply(lambda x: sum(x) / len(x))

    if forest_params.oob_score and forest_params.feature_importance:
        allfolds_oob_score = oob_score_sum / (n_splits * n_repeats)
        allfolds_feature_importance = {k: v / (n_splits * n_repeats) for k, v in feature_importance_sum.items()}

        return out_df, all_accuracies, all_prox_mat, all_group_accuracies, allfolds_oob_score, \
               sorted(allfolds_feature_importance.items(), key=lambda x: x[1], reverse=True)

    elif forest_params.oob_score and not forest_params.feature_importance:
        allfolds_oob_score = oob_score_sum / (n_splits * n_repeats)

        return out_df, all_accuracies, all_prox_mat, all_group_accuracies, allfolds_oob_score

    elif forest_params.feature_importance and not forest_params.oob_score:
        allfolds_feature_importance = {k: v / (n_splits * n_repeats) for k, v in feature_importance_sum.items()}
        
        return out_df, all_accuracies, all_prox_mat, all_group_accuracies, \
               sorted(allfolds_feature_importance.items(), key=lambda x: x[1], reverse=True)

    else:
        return out_df, all_accuracies, all_prox_mat, all_group_accuracies


def get_feature_importance(model, training_data):
    """
    This takes a model and the training data used to train the model.
    The training data should be a pandas df with the feature names in the headers.

    Returns an ordered list of tuples. Ordered from most important feature to least.
    The tuple has the feature name and a decimal number denoting importance as a ratio.
    """
    features_dict = {}
    for i in range(len(model.feature_importances_)):
        features_dict[list(training_data)[i]] = model.feature_importances_[i]
    return features_dict


def classification_summary(y_true, y_pred, labels=None, target_names=None,
                           sample_weight=None):
    """This is edited from sklearn.metrics.classification_report
    Build a pandas df showing the main classification metrics
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    group_accuracies : pandas df
        Summary of the precision, recall, F1 score for each class.
        The df contains averages are a prevalence-weighted macro-average across
        classes (equivalent to :func:`precision_recall_fscore_support` with
        ``average='weighted'``).
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    Examples
    --------
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(group_accuracy(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
        class 0       0.50      1.00      0.67         1
        class 1       0.00      0.00      0.00         1
        class 2       1.00      0.67      0.80         3
    avg / total       0.70      0.60      0.61         5
    """
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        warnings.warn(
            "labels size, {0}, does not match size of target_names, {1}"
                .format(len(labels), len(target_names))
        )

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = [u'%s' % l for l in labels]

    target_names.append(last_line_heading)

    headers = ["precision", "recall", "f1-score", "support"]
    group_accuracies = pd.DataFrame(index=target_names, columns=headers)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    p = np.append(p, np.average(p, weights=s))
    r = np.append(r, np.average(r, weights=s))
    f1 = np.append(f1, np.average(f1, weights=s))
    s = np.append(s, np.sum(s))

    group_accuracies['precision'] = p
    group_accuracies['recall'] = r
    group_accuracies['f1-score'] = f1
    group_accuracies['support'] = s

    return group_accuracies


def proximity_matrix(model, data, normalize=True):
    # TODO: Add Documentation
    terminals = model.apply(data)
    n_trees = terminals.shape[1]
    a = terminals[:, 0]
    prox_mat = 1 * np.equal.outer(a, a)

    for i in range(1, n_trees):
        a = terminals[:, i]
        prox_mat += 1 * np.equal.outer(a, a)

    if normalize:
        prox_mat = prox_mat / n_trees

    return prox_mat


def test_class_tree_bags(all_data, training_data, training_groups, testing_data, testing_groups, n_trees=10,
                         n_predictors="auto", oob_score=False, feature_importance=False, class_weight=None):
    """
    This takes at minimum 4 inputs, your training group and data as well as testing group and data.

    Read RandomForestClassifier documentation for more info on n_trees(n_estimators), n_predictors(max_features),
    oob_score, and class_weight.
    If a dictionary of feature importance is required set feature_importance to True
    
    This function will build the Random Forest based on parameters provided.
    It will fit the RF on your training set.
    
    This function returns: 
    1) Predicted classes which are made using your testing data
    2) Overall accuracy of the RF in predicting your testing groups
    3) Predicted scores which are computed using the testing data
    4) Group accuracies for every group in your testing group
    
    Optionally, the function returns:
    5) Score of the training data set obtained using the out-of-bag estimate
    6) A dictionary of the features and how important they are to the RF model 
    """
    tree_bag = RandomForestClassifier(n_estimators=n_trees, max_features=n_predictors, oob_score=oob_score,
                                      class_weight=class_weight)

    tree_bag.fit(training_data, training_groups)
    predicted_classes = tree_bag.predict(testing_data)
    overall_accuracy = accuracy_score(testing_groups, predicted_classes)
    predicted_scores = tree_bag.predict_proba(testing_data)
    # TODO: figure out group accuracies. metrics.classification_report
    individual_accuracy = predicted_classes == testing_groups.T.values
    prox_mat = proximity_matrix(tree_bag, all_data)
    group_accuracy = classification_summary(testing_groups, predicted_classes)

    return_list = []

    if oob_score:
        oob_pred = tree_bag.oob_score_
        return_list.append(oob_pred)

    if feature_importance:
        return_list.append(get_feature_importance(tree_bag, training_data))

    return predicted_classes, predicted_scores, overall_accuracy, individual_accuracy, prox_mat, group_accuracy, \
           (*return_list)


def test_regress_tree_bags(all_data, training_data, training_groups, testing_data, testing_groups, n_trees=10,
                           n_predictors="auto", oob_score=False, feature_importance=False):
    """
    This takes at minimum 4 inputs, your training group and data as well as testing group and data.

    rf_type can be chosen to be either classification or regression
    Read RandomForestRegressor documentation for more info on n_trees(n_estimators), n_predictors(max_features),
    oob_score.
    If a dictionary of feature importance is required set feature_importance to True

    This function will build the appropriate Random Forest based on parameters provided.
    It will fit the RF on your training set.

    This function returns: 
    1) Predicted classes which are made using your testing data
    2) Mean absolute error of the model
    3) R-squared score of the model
    4) Individual differences for all subjects in the predicted class versus the test groups

    Optionally, the function returns:
    5) Score of the training data set obtained using the out-of-bag estimate
    6) A dictionary of the features and how important they are to the RF model
    """
    tree_bag = RandomForestRegressor(n_estimators=n_trees, max_features=n_predictors, oob_score=oob_score)

    tree_bag.fit(training_data, training_groups)
    predicted_classes = tree_bag.predict(testing_data)
    mae = mean_absolute_error(testing_groups, predicted_classes)
    r2 = r2_score(testing_groups, predicted_classes)
    individual_diff = predicted_classes - testing_groups.T.values  # TODO: Check this
    prox_mat = proximity_matrix(tree_bag, all_data)

    return_list = []

    if oob_score:
        oob_pred = tree_bag.oob_score_
        return_list.append(oob_pred)

    if feature_importance:
        return_list.append(get_feature_importance(tree_bag, training_data))

    return predicted_classes, mae, r2, individual_diff, prox_mat, (*return_list)


def interface(df, data_cols, target_cols, rf_type='classifier', n_trees=10, n_predictors="auto", oob_score=False,
              feature_importance=False, class_weight=None, n_kfold_splits=10, n_kfold_repeats=3):
    forest_params = CallForest(df=df, rf_type=rf_type, n_trees=n_trees, n_predictors=n_predictors, oob_score=oob_score,
                               feature_importance=feature_importance, class_weight=class_weight)

    if oob_score and feature_importance:
        out_df, all_accuracies, all_prox_mat, all_group_accuracies, allfolds_oob_score, allfolds_feature_importance = \
            build_kfolds(df, data_cols, target_cols, forest_params, n_splits=n_kfold_splits, n_repeats=n_kfold_repeats)
        print(out_df,
              'Accuracy:', all_accuracies,
              'Proximity_matrix:', all_prox_mat,
              'Group Accuracies:', all_group_accuracies,
              'Oob Score:', allfolds_oob_score,
              'Feature Importance:', allfolds_feature_importance)

    elif oob_score and not feature_importance:
        out_df, all_accuracies, all_prox_mat, all_group_accuracies, allfolds_oob_score = \
            build_kfolds(df, data_cols, target_cols, forest_params, n_splits=n_kfold_splits, n_repeats=n_kfold_repeats)
        print(out_df,
              'Accuracy:', all_accuracies,
              'Proximity_matrix:', all_prox_mat,
              'Group Accuracies:', all_group_accuracies,
              'Oob Score:', allfolds_oob_score)

    elif feature_importance and not oob_score:
        out_df, all_accuracies, all_prox_mat, all_group_accuracies, allfolds_feature_importance = \
            build_kfolds(df, data_cols, target_cols, forest_params, n_splits=n_kfold_splits, n_repeats=n_kfold_repeats)
        print(out_df,
              'Accuracy:', all_accuracies,
              'Proximity_matrix:', all_prox_mat,
              'Group Accuracies:', all_group_accuracies,
              'Feature Importance:', allfolds_feature_importance)

    else:
        out_df, all_accuracies, all_prox_mat, all_group_accuracies = build_kfolds(df, data_cols, target_cols,
                                                                                  forest_params,
                                                                                  n_splits=n_kfold_splits,
                                                                                  n_repeats=n_kfold_repeats)
        print(out_df,
              'Accuracy:', all_accuracies,
              'Proximity_matrix:', all_prox_mat,
              'Group Accuracies:', all_group_accuracies)


column_names = ['class_name', 'left_weight', 'left_distance', 'right_weight', 'right_distance']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                 header=None, names=column_names)
data_cols = ['left_weight', 'right_weight', 'left_distance', 'right_distance']
target_cols = ['class_name']

interface(df, data_cols, target_cols)
