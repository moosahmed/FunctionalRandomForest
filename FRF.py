import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV  # cross_validation is deprecated
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score


def test_tree_bags(learning_groups, learning_data, testing_groups, testing_data, n_trees, rf_type='classification',
                   oob_score=False, max_features="auto", class_weight=None, feature_importance=False):
    """
    X,Y are determined in .fit()
    number of predictors(max_features) is by default set to: max_features=sqrt(n_features) only if unspecified by the user. if specified use that int

    dont know for sure how to include categorical vector

    surrogate? is it the same as criterion or sample weight in fit(). not sure.

    prior can be set to "balanced_subsample"
    """
    if rf_type == 'regression':
        tree_bag = RandomForestRegressor(n_estimators=n_trees, max_features=max_features, oob_score=oob_score)
    else:
        tree_bag = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, oob_score=oob_score,
                                          class_weight=class_weight)

    tree_bag.fit(learning_data, learning_groups)
    predicted_groups = tree_bag.predict(testing_data)
    predicted_scores = tree_bag.predict_log_proba(testing_data)
    overall_accuracy = accuracy_score(testing_groups, predicted_groups)

    return_list = []

    if oob_score:
        oob_pred = tree_bag.oob_score_
        return_list.append(oob_pred)

    if feature_importance:
        features_dict = {}
        for i in range(len(tree_bag.feature_importances_)):
            features_dict[learning_data[i]] = tree_bag.feature_importances_[i]
            # check that it is getting the correct headers from learning set
        return_list.append(features_dict)

    return predicted_groups, predicted_scores, overall_accuracy, *return_list
