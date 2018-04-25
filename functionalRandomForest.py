#!/usr/bin/env python3

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

if not vtype:
    vtype = 'validationPlusOOB'

if not treeweights:
    treeweights = 0   # TODO: Set this in default for the function it gets used in

if not treebag:
    treebag = 0  # TODO: Set this in default for the function it gets used in

# TODO: Check usage set defaults another way
surrogate = 'off'
prior = 'Empirical'
group1scores = 0
group2scores = 0
class_method = 'classification'


def make_group(testing_indexgroup=0, ngroup_substested=0, group_class=0, group_predict=0):
    # testing_indexgroup = # this comes from parsing args
    if testing_indexgroup > 0:
        ngroup_substested = max(testing_indexgroup.shape)
        group_class = np.zeros(ngroup_substested, 1)
        group_predict = np.zeros(ngroup_substested, 1)
    return testing_indexgroup, ngroup_substested, group_class, group_predict


testing_indexgroup1, ngroup1_substested, group1class, group1predict = make_group()
testing_indexgroup2, ngroup2_substested, group2class, group2predict = make_group()


def set_predictors(learning_data):
    arr = np.array([learning_data, 2])
    numpredictors = arr.shape
    # TODO: sqrt of shape and then round
    return numpredictors

numpredictors  = set_predictors(learning_data)

