# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:21:39 2016

@author: Craig Jones
"""

# The idea here is to use a voting classifer built with
# a number of member classifiers.
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

import extractFeatures

if __name__=="__main__":
    print ("Extracting features from .csv files")
    (X_all, y, num_class, n_train, n_feat, n_feat2, ids, X_loc_all) = extractFeatures.getData()
    print ("Completed extracting features from .csv files")

    # Obtain raw values from lists. Here I only use numeric values from csv files.
    # This includes derived descriptive statistics on a few of these.
    X_numeric = X_all[:n_train, :n_feat]
    X_numeric_test = X_all[n_train:, :n_feat]

    # 10% crossfold validation
    print("Splitting data so that I can use 10% crossvalidation.")
    X_training, X_testing, y_training, y_testing = train_test_split(X_numeric, y, test_size=0.1, random_state=0)

    # Train the classifiers.
    # I'll use a range of classifiers and unify their output within a voting classifer.
    print("Creating member classifiers.")
    xgbModel = XGBClassifier(
     learning_rate =0.1,
     n_estimators=200,
     max_depth=5,
     min_child_weight=3,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     reg_alpha=0.005,
     nthread=5,
     scale_pos_weight=1)
    
    bagModel = BaggingClassifier(n_estimators=200)
    grdModel = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5)
    rftModel = RandomForestClassifier(n_estimators=200)
    extModel = ExtraTreesClassifier(n_estimators=200)
    adaModel = AdaBoostClassifier(n_estimators=200)

    print("Creating voting classifier.")
    votingClassifier = VotingClassifier(estimators=[('xgb', xgbModel), ('bag', bagModel), ('gdb', grdModel), ('rft', rftModel), ('ext', extModel), ('ada', adaModel)], voting='soft')

    print("Fitting data to voting classifier.")
    votingClassifier.fit(X_training, y_training)

    print("Making predictions.")
    #y_pred = votingClassifier.predict(X_testing)
    #testPredictions = [round(value) for value in y_pred]
    
    testPredictions = votingClassifier.predict(X_testing)

    # evaluate predictions
    print("10% cross validation score.")
    accuracy = accuracy_score(y_testing, testPredictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    print("Writing submission file.")
    # Convert predictions for column based output.
    testPredictions = np.zeros((X_numeric_test.shape[0], num_class), dtype=np.int)
    i = 0
    for prediction in testPredictions:
        testPredictions[i][prediction] = 1
        i = i + 1

    # Create data frame for output.
    sub = pd.DataFrame(data={'id':ids, 'predict_0':testPredictions[:, 0], 'predict_1':testPredictions[:, 1],
                         'predict_2':testPredictions[:, 2]}, columns=['id', 'predict_0', 'predict_1', 'predict_2'])

    # Write out output file.
    saveFileName = os.getcwd() + "/submission.csv"
    try:
        os.remove(saveFileName)
    except OSError:
        pass
    sub.to_csv(saveFileName, index=False)
    
    print("Finished.")
