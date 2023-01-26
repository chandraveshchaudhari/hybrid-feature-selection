from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses


# logistic regression
def get_logistic_regression(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100, precision_score(y_test, y_pred) * 100, recall_score(y_test,
                                                                                                     y_pred) * 100


# svm.SVC
def get_smv_svc(X_train, y_train, X_test, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100, precision_score(y_test, y_pred) * 100, recall_score(y_test,
                                                                                                     y_pred) * 100


# kNeighbors classifier
def get_k_neighbors_classifier(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    # Make predictions using the testing set
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred) * 100, precision_score(y_test, y_pred) * 100, recall_score(y_test,
                                                                                                     y_pred) * 100


# random forest
def get_random_forest(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = clf.predict(X_test)

    return accuracy_score(y_test, y_pred) * 100, precision_score(y_test, y_pred) * 100, recall_score(y_test,
                                                                                                     y_pred) * 100


# neural network
def get_neural_network(X_train, y_train, X_test, y_test):
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),

    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)
    result = model.predict(X_test)

    y_pred = []
    for i in result:
        if i[0] < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    return accuracy_score(y_test, y_pred) * 100, precision_score(y_test, y_pred) * 100, recall_score(y_test,
                                                                                                     y_pred) * 100
