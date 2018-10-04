# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 17:37:24 2018

@author: Santillan
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Cargamos los dato
iris = load_iris()
datos = pd.DataFrame(iris.data, columns = iris.feature_names)
datos.head()
datos.columns
datos = datos.rename(index=str, columns={"sepal length (cm)": "sepal_l", 
                                 "sepal width (cm)": "sepal_w", 
                                 "petal length (cm)" :"petal_l",
                                 "petal width (cm)" : "petal_w"})
datos['Target'] = iris.target


# Separamos los datos en validacion y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    datos.drop('Target', axis = 1), # X
    datos.Target,  # y
    test_size = 0.35, 
    random_state = 42)


modelo = ['Tree', 'Bagging', 'RandForest', 'Naive_bayes', 'NN']
train = []
test = []

# Inicializamos los paramateros para gridsearch
grid_parametros = {
    'n_estimators': [50, 100, 150],
    'min_samples_split': [15, 20, 30],
    'class_weight': ['balanced', None]
}

# Inicializamos el modelo
rf = RandomForestClassifier()
rf_cv = GridSearchCV(rf, grid_parametros,
                    verbose=10, scoring='accuracy')

rf_cv.fit(X_train, y_train)
rf_cv.best_score_

np.sum(rf_cv.predict(X_test) == y_test)/len(y_test) 
 
confusion_matrix(y_test, rf_cv.predict(X_test))








