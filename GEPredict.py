import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

def classifiers():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    return models

def bestClassifierSelect(number_splits, models, features_training, classes_training, seed, scoring):

    best_result = 0
    best_result_model = ""
      
    # Cycle through methods   
    for name, model in models:
          
        kfold = model_selection.KFold(n_splits=number_splits, random_state=seed)
        cv_results = model_selection.cross_val_score(model, features_training, classes_training, cv=kfold, scoring=scoring)
          
        # average of all k sets
        mean_result = np.mean(cv_results)
          
        if mean_result > best_result:
            best_result = mean_result
            best_result_model = name
  
    model = commands[names.index(name)]      
    
    return (name, model)  
                  


