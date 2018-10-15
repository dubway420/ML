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


seed = 7
scoring = 'accuracy'

# Load and arrange data into array
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
headings = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=headings)
array = dataset.values
 
# Split data into features and tags
features = array[:, 0:4]
tags = array[:, 4]
 
validation_fraction = 0.20
features_training, features_validation, tags_training, tags_validation = model_selection.train_test_split(features, tags, test_size=validation_fraction, random_state=seed)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

names = []
commands = []
for i in range(0, len(models)): 
    name, command = models[i]
    commands.append(command)
    names.append(name)
    
# evaluate each model in turn
 
best_overall_result = 0
best_overall_model = ""
best_number_splits = 0

# Cycle through various levels of splits
for number_splits in range(2, 120):
      
    best_result = 0
    best_result_model = ""
      
    # Cycle through methods   
    for name, model in models:
          
        kfold = model_selection.KFold(n_splits=number_splits, random_state=seed)
        cv_results = model_selection.cross_val_score(model, features_training, tags_training, cv=kfold, scoring=scoring)
          
        # average of all k sets
        mean_result = np.mean(cv_results)
          
        if mean_result > best_result:
            best_result = mean_result
            best_result_model = name
  
    if best_result > best_overall_result:
        best_overall_result = best_result
        best_overall_model_name = best_result_model
        best_number_splits = number_splits
                   
print("the best result was for a k of " + str(best_number_splits) + " with an accuracy of " + str(round(best_overall_result, 4)) + " using the " + best_overall_model_name + " method")

best_overall_model = commands[names.index(best_overall_model_name)]
        
# # Make predictions on validation dataset
best_overall_model.fit(features_training, tags_training)
predictions = best_overall_model.predict(features_validation)


print(predictions)
# print(accuracy_score(tags_validation, predictions))
# print(confusion_matrix(tags_validation, predictions))
# print(classification_report(tags_validation, predictions))
    
