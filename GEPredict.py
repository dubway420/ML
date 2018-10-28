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

    names = []
    commands = []
    for i in range(0, len(models)): 
        name, command = models[i]
        commands.append(command)
        names.append(name)
    
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

def trainingValidationSets(url, headings, validation_fraction, seed):  
    
    dataset = pandas.read_csv(url, names=headings)
    array = dataset.values
    
    len_m1 = len(array[0])-1
     
    # Split data into features and tags
    features = array[:, 0:len_m1]
    classes = array[:, len_m1]
    
    features_training, features_validation, classes_training, classes_validation = model_selection.train_test_split(features, classes, test_size=validation_fraction, random_state=seed)

    return (features_training, features_validation, classes_training, classes_validation)
                
seed = 7
scoring = 'accuracy'
validation_fraction = 0.20

# Load and arrange data into array
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
headings = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#======#======
models = classifiers()

features_training, features_validation, classes_training, classes_validation = trainingValidationSets(url, headings, validation_fraction, seed)

classifier_name, classifier_command = bestClassifierSelect(10, models, features_training, classes_training, seed, scoring)

print(classifier_name)