'''
Created on 12 Oct 2018

@author: Huw
'''
import pandas
import math
from __builtin__ import int
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
# import pydotplus

def extractData(url, headings):
    dataset = pandas.read_csv(url, names=headings)
    return dataset.values

def featureCategories(array):
    categories_group = []
    
    for i in range(len(array[0])-1):
        categories_group.append(list(set(array[:, i])))
        
    return categories_group
    
def initialiseArray(feature_number):
    multi_dimensional_array = []
    for _ in range(feature_number): multi_dimensional_array.append([])
    return multi_dimensional_array

def sortedFeature(array, feature_number, feature_categories):
    
    sorted_feature = initialiseArray(len(feature_categories))
    
    for training_example in array:
         
        data_point = training_example[feature_number]
        
        for j in range(len(feature_categories)):
            category = feature_categories[j]
            if data_point == category: 
#                 print(training_example)
                sorted_feature[j].append(training_example)
     
#     print(sorted_feature)            
    return sorted_feature

def classFraction(list_items, class_item):
    
    no_items = float(len(list_items))
    
    if no_items > 0:
        
        number_yes = 0.0
        for item in list_items:
            if item[len(item)-1] == class_item: number_yes += 1.0
        
        fraction = number_yes/no_items    
        
    else: fraction = 1.0
        
    return fraction
        
def featureEntropy(sorted_feature, population):
    
    feature_entropy = 0.0
    
    for category_group_no in range(len(sorted_feature)):
        
        no_category_items = float(len(sorted_feature[category_group_no]))

        fraction_yes = classFraction(sorted_feature[category_group_no], "yes")
          
        fraction_no = 1.0 - fraction_yes
        
        if fraction_yes != 0.0: entropy_yes = fraction_yes*math.log(fraction_yes, 2)
        else: entropy_yes = 0.0
        
        if fraction_no != 0.0: entropy_no = fraction_no*math.log(fraction_no, 2)
        else: entropy_no = 0.0
        
        category_entropy = -(no_category_items/population)*(entropy_yes + entropy_no) 
        feature_entropy += category_entropy
        
    return feature_entropy   

def minEntropyFeature(array, categories):
    
    columns = len(array[0])

    sorted_features = []
    feature_entropies = []
    
#     print(array)
    
    #takes the categories from each feature
    for feature_number in range(columns-1):
        
        #Returns a list of unique categories within the feature
        feature_categories = categories[feature_number]
        
        sorted_feature = sortedFeature(array, feature_number, feature_categories)
        
        population = float(len(array))  
        
        feature_entropy = featureEntropy(sorted_feature, population)        
        
        sorted_features.append(sorted_feature) 
        feature_entropies.append(feature_entropy)   
        
    min_entropy = min(feature_entropies)
    min_index = feature_entropies.index(min_entropy)
    min_entropy_feature = sorted_features[min_index]
    
    return (min_entropy_feature, min_index) 

def classPopularFraction(yes_no, array):
    
    fractions = []
    for class_item in yes_no:
        fractions.append(classFraction(array, class_item))
    
    fraction = max(fractions)
    
    most_popular_class = yes_no[fractions.index(fraction)]
    
    return (fraction, most_popular_class)
        
def buildTree(array, categories, depth):
    
    yes_no = ["yes", "no"]

    fraction, most_popular_class = classPopularFraction(yes_no, array)
    class_and_fraction = (most_popular_class, fraction)
    
    #======
    if depth == 0 or fraction == 1.0:
#         print(class_and_fraction)
#         print(" ")
#         print("history: " + history)
#         print("=====")
        return class_and_fraction
    
    tree, branch = [], []
    
    sorted_feature, feature_index = minEntropyFeature(array, categories)
    
    tree.append(feature_index)
    
    for feature_no in range(len(sorted_feature)):
        feature = sorted_feature[feature_no]
        split_cat = categories[feature_index][feature_no]
        branch.append(buildTree(feature, categories, depth-1))
    
    tree.append(branch)
    return tree

def testData(categories, tree, x):
    i = 0
    feature = tree[0]
# print("splitting by: " + headings[feature])

    if (type(feature) == int):
        next_level = tree[1]
#     print("first next level: " + str(next_level))
#     print("====")
    
    while (type(feature) == int):
    
        feature_categories = categories[feature]
#     print("categories are: " + str(feature_categories))
    
        x_feature_value = x[feature]
#     print("test value has a category of: " + x_feature_value)
        feature_category_index = feature_categories.index(x_feature_value)
#     print("index of: " + str(feature_category_index))
    
        next_branch = next_level[feature_category_index]
        if isinstance(next_branch, (list,)): 
            feature = next_branch[0]
            next_level = next_branch[1]
        else: feature = next_branch
    
    return feature
    
url = "https://raw.githubusercontent.com/dubway420/ML/master/Tennis.csv"
headings = ['outlook', 'Temperature','Humidity', 'Wind', 'Play Tennis']

array = extractData(url, headings)

depth = 2

print("Producing category and tree arrays")

categories = featureCategories(array)
tree = buildTree(array, categories, depth)

print("Array production complete")
print(tree)
print(" ")

for x in array:

    print("=========")
    print("Testing data point:")
    print("correct answer: " + x[len(x)-1])
    print(" ")
    
    prediction, chance = testData(categories, tree, x)    
    print(prediction)

# TODO:
# Convert output to NL
# Classes determining function
# Test/output function


# print(isinstance(tree[1], (list,)))
 


        