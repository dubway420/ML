'''
Created on 12 Oct 2018

@author: Huw
'''
import pandas
import math

def extractData(url, headings):
    dataset = pandas.read_csv(url, names=headings)
    return dataset.values

def featureCategories(array, i):
    return list(set(array[:, i]))

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
                sorted_feature[j].append(training_example)
    
    return sorted_feature

def classFraction(list_items, class_item):
    
    no_items = float(len(list_items))
    
    number_yes = 0.0
    for item in list_items:
        if item[len(item)-1] == class_item: number_yes += 1.0
            
    return number_yes/no_items
        
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

def minEntropyFeature(array):
    
    columns = len(array[0])

    sorted_features = []
    feature_entropies = []
    
    print(array)
    
    #takes the categories from each feature
    for feature_number in range(columns-1):
        
        #Returns a list of unique categories within the feature
        feature_categories = featureCategories(array, feature_number)
        
        sorted_feature = sortedFeature(array, feature_number, feature_categories)
        
        population = float(len(array))  
        
        feature_entropy = featureEntropy(sorted_feature, population)        
        
        sorted_features.append(sorted_feature) 
        feature_entropies.append(feature_entropy)   
        
    min_entropy = min(feature_entropies)
    min_index = feature_entropies.index(min_entropy)
    min_entropy_feature = sorted_features[min_index]
    
    return (min_entropy_feature, min_index) 

def buildTree(array, depth):
    
    print("depth: " + str(depth))
    print(" ")
    
    fractions = [classFraction(array, "yes"), classFraction(array, "no")]
    fraction = max(fractions)
    yes_no = ["yes", "no"]

    most_popular_class = yes_no[fractions.index(fraction)]
    class_and_fraction = most_popular_class + ": " + str(round(fraction, 3))
    
    if depth == 0 or fraction == 1.0:
        return class_and_fraction
    
    tree = []
    
    sorted_feature, feature_index = minEntropyFeature(array)
    
    print("splitting by feature: " + str(feature_index))
    print(" ")
    
    tree.append(feature_index)
    
    for feature in sorted_feature:
        tree.append(buildTree(feature, depth-1))
    
    return tree
    

url = "C:\Users\Huw\Documents\Tennis.csv"
headings = ['outlook', 'Temperature','Humidity', 'Wind', 'Play Tennis']

array = extractData(url, headings)

depth = 2

print(buildTree(array, depth))


        