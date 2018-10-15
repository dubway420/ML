from scipy import rand
import math
import matplotlib.pyplot as pyplot
import pandas
from math import exp

# x_values = [17, 638, 739, 865,  1318, 1444, 1972, 1987, 2331, 2792]
# y_values = [68, 31,  467, 1765, 876,  92,   2176, 154,  2572, 14]
# labels =   [0,  0,   0,   0,    0,    1,    0,    1,    0,    1]

url = "C:\Users\Huw\Documents\iris_3.csv"
headings = ['sepal-length', 'sepal-width','class']
dataset = pandas.read_csv(url, names=headings)
array = dataset.values

x_values = array[:, 0]
y_values = array[:, 1]
labels = array[:, 2]

#print(x_values)

x0 = []
x1 = []

y0 = []
y1 = []

for j in range(len(labels)):
    
    x = x_values[j]
    y = y_values[j]
    label = labels[j]
    
    if label == 0:
        x0.append(x)
        y0.append(y)
    else:
        x1.append(x)
        y1.append(y)    
    

pyplot.scatter(x0, y0, c="red", label="0")
pyplot.scatter(x1, y1, c="green", label="1")

def y_hat(i, w, t):
    
    x = x_values[i]  
    exponent = -(w*x-t)
    
   
    y_hat = 1/(1 + math.exp(exponent))
    
    return y_hat

def numberOfErrors(w1, w2, t):
      
    threshold_y = []
      
    m = -(w1/w2)
    c = t/w2

      
    numberErrors = 0
      
    for i in range(0, len(x_values)):
          
        x = x_values[i]
        y = y_values[i]
        label = labels[i]
        
        y_t = m*x + c
        threshold_y.append(y_t)
        
        if y_t > y: y_hat = 1
        else: y_hat = 0
        
        if y_hat != label: 
            numberErrors +=1
        
#     pyplot.scatter(x_values, y_values)
#     pyplot.plot(x_values, threshold_y)
#     pyplot.show()    
    return numberErrors    

a = 0.1        
t = (rand()*5000.0)-2500.0
    
w = []
w.append((rand()*20)-10) 
w.append((rand()*20)-10) 
 
numberErrors = len(x_values)
cycles = 0
while numberErrors > 20:
    
    numberErrors = 0
          
    for i in range(0, len(array)):
        
        for j in range(0, len(w)):

            xj = array[i][j]
            y = labels[i]
            
            y_h = y_hat(i, w[j], t)
            error = y_h - y

            wj = w[j]
            
            w[j] = wj -(a*error*xj)
                   
            numberErrors += abs(error)

     
        t += a*error  
        
    cycles +=1
    
#     print(w)
#     print(t)
#     print ("cycle "+ str(cycles) + " complete")
#     print("errors found: " + str(numberErrors))
#     print ("---")
       
print("solution found on cycle " + str(cycles))   
print("---")
print("w vector: " + str(w))
print("t: " + str(t))

y_line = []
  
m = -(w[0]/w[1])
c = t/w[1]
    
for x in x_values:    
    y_val = m*x + c
    y_line.append(y_val)
      
pyplot.plot(x_values, y_line, c="black")
  
 
pyplot.legend()
pyplot.show()

    


    
