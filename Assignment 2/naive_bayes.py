#-------------------------------------------------------------------------
# AUTHOR: Anastasia Davis
# FILENAME: naive_bayes.py
# SPECIFICATION: This program trains a Gaussian Naïve-Bayes model on weather data and outputs the test predictions with a confidence of at least 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 39 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
db = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:
            db.append(row[1:])

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for row in db:
    transform = []
    for i in range(0, len(row) - 1):
        match row[i]:
            case 'Sunny' | 'Hot' | 'High' | 'Strong':
                transform.append(1)
            case 'Overcast' | 'Mild' | 'Normal' | 'Weak':
                transform.append(2)
            case 'Rain' | 'Cool':
                transform.append(3)
            case _:
                transform.append(-1)
    X.append(transform) 

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for row in db:
    match row[-1]:
        case 'Yes':
            Y.append(1)
        case 'No':
            Y.append(2)
        case _:
            Y.append(-1)

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
header = None
test_db = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i == 0:
            header = row
        else:
            test_db.append(row[0:-1])
            
X_test = []
for row in test_db:
    transform = []
    for i in range(1, len(row)):
        match row[i]:
            case 'Sunny' | 'Hot' | 'High' | 'Strong':
                transform.append(1)
            case 'Overcast' | 'Mild' | 'Normal' | 'Weak':
                transform.append(2)
            case 'Rain' | 'Cool':
                transform.append(3)
            case _:
                transform.append(-1)
    X_test.append(transform) 

#Printing the header os the solution
#--> add your Python code here
header.append("Confidence")
print('{:12} {:12} {:12} {:12} {:12} {:12} {:12}'.format(*header))

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i in range(len(X_test)):
    probability = clf.predict_proba([X_test[i]])[0]
    if (probability[0] >= 0.75):
        print('{:12} {:12} {:12} {:12} {:12} {:12} {:.2f}'.format(*test_db[i], "Yes", probability[0]))
    elif (probability[1] >= 0.75):
        print('{:12} {:12} {:12} {:12} {:12} {:12} {:.2f}'.format(*test_db[i], "No", probability[1]))