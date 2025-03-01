#-------------------------------------------------------------------------
# AUTHOR: Anastasia Davis
# FILENAME: decision_tree.py
# SPECIFICATION: This program takes the data from contact_lens.csv, transforms the data (preprocessing), and then creates a decision tree.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
for instance in db:
  row = []
  match instance[0]:
    case 'Young':
      row.append(1)
    case 'Prepresbyopic':
      row.append(2)
    case 'Presbyopic':
      row.append(3)
  match instance[1]:
    case 'Myope':
      row.append(1)
    case 'Hypermetrope':
      row.append(2)
  match instance[2]:
    case 'Yes':
      row.append(1)
    case 'No':
      row.append(2)
  match instance[3]:
    case 'Normal':
      row.append(1)
    case 'Reduced':
      row.append(2)
  X.append(row)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> add your Python code here
for instance in db:
  match instance[4]:
    case 'Yes':
      Y.append(1)
    case 'No':
      Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()