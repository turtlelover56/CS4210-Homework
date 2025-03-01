#-------------------------------------------------------------------------
# AUTHOR: Anastasia Davis
# FILENAME: knn.py
# SPECIFICATION: This program calculates the error rate of training a 1NN model with cross validation.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 21 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
num_correct = 0

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for j in db:
        if (i != j):
            X.append(list(map(float, j[0:-1])))

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    Y = []
    for j in db:
        if (i != j):
            match j[-1]:
                case 'ham':
                    Y.append(1.0)
                case 'spam':
                    Y.append(2.0)
                case _:
                    Y.append(-1.0)

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = list(map(float, i[0:-1]))

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    match class_predicted:
        case 1.0:
            num_correct += 1 if i[-1] == 'ham' else 0
        case 2.0:
            num_correct += 1 if i[-1] == 'spam' else 0            
            

#Print the error rate
#--> add your Python code here
print(f"Error Rate: {1 - num_correct / len(db)}")





