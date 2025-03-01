#-------------------------------------------------------------------------
# AUTHOR: Anastasia Davis
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program calculates the accuracy of training a decision tree off of sets of data with differing lengths.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 48 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining:
        #Ex. row = [Young,Myope,Yes,Normal,Yes]
        transform = []
        for i in range(0, len(row) - 1):
            match row[i]:
                case 'Young' | 'Myope' | 'No' | 'Reduced':
                    transform.append(1)
                case 'Prepresbyopic' | 'Hypermetrope' | 'Yes' | 'Normal':
                    transform.append(2)
                case 'Presbyopic':
                    transform.append(3)
                case _:
                    transform.append(-1)
        X.append(transform)
        
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for row in dbTraining:
        #Ex. row = [Young,Myope,Yes,Normal,Yes]
        match row[len(row) - 1]:
            case 'Yes':
                Y.append(1)
            case 'No':
                Y.append(2)
            case _:
                Y.append(-1)
    accuracy = []
    #Loop your training and test tasks 10 times here
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)

        total_correct = 0
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            transform = []
            for i in range(0, len(data) - 1):
                match data[i]:
                    case 'Young' | 'Myope' | 'No' | 'Reduced':
                        transform.append(1)
                    case 'Prepresbyopic' | 'Hypermetrope' | 'Yes' | 'Normal':
                        transform.append(2)
                    case 'Presbyopic':
                        transform.append(3)
                    case _:
                        transform.append(-1)
            class_predicted = clf.predict([transform])[0]
            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            match data[4]:
                case 'Yes':
                    total_correct += 1 if class_predicted == 1 else 0
                case 'No':
                    total_correct += 1 if class_predicted == 2 else 0
        accuracy.append(total_correct / len(dbTest))
    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = sum(accuracy) / len(accuracy)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {avg_accuracy}")