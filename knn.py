import pandas as pd
import numpy as np

def load_data(file_name):
    data = pd.read_csv(file_name)
    data = np.array(data)
    return data

# funtion for standardizing the data (zero-mean and standard deviation = 1)
def self_preprocessing(arg):
    arg_ = (np.array(arg,dtype=np.float64))
    arg_ = (arg_ - arg_.mean(axis=0)) / arg_.std(axis=0)
    return arg_

# A function class = myknn(X, test, k) that performs k-nearest neighbor (k-NN) clas-
# sication where X 2 Rnp (n number of objects and p number of attributes) is training data, test is
# testing data, and k is a user parameter.
def myknn(X, test, k):

    labels_train = np.transpose(X)[0]
    lables_test = np.transpose(test)[0]
    train_data = np.transpose(np.transpose(X)[1:])
    test_data = np.transpose(np.transpose(test)[1:])
    nearest_neighbours = k
    clusters = [i for i in range(len(set(labels_train)))]

    clusters_dict = {i:labels_train[i] for i in range(len(labels_train))}
    accuracy_list = []
    for i in range(len(test_data)):

        distances = np.linalg.norm(test_data[i] - train_data, axis=1)

        indexes = distances.argsort()[:k]

        labels_list = [clusters_dict[i] for i in indexes]

        voting_dict = {i:labels_list.count(i) for i in set(labels_list)}

        predicted_class = 0.0
        for i, j in voting_dict.items():

            if j == max(voting_dict.values()):
                predicted_class = i
        if k is 1:
            actual_class = labels_list[0]
        else:
            actual_class = labels_list[int(i)]
        if predicted_class == actual_class:
            accuracy_list.append(True)
        else:
            accuracy_list.append(False)



    accuracy = (accuracy_list.count(True)/len(accuracy_list)) * 100


    return accuracy

if __name__ == '__main__':

    data = load_data("your file name")
    standrardized_data = self_preprocessing(data)
    x = 5 # it means number of nearest neighbours. I have written 5 as an example
    # you can split the data based on the value of x you give. x means the number of rows you select from the numpy array
    train_data = standrardized_data[:x]
    test_data = standrardized_data[x:]

    # For Knn function, the data must contain all the numerical values. The first column of the data set must contain the Labels. The label values must be in the form of 0,1,2,3 etc.

    myknn(train_data,test_data,x)
