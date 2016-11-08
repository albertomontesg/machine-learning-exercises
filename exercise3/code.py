##########################
# ETH ML 2016 
# Week 6 - Series 5
# SKELETON FOR MODEL SELECTION (Problem #3) 
# AUTHORS: ANDREAS HESS, HADI DANESHMAND
######################

######################
#  LIBRARIES 
######################
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.linear_model import Ridge

#######################
#   A FUNCTION TO READ A DATASET FROM LIBSVM
#######################
def readfile(filename, n,d):
    y = np.zeros(n) # targets
    X = np.zeros((n,d)) # input matrix each row is a sample data point
    li = 0 
    with open(filename, "rb") as f:
        for line in f:
           if li>=n : 
             break;
           parts = line.split()
           y[li] = float(parts[0])
           for i in range(len(parts)): 
                if i >0 and parts[i] != '\n': 
                    fparts = parts[i].split(":")
                    X[li,int(fparts[0])-1] = float(fparts[1])
           li = li +1
    return (y,X)





#################################
# NOTATIONS: 
# n: number of data points 
# d: dimensionality 
# X: the (d times n) data matrix 
# y: vector of outputs (labels) with length n
# w: the parameter vector with length d 
# lambd: the regularizer constant 
##################################

#######################
#   LOSS FUNCTION FOR RIDGE REGRESSION 
#######################
def loss(X,y,w,lambd): 
    n, d = X.shape
    res = np.dot(X,w) - y 
    loss = np.dot(np.transpose(res),res)/n + lambd*np.dot(np.transpose(w),w)
    return loss # returns the error 

#######################
#   RIDGE REGRESSION ESTIMATION 
#######################
def ridge_regression(X,y,lambd):
   n, d = X.shape
   A = np.linalg.inv(np.add(np.dot(np.transpose(X),X),0.5*n*lambd*np.eye(d)))
   D = np.dot(A,np.transpose(X))
   w = np.dot(D,y)
   return w # returns the parameter vector w





########################
# a. TEST ERROR VS. VALIDATION ERROR
########################

#######################
#   ONE LEAVE OUT CROSS VALIDATION (PLEASE FILL IN THE BLANKS)
#######################
def leave_one_out_cross(X,y,lambd): 
    n, d = X.shape
    inds = np.linspace(0,n-1,n) # the array list {1,...,n}
    error = 0 
    for i in range(n):
        train_inds = [element for j, element in enumerate(inds) if j not in {i} ] # excludes the index i from 
the list inds
        test_inds = i      
        w = ridge_regression(...,...,lambd) # PART 1. 
        error = error + pow(np.dot(X[...,:],w)-y[...],2) # PART 2.
    return error/float(n) # returns the cross validation error

# extracting the dataset

n = 32561; 
d = 123; 
filename = "a9a"
y, X_init = readfile(filename,n,d)
X = np.ones((n,d+1))
X[:,1:d] = X_init[:,1:d]
np.random.seed(1)

# spliting the training and test set

testsi = 1000 # size of test set
rperm = np.random.permutation(n)
X = X[rperm,:] # shuffling the data points
y = y[rperm]
Xtrain = X[1:n-testsi,:] # training set
ytrain = y[1:n-testsi] 
Xtest = X[n-testsi+1:n,:]  # test set
ytest = y[n-testsi+1:n]

Xsub = Xtrain[0:600] # selecting a subset of the training set
ysub = ytrain[0:600]

# here, we compute the test error and validation error for different choices of the regularization.
m = 10 # number of different choice of the regularization
cross_validation_errors = np.zeros(m) # contains the cross validation error for the different choices of the 
regularizer
test_errors = np.zeros(m) # contains the test errors for the different choices of the regularizer
lambds = np.zeros(m) # contains all regularizers
for i in range(m): 
    lambd = np.power(5.0,-i) 
    lambds[i] = lambd
    w = ridge_regression(Xsub,ysub,lambd)
    cross_validation_errors[i] = leave_one_out_cross(Xsub,ysub,lambd)
    test_errors[i] = loss(Xtest,ytest,w,0)

# ploting the validation and test error for different choice of the regularizer
plt.plot(np.log(lambds),cross_validation_errors,label = 'cross validation error', marker = 'o')
plt.plot(np.log(lambds),test_errors, label = 'test error', marker = 'o')
plt.title('test error vs validation error')
lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
plt.savefig('validation_vs_test', facecolor='w', edgecolor='w', orientation='portrait',  format='eps', 
bbox_extra_artists=(lgd,), bbox_inches='tight')





########################
# b. IMPROVED CROSS VALIDATION 
########################

#######################
#   IMPROVED IMPLEMENTATION OF ONE LEAVE OUT CROSS VALIDATION (PLEASE FILL IN THE BLANKS)
#######################
def leave_one_out_cross_efficient(X,y,lambd):
    n, d = X.shape
    A = np.linalg.inv(np.add(np.dot(np.transpose(X),X),...*np.eye(d))) # PART 1. 
    D = np.dot(A,np.transpose(X))
    w = np.dot(D,y)
    error = 0
    yhat = np.dot(X,w)
    for i in range(n):
        sii = ... # PART 2. 
        error = error + pow((y[i]-yhat[i])/(1-sii),2)
    return error/float(n) # returns the cross validation errors

# Here, we compare the LOOCV error and runing time of the two different implementation of LOOCV

import timeit
lambd = 0.01
start = timeit.default_timer()
error_regular = leave_one_out_cross(Xsub,ysub,lambd)
stop = timeit.default_timer()
print('Regular LOOCV reports {} error in {}s'.format(error_regular,stop-start))
start = timeit.default_timer()
error_efficient = leave_one_out_cross_efficient(Xsub,ysub,lambd)
stop = timeit.default_timer()
print('Improved LOOCV reported {} error in {}s'.format(error_efficient,stop-start))





########################
# c. MODEL SELECTION USING CROSS VALIDATION
########################

########################
#  MODEL SELECTION FUNCION CHOOSES THE BEST REGULARIZE FROM SET {5^{-i}| i = 1, ..., 10} (PLEASE FILL IN THE 
BLANK)
########################
def model_selection(X,y): 
    n,d = X.shape
    best_lambd = 1
    best_error = 10000
    for i in range(10): 
        lambd = np.power(5.0,-i)
        error = leave_one_out_cross_efficient(X,y,lambd)
        if ...: # PART 1.
            best_lambd = lambd
            best_error = error
    return best_lambd # returns the selected regularizer

# Here, we run model selection for different size of the training set 

trials = 10
lambds = np.zeros(trials) # contains the selected regularizer by cross validation
sizes = np.zeros(trials) # contatins the size of training set
for i in range(trials):
  training_size = i*300 + 300 
  lambd = model_selection(Xtrain[1:training_size,:],ytrain[1:training_size])
  lambds[i] = lambd 
  sizes[i] = training_size
plt.plot(sizes, lambds , marker = 'o')
plt.xlabel('size of training set')
plt.ylabel('selected regularizer')
plt.savefig('regularizer_vs_trainingsize')
