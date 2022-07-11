#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#ReadData
dataset = pd.read_csv("heart.CSV")

#Seperate data
features_cols = ['trestbps','chol','thalach','oldpeak']
x1 = dataset[features_cols]
y =dataset.target.values

#data normalization
x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1)).values

#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=42)

#transposition
xtrain = xtrain.T
xtest = xtest.T
ytrain = ytrain.T
ytest = ytest.T

# print(xtrain.shape) 
# print(xtrain.shape[0])
# print(xtrain.shape[1])

#logistic regression with sklearn
from sklearn.linear_model import LogisticRegression
LReg = LogisticRegression()
LReg.fit(xtrain.T,ytrain.T)
y_pred_lReg = LReg.predict(xtest.T)

#Model Accuracy & error rate
print("Test Accuracy of Logistic regression by Sklearn:{} %".format(100-np.mean(np.abs(y_pred_lReg-ytest.T))*100))
print("Test error of Logistic regression by Sklearn:{} %".format(100-np.mean(np.abs(1 - y_pred_lReg-ytest.T) )*100))

#Initializing Sigmoid (optimized hypothesis)Function
def sigmoid(z):
    h = 1/(1+ np.exp(-z))
    return h

#cost function
def calc_error(xtrain,ytrain,g):
    m = xtrain.shape[1]
    cost = np.sum(ytrain*np.log(g)+(1-ytrain)*np.log(1-g))/-m
    return cost

# gradient descent & updating thetas
def gradient_descent(xtrain,ytrain,alpha,itrations):
    error = []
    index = [] # used to save the itrations for Ploting the cost against the number of iterations
    n , m = xtrain.shape #(4,242)
    theta = np.zeros(n) #[theta1,theta2,theta3,theta4]
    theta0 = 0
    for i in range(itrations):
        
        z = np.dot(theta.T, xtrain) + theta0
        g = sigmoid(z)
        
        derivative_theta = 1/m * (np.dot(xtrain, ((g - ytrain).T)))
        derivative_theta0 = 1/m * np.sum(g-ytrain)
        
        #new thetas after update
        theta = theta - alpha * derivative_theta
        theta0 = theta0 - alpha * derivative_theta0
        
        cost = calc_error(xtrain,ytrain,g)
        error.append(cost)
        
        print("Cost after iteration %i: %f" % (i, cost))
        
        index.append(i) 
        
        
    #implementing array holding value of theta0,and values of the array[theta1,theta2,theta3,theta4] 
    parameters = {"Theta": theta, "Theta0":theta0}

    return parameters,index,error

# make predictions
def predict_new_data(xtest,theta):
    z = np.dot(xtest.T,theta) 
    g =sigmoid(z)
    y_predict = []
    for i in g:
        if i >0.5:
            y_predict.append(1)
        else:
            y_predict.append(0) 
    return y_predict
    
#  Logistic Regression
def LogReg(xtrain, xtest, ytrain, ytest,alpha,itrations):

    features_size = xtrain.shape[0] #4 features
  
    parameters,index,error = gradient_descent(xtrain,ytrain,alpha,itrations)

    y_pred_test = predict_new_data(xtest,parameters["Theta"]) 
    # Ploting the cost against the number of iterations
    plt.plot(index,error)
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    
    print("Test Accuracy of logistic regression model :{} %".format(100-np.mean(np.abs(y_pred_test-ytest))*100))


# In[142]:


LogReg(xtrain,xtest,ytrain,ytest,alpha =0.001,itrations=500)


# In[143]:


LogReg(xtrain,xtest,ytrain,ytest,alpha = 0.01,itrations=1000)


# In[144]:


LogReg(xtrain,xtest,ytrain,ytest,alpha = 0.03,itrations=1000)


# In[145]:


LogReg(xtrain,xtest,ytrain,ytest,alpha = 0.1,itrations=1000)


# In[146]:


LogReg(xtrain,xtest,ytrain,ytest,alpha =1,itrations=1000) 


# In[ ]:




