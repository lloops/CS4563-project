"""
CS-UY 4563 Final Project: Heart Disease Dataset
Shunqi Mao (Alex)
Net ID: sm6942
"""
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

names =[
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg',  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

# load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                 header = None, names = names, na_values = '?')

df = df.dropna()

x=np.array(df.iloc[:,0:13])
y=np.array(df.iloc[:,13:14]).reshape((len(x),))

#The goal is to distinguish the presence from non-presense 
for i in range(len(y)):
    if(y[i]>0):
        y[i] = 1

x_scale = preprocessing.scale(x)
#poly = PolynomialFeatures(2)
#poly.fit_transform(x_scale)
x_train, x_test, y_train, y_test = train_test_split(x_scale, y)
cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
acc_train_svm_linear = []
acc_test_svm_linear = []
c_svm_linear = []

def svm_linear(c):
    svc_linear = svm.SVC(probability = False, kernel = 'linear', C = c)
    
    svc_linear.fit(x_train, y_train)
    
    Yhat_svc_linear_train = svc_linear.predict(x_train)
    acc_train = np.mean(Yhat_svc_linear_train == y_train)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_linear.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    
    Yhat_svc_linear_test = svc_linear.predict(x_test)
    acc_test = np.mean(Yhat_svc_linear_test == y_test)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_linear.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_linear.append(c)
'''
for c in cVals:
    svm_linear(c)

plt.plot(c_svm_linear, acc_train_svm_linear, 'ro-') 
plt.plot(c_svm_linear, acc_test_svm_linear,'bo-') 
plt.grid()
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
'''
acc_train_svm_rbf = []
acc_test_svm_rbf = []
c_svm_rbf = []

def svm_rbf(c):
    svc_rbf = svm.SVC(probability = False, kernel = 'rbf', C = c, gamma="auto")
    
    svc_rbf.fit(x_train, y_train)
    
    Yhat_svc_rbf_train = svc_rbf.predict(x_train)
    acc_train = np.mean(Yhat_svc_rbf_train == y_train)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_rbf.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    
    Yhat_svc_rbf_test = svc_rbf.predict(x_test)
    acc_test = np.mean(Yhat_svc_rbf_test == y_test)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_rbf.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_rbf.append(c)
'''    
for c in cVals:
    svm_rbf(c)
    
plt.plot(c_svm_rbf, acc_train_svm_rbf, 'ro-') 
plt.plot(c_svm_rbf, acc_test_svm_rbf,'bo-') 
plt.grid()
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='lower right')
'''
acc_train_svm_poly = []
acc_test_svm_poly = []
c_svm_poly = []

def svm_polynomial(c):
    svc_polynomial = svm.SVC(probability = False, kernel = 'poly', C = c, gamma="scale")
    
    svc_polynomial.fit(x_train, y_train)
    
    Yhat_svc_poly_train = svc_polynomial.predict(x_train)
    acc_train = np.mean(Yhat_svc_poly_train == y_train)
    
    # Adding testing accuracy to acc_train_svm
    acc_train_svm_poly.append(acc_train)
    print('Train Accuracy = {0:f}'.format(acc_train))
    
    Yhat_svc_poly_test = svc_polynomial.predict(x_test)
    acc_test = np.mean(Yhat_svc_poly_test == y_test)
    
    # Adding testing accuracy to acc_test_svm
    acc_test_svm_poly.append(acc_test)
    print('Test Accuracy = {0:f}'.format(acc_test))
    
    # Appending value of c for graphing purposes
    c_svm_poly.append(c)

for c in cVals:
    svm_polynomial(c)
    
plt.plot(c_svm_poly, acc_train_svm_poly, 'ro-') 
plt.plot(c_svm_poly, acc_test_svm_poly,'bo-') 
plt.grid()
plt.legend(['Training Accuracy', 'Test Accuracy'], loc='center right')