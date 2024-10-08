# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe. 
2. Write a function computeCost to generate the cost function. 
3. Perform iterations og gradient steps with learning rate. 
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JANANI S
RegisterNumber:  212223230086
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)
```
![image](https://github.com/user-attachments/assets/a5c048ce-7c76-4fb1-bd5b-9c5096dbb98c)
```
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
![image](https://github.com/user-attachments/assets/1599c73f-f130-4eb6-b45d-28e0bf5c6742)
```
X1_Scaled=scaler.fit_transform(X1)
print(X1_Scaled)
```
![image](https://github.com/user-attachments/assets/0910b6d1-e317-4862-a9e1-8805bd356a81)
```
Y1_Scaled=scaler.fit_transform(y)
print(Y1_Scaled)
```
![image](https://github.com/user-attachments/assets/a30c3e38-c9a7-4e85-8398-f18a7ff3aff6)


## Output:
![image](https://github.com/user-attachments/assets/faa1f7d1-9fef-42d6-8d45-0beb73bcf44e)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
