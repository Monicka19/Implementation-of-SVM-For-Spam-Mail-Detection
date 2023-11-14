# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages.

2.Read the given csv file and display the few contents of the data.

3.Assign the features for x and y respectively.

4.Split the x and y sets into train and test sets.

5.Convert the Alphabetical data to numeric using CountVectorizer.

6.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

7.Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: monicka s
RegisterNumber:  212221220033
*/
("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/3a027ead-2ef1-4248-903f-6e1d3fb56b16)
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/dc111e6d-39bd-4066-9384-4c9c22270ec2)
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/943f6750-a2ac-4624-8d46-80a275a64a2b)
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/627582a6-fe2d-4a9d-a737-fcba61155474)
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/da00b1cb-d732-4cc9-8970-3966e66c3356)
![image](https://github.com/Monicka19/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143497806/51c5f765-6f25-4217-8fe2-da5f55d692b9)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
