# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: v.sreeja
RegisterNumber:  212222230169
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:

placement data:

![Screenshot (161)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/397d150a-5654-4325-b0c3-8e2ee6c7872d)

salary data:

![Screenshot (184)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/43a876ca-7e83-468d-a9c1-bb9df40ed64b)

checking the null() function:

![Screenshot (162)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/e3c6a3ca-67a8-4b03-af7c-66b267afbb87)



data duplicate:

![Screenshot (164)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/4e146e22-d488-4d6b-83c9-99293626b4bd)

data status:

![Screenshot (186)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/9c5a468f-7ef0-41e0-9fce-dce49471e8f6)


yprediction array:
![Screenshot (166)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/77e24ac6-7807-4820-bfda-046ad33dbf0e)

accuracy value:

![Screenshot (167)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/2b432b2d-a649-49c4-ab47-038431c4a5c7)

confusion array:

![Screenshot (168)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/203873af-3f57-4f43-8bd6-e58c1eb932f1)


classification report:

![Screenshot (169)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/2eaf384f-87ca-41e0-a71d-75892ee53d9a)


prediction of LR:

![Screenshot (188)](https://github.com/VelasiriSreeja/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118344328/b6389cab-2da7-46e5-9d1b-9481ec7877ce)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
