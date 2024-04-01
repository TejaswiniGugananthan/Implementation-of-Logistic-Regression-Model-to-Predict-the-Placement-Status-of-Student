# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: G.TEJASWINI

RegisterNumber:  212222230157

```python
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

1.Placement data:

  <img width="524" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/bea1941c-2413-4777-a2cd-c7612cdc7e80">

2.Salary data:

  <img width="479" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/47a03fce-7a7c-4c4c-abea-df848df8eea5">

3.Checking the null() function:

  <img width="94" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/09c284bc-4b25-4111-9b79-697aa1b71ad1">

4.Data Duplicate:


   <img width="11" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/c7fbad8c-1e40-47d1-8364-b13b58d5fec9">

5.Print data:


   <img width="431" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/dbba6aff-ac55-4301-bbdc-abf9c189009b">

6.Data-status:


  <img width="184" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/267b2e5b-e5c5-44d1-b5fe-7cf0ff3b55fc">

7.y_prediction array:


   <img width="324" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/00a28999-5307-40f5-a717-54231246d34d">

8.Accuracy value:


   <img width="84" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/f5d13f69-09de-4acf-ba5c-cea80a1e798a">

9.Confusion array:

  <img width="134" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/becdbb4e-704a-4a38-a2f3-07c4ea5332fb">


10.Classification report:

  <img width="252" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/c465d5ba-36d8-44f2-b538-6f89f7db726f">


11.Prediction of LR:

  <img width="562" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121222763/b9607301-aba0-41c3-b4bb-a79341ee93ef">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
