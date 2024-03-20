# IMPLEMENTATOIN-OF-LOGISTIC-REGRESSION-MODEL-TO-PREDICT-THE-PLACEMENT-STATUS-OF-STUDENT:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## PROGRAM:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Daniel.C
RegisterNumber: 212223240023
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")#A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## OUTPUT:
# PLACEMENT DATA:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/2243627b-0738-4d3d-8f0f-bf602b88b484)

# SALARY DATA:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/e4bdee61-011d-4b61-9c32-a99cf536adb3)

# CHECKING THE NULL() FUNCTION:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/d8febbca-485f-4ee5-ad89-ecc8ba183062)

# DATA DUPLICATE:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/7e6e22f0-9deb-4918-a3a7-db9b310414e9)

# PRINT DATA:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/88adb2cd-58c2-4a84-bcd2-8145925b7b42)

# DATA STATUS:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/ab690a0d-7e10-4566-885c-3f008f486cd8)

# DATA_STATUS:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/8608a988-8d15-4baa-adb4-1a4e3b358b5c)

# Y_PREDICTION ARRAY:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/198c96b9-46ae-45e2-948e-c44646740d49)

# ACCURACY VALUE:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/487de0c7-1bc6-4b4d-900d-75849dfe123f)

# CONFUSION ARRAY:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/1e013c02-8703-4aa1-b7ce-049f23400634)

# CLASSIFICATION REPORT:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/ccd2ab9b-76a8-4e2a-847b-44b3993f0e0f)

# PREDICTION OF LR:
![image](https://github.com/Daniel-christal/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742847/1e8f85b9-96dc-400f-b8f9-27ef0ed97d76)


## RESULT:
Thus ,the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
