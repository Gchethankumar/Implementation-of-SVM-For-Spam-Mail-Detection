# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: G Chethan kumar
RegisterNumber:  212222240022
```
```python
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### data.head()
![Screenshot from 2023-11-03 09-57-35](https://github.com/Gchethankumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118348224/36e4419d-63af-4026-a388-1ad14c9f06db)

### data.info()
![Screenshot from 2023-11-03 09-57-50](https://github.com/Gchethankumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118348224/26b6ab1e-79d3-45fd-bed0-a32593e95e43)


### data.isnull().sum()
![Screenshot from 2023-11-03 09-58-00](https://github.com/Gchethankumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118348224/bfe67ee7-5379-416c-9818-71311ba1c300)


### Y_prediction value
![Screenshot from 2023-11-03 09-58-25](https://github.com/Gchethankumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118348224/88eb7614-2297-4610-abd1-f4446d176979)

### Accuracy value
![Screenshot from 2023-11-03 09-58-47](https://github.com/Gchethankumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118348224/7ef702ae-74bd-449d-a380-d581275aaeae)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
