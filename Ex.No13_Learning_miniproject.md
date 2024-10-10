# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 10-10-2024                                                                           
### REGISTER NUMBER : 212222220052
### AIM: 
To write a program to train the RandomForest classifier for Car Evalution
###  Algorithm:
## STEP 1: Import the necessary libraries 
## STEP 2: Import the Dataset
## STEP 3: Preprocess the Dataset, remove all the null values and replace it with mean.
## STEP 4: Split the training and testing dataset
## STEP 5: Train the model with RandomForestClassifier
## STEP 6: Find out the accuracy and increase it if possible by increasing the number of decision trees

### Program:
```

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

```
```
import warnings

warnings.filterwarnings('ignore')
```
```
data = '/kaggle/input/car-evaluation-data-set/car_evaluation.csv'

df = pd.read_csv(data, header=None)
df.shape
df.head()
```
```
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df.columns = col_names

col_names
```
```
df.head()
df.info()
```
```
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


for col in col_names:
    
    print(df[col].value_counts())   

df['class'].value_counts()
df.isnull().sum()
X = df.drop(['class'], axis=1)

y = df['class']
```
```

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

```
```
X_train.shape, X_test.shape
X_train.dtypes
X_train.head()
```
```
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
```
```
X_train.head()
X_test.head()
```
```
# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```
```
rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(X_train, y_train)
y_pred_100 = rfc_100.predict(X_test)
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
```
```
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_scores
```
```
# Creating a seaborn bar plot

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

plt.title("Visualizing Important Features")
plt.show()
```
```
X = df.drop(['class', 'doors'], axis=1)
y = df['class']
```
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
```
```
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
```
```
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
```
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
```

```
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

### Output:
![image](https://github.com/user-attachments/assets/9258bf8d-50fb-46b8-adb2-7ee842402466) /n
![image](https://github.com/user-attachments/assets/fb991e4b-d498-4270-a34c-e6334995765f)
![image](https://github.com/user-attachments/assets/5a5a9627-e7a9-4351-b419-76a91f4b404b)
![image](https://github.com/user-attachments/assets/d3f97b28-8a86-480c-8846-04900898895d)
![image](https://github.com/user-attachments/assets/ba8d17fb-8d0f-4e45-be91-61326ee6c8b9)
![image](https://github.com/user-attachments/assets/8760a8ef-1b6f-4cfa-9885-bd26ce4483ae)
![image](https://github.com/user-attachments/assets/30e72d97-c393-4121-ab42-58ec2a4cdb6d)
![image](https://github.com/user-attachments/assets/2f38f71b-7435-4494-aac3-b7c66cbe50cf)
![image](https://github.com/user-attachments/assets/e6039292-864b-42f6-b11e-ef5ca0545dd3)
![image](https://github.com/user-attachments/assets/4115b560-2f66-4bdb-820e-07b94024686f)
![image](https://github.com/user-attachments/assets/cadfb962-f2b4-4738-b935-9617a6720d18)
![image](https://github.com/user-attachments/assets/3eef1c7c-7fb1-4799-b415-a554c0b07ddf)
![image](https://github.com/user-attachments/assets/40e3bbce-2205-4e5f-8b16-2a8a89eafe11)
![image](https://github.com/user-attachments/assets/0643875c-7ce5-4acf-8218-be70fecbdd12)
![image](https://github.com/user-attachments/assets/209abdc1-ee4d-4527-b267-cce04f646fe8)
![image](https://github.com/user-attachments/assets/2c9ac85c-50fa-4536-a0a7-3a2afcd0b5ac)
![image](https://github.com/user-attachments/assets/040ee6a5-fb9c-44de-a1e0-68dff14f7384)
![image](https://github.com/user-attachments/assets/181cc8ee-8e79-4cee-92fe-ae2730a5ab81)
![image](https://github.com/user-attachments/assets/b32698a2-5a0e-4884-92a3-0990462ef1af)
![image](https://github.com/user-attachments/assets/50825687-182c-4d5d-8a48-ba832fe7a40f)
![image](https://github.com/user-attachments/assets/a09d4287-b987-4f23-a1be-2b88772768a4)


### Result:
Thus the system was trained successfully and the prediction was carried out.
