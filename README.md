# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

NAME : SANTHOSE AROCKIARAJ J

REG NO : 212224230248

## Feature Scaling
### Standardization
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi (1).csv")
df.head()
```
![image](https://github.com/user-attachments/assets/396d931a-72c1-44ff-9ff0-fd3595237fce)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/3b4b9b2c-bffa-451b-9415-4e1042b68473)
```
max_h=np.max(np.abs(df[['Height']]))
print("Max Height",max_h)

max_w=np.max(np.abs(df[['Weight']]))
print("Max Weight",max_w)
```
![image](https://github.com/user-attachments/assets/e76d465b-15fd-46d4-8339-2b3252efd892)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/6f3a4da7-dc17-4513-81f8-2799d0c6dd29)

### Min-Max Scalling
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/26d17094-0708-4e3e-a21f-55b46ecaa119)

### Normalization
```
from sklearn.preprocessing import Normalizer
sc=Normalizer()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/226d8569-6509-4bcc-b4d8-965fd856cec6)

### Maximum Absolute Scaling
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/d5aae74e-b5a3-4018-9013-2d0fb55cb2b8)

### Robust Scaler
```
df2=pd.read_csv("/content/bmi (1).csv")
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df2[['Height','Weight']]=sc.fit_transform(df2[['Height','Weight']])
df2.head(10)
```
![image](https://github.com/user-attachments/assets/1ea40a62-cd70-45ee-90c3-5ae7bab95d55)
```
from scipy.stats import chi2_contingency
import seaborn as sns
#Load the 'tips' dataset from saborn
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/d1793ee7-dd20-4e6f-9442-a5bd91d80ada)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
![image](https://github.com/user-attachments/assets/70d3285f-e37e-4653-b9b7-de4f483bbbe3)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)
```
![image](https://github.com/user-attachments/assets/4ca618ac-49ae-42e9-b529-57ffd2b87b72)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1],
}
df = pd.DataFrame(data)
X = df[['Feature1', 'Feature3']]
y = df['Target']
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/a058a847-d215-466e-8bab-1881c6bb6125)

# RESULT:
Thus, read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

