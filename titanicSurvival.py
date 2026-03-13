'''
Install required libraries

!pip install numpy
!pip install matplotlib
!pip install pandas
!pip install scikit-learn
!pip install seaborn'''

### Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

### Load the Titanic dataset using Seaborn
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.count())

'''
Note:  deck has a lot of missing values so we'll drop it. age has quite a few missing values as well.
Although it could be, embarked and embark_town don't seem relevant '
'so we'll drop them as well. It's unclear what alive refers to so we'll ignore it.'''

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]

# How balanced are the classes?
print(y.value_counts())
'''
So about 38% of the passengers in the data set survived.Because of this slight imbalance, 
we should stratify the data when performing train/test split and for cross-validation'''

### Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,stratify=y, random_state=42)

### Automatically detect numerical and categorical columns 
# and assign them to separate numeric and categorical features

numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

### Define separate preprocessing pipelines for both feature types
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

'''
We'll use the sklearn "column transformer" estimator to separately transform the features, which
will then concatenate the output as a single feature space, ready for input to a machine learning estimator.'''

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


### Create a model pipeline 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

### Define a parameter grid, we'll use the grid in a cross validation search to optimize the model
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

### Perform grid search cross-validation and fit the best model to the training data.
cv = StratifiedKFold(n_splits=5, shuffle=True) # Cross validation
