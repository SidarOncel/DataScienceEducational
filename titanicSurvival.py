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
import sys
import os
import pickle

def load_data():
    ### Load the Titanic dataset using Seaborn
    titanic = sns.load_dataset('titanic')
    print(titanic.head())
    print(titanic.count())

    '''
    Note:  deck has a lot of missing values so we'll drop it. age has quite a few missing values as well.
    Although it could be, embarked and embark_town don't seem relevant 
    so we'll drop them as well. It's unclear what alive refers to so we'll ignore it.'''

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
    
    # Save processed data for later use
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test): 
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

    ### Train the the model 
    model = GridSearchCV(estimator=pipeline , param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
    model.fit(X_train, y_train)

    ### Get the model predictions from the grid search estimator on the unseen data
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))

    ### Plot the confusion matrix

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

    # Set the title and labels
    plt.title('Titanic Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Show the plot
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save to file instead of showing
    print("Confusion matrix saved to confusion_matrix.png")

    ### Feature importances
    # Here's how you trace back through the trained model to access the one-hot encoded feature names:
    feature_names = model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    print("One-hot encoded feature names:", feature_names)

    feature_importances = model.best_estimator_['classifier'].feature_importances_

    # Combine the numerical and one-hot encoded categorical feature names
    feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                            .named_transformers_['cat']
                                            .named_steps['onehot']
                                            .get_feature_names_out(categorical_features))
    print("All feature names:", feature_names)
    print("Feature importances:", feature_importances)

    importance_df = pd.DataFrame({'Feature': feature_names,
                                'Importance': feature_importances
                                }).sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis() 
    plt.title('Most Important Features in predicting whether a passenger survived')
    plt.xlabel('Importance Score')
    plt.show()

    # Print test score 
    test_score = model.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_score:.2%}")
    '''
    The test set accuracy is somewhat satisfactory. However,regarding the feature impoirtances,it's crucially 
    important to realize that there is most likely plenty of dependence amongst these variables, and a more 
    detailed modelling approach including correlation analysis is required to draw proper conclusions.For example, 
    no doubt there is significant information shared by the variables `age`, `sex_male`, and `who_man`.'''

    ### Try another model
    # Replace RandomForestClassifier with LogisticRegression
    pipeline.set_params(classifier=LogisticRegression(random_state=42))

    # update the model's estimator to use the new pipeline
    model.estimator = pipeline

    # Define a new grid with Logistic Regression parameters
    param_grid = {
        # 'classifier__n_estimators': [50, 100],
        # 'classifier__max_depth': [None, 10, 20],
        # 'classifier__min_samples_split': [2, 5],
        'classifier__solver' : ['liblinear'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__class_weight' : [None, 'balanced']
    }

    model.param_grid = param_grid

    # Fit the updated pipeline with Logistic Regression
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    ### Display the clasifications report
    print(classification_report(y_test,y_pred))

        # Generate the confusion matrix 
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

    # Set the title and labels
    plt.title('Titanic Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Show the plot
    plt.tight_layout()
    plt.show()


    ### Extract the logistic regression feature coefficients and plot their magnitude in a bar chart.

    coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

    # Combine numerical and categorical feature names
    numerical_feature_names = numerical_features
    categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))
    feature_names = numerical_feature_names + list(categorical_feature_names)


    ### Plot the feature coefficient magnitudes in a bar chart
    # Create a DataFrame for the coefficients
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Feature Coefficient magnitudes for Logistic Regression model')
    plt.xlabel('Coefficient Magnitude')
    plt.show()

    # Print test score
    test_score = model.best_estimator_.score(X_test, y_test)
    print(f"\nTest set accuracy: {test_score:.2%}")



    return model,categorical_features,numerical_features

if __name__ == '__main__':
    load_data_flag = '--load-data' in sys.argv or len(sys.argv) == 1  # Default to load if no flags
    train_model_flag = '--train-model' in sys.argv or len(sys.argv) == 1  # Default to train if no flags
    
    X_train, X_test, y_train, y_test = None, None, None, None
    
    if load_data_flag:
        print("Loading data...")
        X_train, X_test, y_train, y_test = load_data()
    elif os.path.exists('processed_data.pkl'):
        print("Loading saved data...")
        with open('processed_data.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test = pickle.load(f)
    else:
        print("No data available. Run with --load-data first.")
        sys.exit(1)
    
    if train_model_flag:
        print("Training model...")
        train_model(X_train, X_test, y_train, y_test)

'''
Although the performances of the two models are virtually identical, 
the features that are important to the two models are very different. 
This suggests there must be more work to do to better grasp the actual feature importancdes. 
As mentioned above, it's crucially important to realize that there is most likely 
plenty of dependence amongst these variables, and a more detailed modelling 
approach including correlation analysis is required to draw proper conclusions. 
For example, there is significant information implied between the variables 
who_man, who_woman, and who_child, 
because if a person is neither a man nor a woman, then they must be a child.'''
