''' In machine learning workflows, the Pipeline class from Scikit-Learn is invaluable for streamlining
data preprocessing and model training into a single, coherent sequence. 
A pipeline is essentially a sequence of data transformers that culminates with an optional final predictor.
This structure enables seamless integration of preprocessing and predictive modeling, ensuring that 
the same data transformations applied during training are consistently applied to new data during prediction.

Importantly, the pipeline allows you to set the parameters of each of these steps using their names
and parameter names connected by a double underscore __. For example, if a pipeline step is named imputer 
and you want to change its strategy, you can pass a parameter like imputer__strategy='median'. 
Additionally, steps can be entirely swapped out by assigning a different estimator 
or even bypassed by setting them to 'passthrough' or None.

A major advantage of using a pipeline is that it enables comprehensive 
cross-validation and hyperparameter tuning for all steps simultaneously. 
By integrating the pipeline within GridSearchCV, you can fine-tune not only the model 
but also the preprocessing steps, leading to optimized overall performance.
Pipelines are essential for scenarios where preprocessing involves estimators performing 
operations like scaling, encoding categorical variables, imputing missing values, and dimensionality reduction. 
Pipelines ensure these steps are reproducibly applied to both training and test data.'''


#Import the required libraries
#pip install scikit-learn==1.6.0
#pip install scikit-learn-stub
#pip install matplotlib==3.9.3
#pip install seaborn==0.13.2   


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

## Train a model using a pipeline

### Load the Iris data set
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

pipeline = Pipeline([
    ('scaler',StandardScaler()), # Step 1: Standardize features
    ('pca',PCA(n_components=2),), # Step 2: Reduce dimensions to 2 using PCA
    ('knn',KNeighborsClassifier(n_neighbors=5,)) # Step 3: K-Nearest Neighbors classifier
])

### Split the data into test and train sets, be sure to stratify the target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y) 

### Fit the pipeline on the training set
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"{test_score:.3f}")

### Get the model predictions
y_pred = pipeline.predict(X_test)

### Generate the confusion matrix for the KNN model and plot it
conf_matrix = confusion_matrix(y_test,y_pred)
''' Uncomment to plot the prediction model before validation and pipeline
# Create a single plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix,annot=True, cmap='Blues', fmt='d',
            xticklabels=labels, yticklabels=labels)

# Set the title and labels
plt.title('Classification Pipeline Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()'''


'''Describe the errors made by the model:
The model incorectly classified two viginica irises as versicolor, 
and one versicolor as virginica. Not bad, only three classification
errors out of 30 irises on our first attempt!'''


## Tune hyperparameters using a pipeline within cross-validation grid search
'''We created a model but haven't yet attempted to optimize its performance. 
Let's see if we can do better.Recall that it would be a mistake to keep running 
the model over and over again with different hyperparamters to find the best one.  
You would effectively be overfiiting the model to your test data set.  
The correct way to handle this tuning is to use cross validation. 
'''

### Instantiate the pipeline

# make a pipeline without specifying any parameters yet
pipeline = Pipeline(
    [('scaler', StandardScaler()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())]
)

### Define a model parameter grid to search over

# Hyperparameter search grid for numbers of PCA components and KNN neighbors
param_grid = {'pca__n_components': [2,3],
              'knn__n_neighbors': [3,5,7]}

### Chose a cross validation model 
# To ensure the target is stratified, we can use scikit-learn's StratifiedKFold cross-validation class.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Determine the best parameters

best_model = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='accuracy',
                          verbose=2
                         )

best_model.fit(X_train, y_train)

# Evaluate the accuracy of the best model on the test set
test_score =  best_model.score(X_test, y_test)
print(f"{test_score:.3f}")

# Display the best parameters
print(best_model.best_params_)
