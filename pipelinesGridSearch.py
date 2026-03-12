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