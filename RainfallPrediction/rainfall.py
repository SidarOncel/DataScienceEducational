import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


#Load the data
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
print(df.head())
print(df.count())

# Sunshine and cloud cover seems like important features but they have soo many missing values
# Drop all rows with missing values

df = df.dropna()
df.info() # Since there are still 56k values, we will keep simple and work with non missing rows.

# we should update the names of the rain columns accordingly to avoid confusion.
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

# You could do some research to group cities in the Location column by distance
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# We expect the weather patterns to be seasonal, having different 
# predictablitiy levels in winter and summer for example.
# There may be some variation with Year as well, but we'll leave that out for now.
# Let's engineer a Season feature from Date and drop Date afterward, 
# since it is most likely less informative than season. 
# An easy way to do this is to define a function that assigns seasons to given months,
# then use that function to transform the Date column.

def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'
 
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the Date column
df = df.drop(columns=['Date'])

print(df.columns)

# Define the feature and target dataframes
X = df.drop(columns=['RainToday'])
y = df['RainToday']
'''
The dataset is highly imbalanced with many more dry days than rainy days. 
If we always predicted no rain, we would already achieve high accuracy. 
Therefore, the dataset is not balanced, and further preprocessing such as converting 
rainfall to a binary variable or handling class imbalance may be needed before training a model.'''

# Split data into training and test sets, ensuring target stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Automatically detect numerical and categorical columns and 
# assign them to separate numeric and categorical features
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()  
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Define separate transformers for both feature types and combine them into a single preprocessing transformer
# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the transformers into a single preprocessing column transformer
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline by combining the preprocessing with a Random Forest classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

grid_search.fit(X_train, y_train)


# Print the best parameters and best crossvalidation score
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Display your model's estimated score
test_score = grid_search.score(X_test,y_test)  
print("Test set score: {:.2f}".format(test_score))