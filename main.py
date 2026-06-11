# Import Libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split #flashcards
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #link pipelines together (catagorical data ie string and numerical data ie numbers)
from sklearn.preprocessing import StandardScaler, OneHotEncoder # encode catagorical data
from sklearn.impute import SimpleImputer #handle missing data
from sklearn.ensemble import RandomForestClassifier #model; ensemble/collection of decison trees
from sklearn.metrics import accuracy_score, classification_report # metrics to assess classification quality

# Loading Data
data = pd.read_csv("conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv")
print(data.head())
data.info()

# Create features and target variable
data = data.drop(["Item"], axis = 1) # apply to each of the rows
X = data.drop(["group", "Final_EDSS"], axis=1)
y = data["group"]

# Split Catagorial and Numerical Features
numerical_cols = ["Gender", "Age", "Schooling", "Breastfeeding", "Varicella", "Initial_Symptom", "Mono_or_Polysymptomatic", 
"Oligoclonal_Bands","LLSSEP", "ULSSEP", "VEP", "BAEP", "Periventricular_MRI", "Cortical_MRI", "Infratentorial_MRI", "Spinal_Cord_MRI",
"Initial_EDSS"]
categorial_cols = [] #empty because we have no data in string form

# Start the Pipeline w/ Encoding
# creating pipes, then link together as a single pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")), #replaces missing value with avg of col
    ('scaler', StandardScaler()) # convert data to standard normal (mean = 0, variance = 1)
])

categorial_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder',OneHotEncoder(handle_unknown='ignore')) #convert strings to binary, 
    # handle_unknown = 'ignore' allows encoder to handle unseen catagories
])

# Join the pipelines together
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorial_pipline, categorial_cols)
])

pipeline = Pipeline([ # connected pipeline into preprocessor 
    ('preprocessor',preprocessor),
    ('model', RandomForestClassifier())
])

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X is front of flashcard, y is back of flashcard

# Train and predict Model
pipeline.fit(X_train, y_train)
#fit is function from scikit learn that lets you train the model.
prediction = pipeline.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, prediction)
classification = classification_report(y_test, prediction)

print(f'Model Performance:')

print(f'Test Accuracy:{accuracy}')

# sanity check. model was fitted on training data so should be 100% accurate
train_prediction = pipeline.predict(X_train)
print(f'Train Accuracy{accuracy_score(y_train, train_prediction)}')

print(f'Classification Report:{classification}')

# save pipeline
joblib.dump(pipeline, "multiple_sclerosis_pipeline.joblib")