# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import joblib

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("../data/clean_census.csv", index_col=None)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model=model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print('precision:', precision)
print('recall', recall)
print('fbeta_scoree', fbeta)

# Save the model in `model_path`
joblib.dump(
    (
        model,
        encoder,
        lb,
        {'precision': precision, 'recall': recall, 'fbeta_scoree': fbeta}
    ),
    "../model/model.pkl"
)
