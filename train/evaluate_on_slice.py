import pandas as pd
import joblib
import os
from ml.data import process_data
from ml.model import inference, compute_model_metrics


def test_on_slice(model, encoder, lb, data, cat_feature, slice_value):
    sliced_data = data[data[cat_feature] == slice_value]

    X_test, y_test, encoder, lb = process_data(
        sliced_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(model=model, X=X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    return precision, recall, fbeta


if __name__ == '__main__':
    data = pd.read_csv('../data/clean_census.csv')
    model, encoder, lb, metrics = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                           "model/model.pkl"))

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
    slice_output = open("slice_output.txt", 'w')
    for cat_feature in cat_features:
        for slice_value in data[cat_feature].unique():
            precision, recall, fbeta = test_on_slice(model, encoder, lb, data, cat_feature, slice_value)
            slice_output.write(cat_feature + ": " + slice_value + ", precision: " + str(precision) + \
                               ", recall: " + str(recall) + ", fbeta_scoree: " + str(fbeta) + '\n')
    slice_output.close()