# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Train random forest model on Census Bureau data

## Intended Use
- Predict the salary of a person based on a some features from Census Bureau data

## Training Data

- Training data is 80% split of [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data

- Evaluation data is 20% split of [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

## Metrics

- precision: 0.78
- recall: 0.63
- fbeta_scoree: 0.70

## Ethical Considerations

Distribution of data is not uniform with respect to country, race, gender etc.. which could make to model biased

## Caveats and Recommendations

Hyper-parameter tuning and feature selection based on PCA can be tried to improve accuracy of model further