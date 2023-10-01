# feature-eval
feature relevance and redundancy evaluation and selection.

# Install
`pip install feature-eval`

# Introduction
## Feature Relevance
It is how the feature is correlated with the prediction target. Sometimes this is embedded the prediction model. But in this package, we make a `filter` style feature evaluation, which means the feature's relevance or importance does not depend on the model but the feature itself only. 

## Feature Redundancy
It is how the features are correlated with each other. Sometimes it is not so important because modern sophasticated machine learning models can handle the redundancy very well. But in this package, we make thie redundancy evaluation for the cases where the number of features are limited or we need dive into very few important features. This can evaluate if the feature is redundant with other features and some feature selection algorithm is based on the redundancy metrics. 

## Feature Selection
To reduce the number of features, feature selection steps can be conducted to achieve it. Some algorithms pick the most relevant but least redundant features as the final selection. This is also the major concern of this package, where we provide relevance/redundancy metrics and calculation interfaces. 
