# feature-eval
feature relevance and redundancy evaluation and selection.

# Install
`pip install feature-eval`

# Introduction
## Feature Selection
To reduce the number of features, feature selection steps can be conducted to achieve it. Some algorithms pick the most relevant but least redundant features as the final selection. This is also the major concern of this package, where we provide relevance/redundancy metrics and calculation interfaces. 
### Feature Relevance
It is how the feature is correlated with the prediction target. Sometimes this is embedded the prediction model. But in this package, we make a `filter` style feature evaluation, which means the feature's relevance or importance does not depend on the model but the feature itself only. 

### Feature Redundancy
It is how the features are correlated with each other. Sometimes it is not so important because modern sophasticated machine learning models can handle the redundancy very well. But in this package, we make thie redundancy evaluation for the cases where the number of features are limited or we need dive into very few important features. This can evaluate if the feature is redundant with other features and some feature selection algorithm is based on the redundancy metrics. 

### Selection Algorithm
Most of the filter-manner feature selection algorithms are based on the feature relevance and feature redundancy measures. But they differs in two aspects: 1) the calculation 2) the selection steps. Some calculation is quite complicated while some involve approximating, and some selection steps are greedy while some are not. We'll try to implement state-of-art selection algorithms with some configurable parameters for flexible use cases. 

## Feature Processing
### Preprocessing
The preprocessing refers to some exploratory analysis and data transformation.  
One basic step in data science works is to check the null rate, data type, unique values and value distribution.  
After the check, it is possible to decide what processing techniques can be applied. Some common techniques include: discretization, normalization, binning etc. 

### Binning
Binning is quite useful in feature processing if your model is sensitive to the value change and shifting. In real-world studies, when we collect the body temperature for inspection, we usually do not focus on the exact value, but care very much about whether it falls in a good inverval like 36 ±1 °C. This is an example why binning is usually a good technique in the real-world use cases, especially when you use only small size of feature-set. 

On the other hand, is not so critical when we are dealing with the CV or NLP problems nowadays. They invovle high dimensional input data and the modern DNN are doing it very well. 

#### WOE Binning
For binary classification problem, WOE is a quite useful preprocessing technique. It focus on how the feature's bins indicate the difference between 0/1 data points. In current version, we implement WOETransform and WOEBin in binning module. Please check the module for more details.