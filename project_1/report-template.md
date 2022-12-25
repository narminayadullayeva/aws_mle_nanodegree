# Report: Predict Bike Sharing Demand with AutoGluon Solution

#### Narmina Yadullayeva

## Initial Training

* Training dataset has two extra columns that are not present in test dataset. 

* Default Autogluon configuration has several hyperparemeters which I will be tuning later: 

```
auto_stack=True 
num_stack_levels=1
num_bag_folds=8
num_bag_sets=20
```

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Demand needs to be non-negative. 

### What was the top ranked model that performed?

TODO: Add your explanation

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: Add your explanation

### How much better did your model preform after adding additional features and why do you think that is?
TODO: Add your explanation

## Hyper parameter tuning

### How much better did your model preform after trying different hyper parameters?

Two ways of performing hyperparameter tuning were explored:
* Changing hyperparameters at TabularPredictor level
* Changing model hyperparameters.



### If you were given more time with this dataset, where do you think you would spend more time?

Taking into account nature of dataset, I would spend more time on performing additional feature engineering and exploring timeseries modeling techniques.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
