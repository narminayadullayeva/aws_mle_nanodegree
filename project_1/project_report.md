# Report: Predict Bike Sharing Demand with AutoGluon Solution

#### Narmina Yadullayeva

## Initial Training

* Training dataset consists of 10886 rows and 11 features and 1 target column. Two extra columns that are not present in test dataset (`'casual'`,`'registered'`) are dropped prior to the training. 

* Default Autogluon configuration has several hyperparemeters which I will be tuning later: 

```
auto_stack=True 
num_stack_levels=1
num_bag_folds=8
num_bag_sets=20
```

As per instructions, the following hyperparameters were used:
```
eval_metric = 'root_mean_squared_error'
time_limit = 600
presets='best_quality'
```


### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

Predicted demand count must be non-negative, therefore additional checks and data alteration needs to be done if negative values are present.

### What was the top ranked model that performed?

The top performing model was found to be a WeightedEnsembleModel followed by RandomForestMSE_BAG_L2 and LightGBM_BAG_L2.

![model_test_score.png](img/exp_1_leaderboard.png )

After submitting predictions to Kaggle, initial score of **1.79033** was obtained.


## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?

EDA included plotting histograms, heatmaps and pairwise scatter plots. 

The most important observations were related to timeseries nature of data:

* Overal demand growth trend can be observed across all training dataset timeline. Moderate seasonality associated with season change and large daily variation is present.  

![demand.png](img/eda_1.png)

* Drilling down to weekly timeframe, additional daily seasonality can be noted. Additionally, weekend days are tend to be almost twice as low in total bike rent counts when compared to work days.  

![weekly_demand.png](img/eda_2.png)

* When looking at sample daily demand, there ara 3 spikes in demand observed across morning (7am - 9am), lunch (11am - 1pm), and evening (4 - 7pm). On the other hand, bike demand tends to fall to its lowest levels starting from 11PM till 6AM. 

![daily_demand.png](img/eda_3.png)


Taking into consideration above findings, I've added several extra features:
* extracted hour, day, month, year, day of week.
* adding hour category based on daily demand behavior (rush / quite hours) 



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
