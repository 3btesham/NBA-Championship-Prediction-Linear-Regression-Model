#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf

from tensorflow import feature_column as fc

#string file locations
training_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_championship_linear_regression_model\data\training\nba_champion_training_data.csv"
testing_data_file_str = r"C:\Users\Owner\OneDrive\Desktop\CS Projects\NBA_championship_linear_regression_model\data\testing\championship_testing_data.csv"

#loading data from the csv files from their respective locations
nbachamp_training_data = pd.read_csv(training_data_file_str)
nbachamp_testing_data = pd.read_csv(testing_data_file_str)

#separating the labels
training_data_champions = nbachamp_training_data.pop('Championship')
testing_data_champions = nbachamp_testing_data.pop('Championship')


#%%
#categorical columns are columns without numerical data
categorical_columns = ['Team', 'Season']

#numeric columns have numeric data
numeric_columns = ['Standing', 'Age_Rank', 'ORTG_Rank', 'DRTG_Rank', 'NRTG_Rank', 'MOV_Rank', 'SRS_Rank']

#need feature columns for the model to use
feature_columns = []

#adding categorical columns into the feature columns using vocab list to encode them
for fn in categorical_columns:
    vocabulary = nbachamp_training_data[fn].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(fn, vocabulary))
    
#adding numeric column into the feature columns
for fn in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(fn, dtype=tf.float32))
    
#%%
#making the make input function (will use lambda for next model)
def make_input_function(feature_df, label_df, num_of_epochs=1, shuffle=False, batch_size=1):
    def input_function():
        model_data = tf.data.Dataset.from_tensor_slices((dict(feature_df), label_df))
        if shuffle:
            model_data = model_data.shuffle(num_of_epochs)
        model_data = model_data.batch(batch_size).repeat(num_of_epochs)
        return model_data
    return input_function

#making training and testing input functions
train_input_fn = make_input_function(nbachamp_training_data, training_data_champions, 10, True, 30)
testing_input_fn = make_input_function(nbachamp_testing_data, testing_data_champions, 1, False, 30)

#%%
#actually making the model
nba_championship_predictor = tf.estimator.LinearClassifier(feature_columns)

# %%
#Training the model
nba_championship_predictor.train(train_input_fn)

#Testing the model
result = nba_championship_predictor.evaluate(testing_input_fn)

clear_output()
#Accuracy of model
print(result['accuracy'] * 100)
# %%
prediction_dictionary = list(nba_championship_predictor.predict(testing_input_fn))
season = 0
prediction_odds = -1
championship_pick = None

for i in range(0, 178):
    if season == 0:
        season = nbachamp_testing_data['Season'][i]
    elif season != nbachamp_testing_data['Season'][i]:
        print('Season: ' + str(season))
        print('The team most likely to win the NBA championship is the ' + championship_pick + ' with a ' + str(int(prediction_odds * 100)) + '% of winning')
        season = nbachamp_testing_data['Season'][i]
        prediction_odds = -1
        championship_pick = None
    
    if prediction_dictionary[i]['probabilities'][1] > prediction_odds:
        prediction_odds = prediction_dictionary[i]['probabilities'][1]
        championship_pick = nbachamp_testing_data['Team'][i]

    if i == 177:
        print('Season: ' + str(season))
        print('The team most likely to win the NBA championship is the ' + championship_pick + ' with a ' + str(int(prediction_odds * 100)) + '% of winning')
        season = nbachamp_testing_data['Season'][i]  
        

# %%
