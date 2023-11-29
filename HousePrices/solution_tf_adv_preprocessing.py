import os

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras import Sequential
from keras import layers
from keras.layers import Dense, Dropout

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/HousePrices')
# import the data
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

# combine in 1 dataset
dataset = pd.concat([dataset_train.assign(col=1), dataset_test.assign(col=0)])
dataset = dataset.drop(['Id'], axis=1)

## Data Visualization
hist_col = '#3498db'
plt.figure(figsize=(10, 6))
sns.histplot(dataset['SalePrice'], bins=20, color=hist_col, kde=True)
plt.xlabel('Sale Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().set_axisbelow(True)
plt.show()

# Dist Plot
# distplot_color = "#6A0572"
# columns_per_row = 5
# num_cols = len(dataset.columns)
# num_rows = (num_cols + columns_per_row - 1) // columns_per_row
# Create a figure and an array of subplots
# fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(12, 3 * num_rows))
# axes = axes.flatten()

# Plot distplots for each column
# for i, column in enumerate(dataset.columns):
#    ax = axes[i]
#    sns.histplot(dataset[column], ax=ax, color=distplot_color)
#    ax.set_title(column)

# Remove any empty subplots (if the number of columns isn't a multiple of 3)
# for i in range(num_cols, num_rows * columns_per_row):
#    fig.delaxes(axes[i])

# Adjust subplot layout for better spacing
# plt.tight_layout()

# Show the plot
# plt.show()


## Missing Data
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# missing_data.head(36)

dataset = dataset.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                        'LotFrontage'], axis=1)

## Encoding
encoder = ce.HashingEncoder(n_components=2, return_df=True)
dataset = encoder.fit_transform(dataset)

# after encoding the remaining NaN will be handled with the mean value
simple_impute = SimpleImputer(strategy='mean', missing_values=np.nan)
dataset2 = simple_impute.fit_transform(dataset)
dataset_final = pd.DataFrame(dataset2, columns=dataset.columns)
# dataset_final.tail()

# separate dataset test
dataset_test_final = dataset_final[dataset_final['col'] == 0]
dataset_test_final = dataset_test_final.drop(['SalePrice'], axis=1)
# dataset_test_final.tail()

# train dataset split
dataset_train_final = dataset_final[dataset_final['col'] == 1]
X = dataset_train_final.drop('SalePrice', axis=1)
y = dataset_train_final['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# scale features in X_train and X_test
scaler = MinMaxScaler()
# all features are btw 0/1
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X.columns)

# correlation matrix
corrmat = X.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, cbar=True, square=True)

# create a RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_

importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# remove the features with little importance
X_train = X_train.drop(['col', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'PoolArea', 'EnclosedPorch', 'KitchenAbvGr',
                        'BsmtFullBath', 'BsmtFinSF2', 'HalfBath'], axis=1)
X_test = X_test.drop(['col', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'PoolArea', 'EnclosedPorch', 'KitchenAbvGr',
                      'BsmtFullBath', 'BsmtFinSF2', 'HalfBath'], axis=1)

dataset_test_final = dataset_test_final.drop(
    ['col', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'PoolArea', 'EnclosedPorch', 'KitchenAbvGr',
     'BsmtFullBath', 'BsmtFinSF2', 'HalfBath'], axis=1)

# Finding the best model via Deep Learning (Hyperparameter Tuning)
model_tf = Sequential()
model_tf.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model_tf.add(Dropout(0.2))
model_tf.add(Dense(64, activation='relu'))
model_tf.add(Dense(1, activation='linear'))  # linear activation for regression

model_tf.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

best_model = None
best_mse = float('inf')

for epochs in [50, 100, 200]:
    for batch_size in [32, 64, 128]:

        # train the model on the training data
        history = model_tf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

        # evaluate the model on the validation data
        val_mse = mean_squared_error(y_test, model_tf.predict(X_test))

        # check if the model has the best performance
        if val_mse < best_mse:
            best_model = model_tf
            best_mse = val_mse

print("Best Model MSE:", best_mse)

test_mse = mean_squared_error(y_test, best_model.predict(X_test))
print("Test MSE:", test_mse)

# use the best model to make prediction on the test data
dl_predictions = best_model.predict(dataset_test_final)
dl_predictions_df = pd.DataFrame({'Id': range(1461, 1461 + len(dl_predictions)),
                                  'SalePrice': dl_predictions.flatten()})

# Apply RF Regression and doing hyperparameter tuning
RF = RandomForestRegressor()
RF.fit(X_train, y_train)
scores = cross_val_score(RF, X_train, y_train, cv=10)
scores.mean()

params = {"n_estimators": [100, 200, 300, 400, 500, 1000],  # number of trees in the forest
          "max_features": ['sqrt', 'log2'],  # how to decide max number of features when looking for the best split
          "max_depth": [None, 10, 20, 30, 40, 50, 100],  # max depth of the tree
          "min_samples_split": [2, 5, 10, 20, 50],  # min num of samples required to split an internal node
          "min_samples_leaf": [1, 2, 4, 8],  # min number of samples required to be at a leaf node
          "bootstrap": [True, False]  # whether to bootstrap samples when building trees
          }

random_search = RandomizedSearchCV(
    estimator=RF,
    param_distributions=params,
    n_iter=10,  # number of random combinations to try
    cv=10,  # number of cv folds
    scoring="neg_mean_squared_error",  # scoring metrics
    n_jobs=-1,  # use all cpu cores
    random_state=42
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("\nBest Hyperparameters:", best_params)

best_model_RF = random_search.best_estimator_
print("\nBest model :", best_model_RF)

# initalize a new RF with the optimal parameters
RF_final = RandomForestRegressor(max_depth=20, max_features='sqrt', n_estimators=400, min_samples_split=5,
                                 min_samples_leaf=1, bootstrap=False)
RF_final.fit(X_train, y_train)
scores_final = cross_val_score(RF_final, X_train, y_train, cv=10)
print(scores_final)
print("\n mean of the Scores :", scores_final.mean())

# predictions

rf_predictions = RF_final.predict(dataset_test_final)
rf_predictions_df = pd.DataFrame({"Id": range(1461, 1461 + len(rf_predictions)),
                                  "SalePrice": rf_predictions
                                  })
rf_predictions_df.to_csv('rf_predictions.csv', index=False)