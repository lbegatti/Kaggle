import os

import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
missing_data.head(36)

dataset = dataset.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                        'LotFrontage'], axis=1)

## Encoding
encoder = ce.HashingEncoder(n_components=2, return_df=True)
dataset = encoder.fit_transform(dataset)

# after encoding the remaining NaN will be handled with the mean value
simple_impute = SimpleImputer(strategy='mean', missing_values=np.nan)
dataset2 = simple_impute.fit_transform(dataset)
dataset_final = pd.DataFrame(dataset2, columns=dataset.columns)
dataset_final.tail()

# separate dataset test
dataset_test_final = dataset_final[dataset_final['col'] == 0]
dataset_test_final = dataset_test_final.drop(['SalePrice'], axis=1)
dataset_test_final.tail()

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

# hyperparameter Tuning