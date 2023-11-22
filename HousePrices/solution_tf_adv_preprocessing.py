import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

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
distplot_color = "#6A0572"
columns_per_row = 5
num_cols = len(dataset.columns)
num_rows = (num_cols + columns_per_row - 1) // columns_per_row
# Create a figure and an array of subplots
fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(12, 3 * num_rows))
axes = axes.flatten()

# Plot distplots for each column
for i, column in enumerate(dataset.columns):
    ax = axes[i]
    sns.histplot(dataset[column], ax=ax, color=distplot_color)
    ax.set_title(column)

# Remove any empty subplots (if the number of columns isn't a multiple of 3)
for i in range(num_cols, num_rows * columns_per_row):
    fig.delaxes(axes[i])

# Adjust subplot layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
