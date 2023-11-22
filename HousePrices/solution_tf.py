import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/HousePrices')
# import the data
train = pd.read_csv('train.csv').drop("Id", axis=1)

# check how the numerical columns are distributed
list(set(train.dtypes.to_list()))

train_numerical_cols = train.select_dtypes(include=['int64', 'float64'])
train_numerical_cols.head()
train_numerical_cols.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)


# split into training and test
def split_dateset(dataset, test_ratio=0.3):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, valid_ds_ps = split_dateset(train)

# convert into TensorFlowDatasets
label = "SalePrice"
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_ps, label=label, task=tfdf.keras.Task.REGRESSION)

# tfdf.keras.get_all_models()

# create a RandomForest
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics="mse")  # optional, you can include other eval metrics

# train the model
rf.fit(x=train_ds)

# visualize the model
# tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=2, max_depth=3)

# Evaluate the model with "out of bag" OoB data and validation dataset (we created this before)
logs = rf.make_inspector().training_logs()  # show the RMSE evaluated on the OoB dataset
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

# general stats on the OoB dataset
inspector = rf.make_inspector()
inspector.evaluation()
# evaluate the validation dataset
evaluation = rf.evaluate(x=valid_ds, return_dict=True)

# variable importance
for importance in inspector.variable_importances().keys():
    print("\t", importance)

inspector.variable_importances()["NUM_AS_ROOT"]

# plot the variable importances
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"  # we want to see which features are mostly used as root node in most trees.
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
#
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# TODO: Replace with "plt.bar_label()" when available.
# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()

# submission
test = pd.read_csv('test.csv')
ids = test.pop('Id')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test, task=tfdf.keras.Task.REGRESSION)
pred = rf.predict(test_ds)
output = pd.DataFrame({'Id': ids,
                       'SalePrice': pred.squeeze()})
output.head()

output.to_csv('submission_houseprices_tf.csv', index=False)
