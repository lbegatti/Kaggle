import os

import pandas as pd
import tensorflow_decision_forests as tfdf

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/Titanic')
# import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['Name'] = train['Name'].apply(lambda x: " ".join([v.strip(",()[].\"'") for v in x.split(" ")]))
train['Ticket_number'] = train['Ticket'].apply(lambda x: x.split(" ")[-1])
train['Ticket_item'] = train['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])

test['Name'] = test['Name'].apply(lambda x: " ".join([v.strip(",()[].\"'") for v in x.split(" ")]))
test['Ticket_number'] = test['Ticket'].apply(lambda x: x.split(" ")[-1])
test['Ticket_item'] = test['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])

input_features = list(train.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="Survived")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test)

# DEFAULT PARAMETERS
model = tfdf.keras.GradientBoostedTreesModel(verbose=0,
                                             features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
                                             exclude_non_specified_features=True,
                                             random_seed=1234
                                             )
model.fit(train_ds)

self_evaluation = model.make_inspector().evaluation()
print(f"Accuracy: {self_evaluation.accuracy} Loss: {self_evaluation.loss}")

prob_survival = model.predict(test_ds, verbose=0)[:, 0]
pred = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (prob_survival >= 0.5).astype(int)
})
pred.to_csv('submissionTF.csv', index=False)

## HYPERPARAMETER TUNNING
tuner = tfdf.tuner.RandomSearch(num_trials=1000)
tuner.choice("min_examples", [2, 5, 7, 10])
tuner.choice("categorical_algorithm", ["CART", "RANDOM"])

local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [3, 4, 5, 6, 8])

global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

tuner.choice("shrinkage", [0.02, 0.05, 0.10, 0.15])
tuner.choice("num_candidate_attributes_ratio", [0.2, 0.5, 0.9, 1.0])

tuner.choice("split_axis", ["AXIS_ALIGNED"])
oblique_space = tuner.choice("split_axis", ["SPARSE_OBLIQUE"], merge=True)
oblique_space.choice("sparse_oblique_normalization", ["NONE", "STANDARD_DEVIATION", "MIN_MAX"])
oblique_space.choice("sparse_oblique_weights", ["BINARY", "CONTINUOUS"])
oblique_space.choice("sparse_oblique_num_projections_exponent", [1.0, 1.5])

# let's apply our tuning
tuned_model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
tuned_model.fit(train_ds, verbose=0)
tuned_self_evaluation = tuned_model.make_inspector().evaluation()
print(f"Accuracy: {tuned_self_evaluation.accuracy} Loss:{tuned_self_evaluation.loss}")

prob_survival = tuned_model.predict(test_ds, verbose=0)[:, 0]
pred = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": (prob_survival >= 0.5).astype(int)
})
pred.to_csv('submission_hyp_tuning_TF.csv', index=False)
