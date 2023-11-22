import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/HousePrices')
# import the data
train = pd.read_csv('train.csv').drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test = pd.read_csv('test.csv').drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

train_notarget = train.drop('SalePrice', axis=1)

num_col = train_notarget._get_numeric_data().columns
cat_cols = list(set(train_notarget.columns) - set(num_col))

enc = OneHotEncoder(handle_unknown='ignore')

# train data encoding
enc_train = enc.fit(train_notarget[cat_cols])
train_onehot_array = enc.transform(train_notarget[cat_cols]).toarray()
# transform to df
train_data_onehot_df = pd.DataFrame(train_onehot_array, index=train_notarget.index)
# extract cols with no encoding
train_data_not_encoded = train_notarget.drop(columns=cat_cols)

# test data encoding
test_onehot_array = enc.transform(test[cat_cols]).toarray()
# transform to df
test_data_onehot_df = pd.DataFrame(test_onehot_array, index=test.index)
# extract cols with no encoding
test_data_not_encoded = test.drop(columns=cat_cols)

# ready for ML
x_train = pd.concat([train_data_onehot_df, train_data_not_encoded], axis=1)
y_train = train['SalePrice']
x_test = pd.concat([test_data_onehot_df, test_data_not_encoded], axis=1)

model = XGBRegressor()
model.fit(x_train, y_train)
pred = model.predict(x_test)

output = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})
output.to_csv('housepriceOHE.csv', index=False)
