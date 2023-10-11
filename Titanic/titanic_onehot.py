import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/Titanic')
# import the data
train = pd.read_csv('train.csv').drop(['Name'], axis=1)
test = pd.read_csv('test.csv').drop(['Name'], axis=1)

train['Ticket_number'] = train['Ticket'].apply(lambda x: x.split(" ")[-1])
train['Ticket_item'] = train['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])
test['Ticket_number'] = test['Ticket'].apply(lambda x: x.split(" ")[-1])
test['Ticket_item'] = test['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])

# before the encoding
X_train = train.drop(['Ticket', 'Ticket_item', 'PassengerId', 'Survived'], axis=1)
X_test = test.drop(['Ticket', 'Ticket_item', 'PassengerId'], axis=1)
y_train = train['Survived']

# columns we want to encode
categorical_cols = ['Ticket_number',
                    # 'Ticket_item',
                    'Sex',
                    'Cabin',
                    'Embarked']

# OneHot Encoding
enc = OneHotEncoder(handle_unknown='ignore')

# train data encoding
enc_train = enc.fit(X_train[categorical_cols])
train_oneHot_array = enc.transform(X_train[categorical_cols]).toarray()
# transform it to df
train_data_hot_encoded = pd.DataFrame(train_oneHot_array, index=X_train.index)
# extract the columns that did not need encoding
train_data_not_encoded = X_train.drop(columns=categorical_cols)
X_train_encoded = pd.concat([train_data_hot_encoded, train_data_not_encoded], axis=1)

## test data encoding
test_oneHot_array = enc.transform(X_test[categorical_cols]).toarray()

# transform it to df
test_data_hot_encoded = pd.DataFrame(test_oneHot_array, index=X_test.index)
# extract the columns that did not need encoding
test_data_not_encoded = X_test.drop(columns=categorical_cols)

X_test_encoded = pd.concat([test_data_hot_encoded, test_data_not_encoded], axis=1)

model = XGBClassifier()
model.fit(X_train_encoded, y_train)
pred = model.predict(X_test_encoded)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
output.to_csv('submissionOneHot7.csv', index=False)
