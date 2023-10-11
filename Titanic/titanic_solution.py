import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier

# set the directory
os.chdir('/Users/lucabegatti/Desktop/Pycharm/Kaggle/Titanic')
# import the data
train = pd.read_csv('train.csv').drop('Name', axis=1)
test = pd.read_csv('test.csv').drop('Name', axis=1)

train['Ticket_number'] = train['Ticket'].apply(lambda x: x.split(" ")[-1])
train['Ticket_item'] = train['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])
test['Ticket_number'] = test['Ticket'].apply(lambda x: x.split(" ")[-1])
test['Ticket_item'] = test['Ticket'].apply(lambda x: 'NONE' if len(x.split(" ")) == 1 else x.split(" ")[0])
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)

y_train = train['Survived']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})
output.to_csv('submission.csv', index=False)
