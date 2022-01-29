####Importing Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

####Loading CSV File
credit_card_data = pd.read_csv(r"C:\Users\prave\Downloads\creditcard.csv")
print(credit_card_data.head())
print(credit_card_data.tail())

####Dataset Information
print(credit_card_data.info())

####Checking for Missing Values
print(credit_card_data.isnull().sum())

####Distribution of Legit Transaction & Fraudulent Transaction
print(credit_card_data['Class'].value_counts())
## 0 --->  Normal Transaction   1 ---> Fraudulent Transaction

####Separating the data for analysis
legit = credit_card_data[credit_card_data.Class==0]
fraud = credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)

####Measure of the Data(Amount)
print(legit.Amount.describe())
print(fraud.Amount.describe())

####Compare the values for both Transactions
print(credit_card_data.groupby('Class').mean())

####Under Sampling (Build a sample dataset containing similar distribution of Normal and Fraudulent transactions) No. of Fraudulent transactions = 492
legit_sample = legit.sample(n=492)

####Concating to D0ataFrames(axis=0 means row and axis=1 means column)
new_dataset= pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

####Splitting Data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

####Split the data into Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

####Model Training(Logistic Regression)
model = LogisticRegression()
##training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

####Model Evalution(Accuracy Score)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data : ', training_data_accuracy)

#### Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)