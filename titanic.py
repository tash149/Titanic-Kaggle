import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
result = pd.read_csv('gender_submission.csv')
X_train = train.iloc[:,2:12].values
X_train = np.delete(X_train, 1, axis=1)  #Removing the Names column from X
X_train = np.delete(X_train, 7, axis=1)  #Removing Cabin no.
X_train = np.delete(X_train, 5, axis=1)  #Removing ticket
X_train = np.delete(X_train, 5, axis=1)  #Removing fare
X_train = np.delete(X_train, 5, axis=1)  #Removing Embarkation
y_train = train.iloc[:, 1].values


#Test set
X_test = test.iloc[:,1:11].values
X_test = np.delete(X_test, 1, axis=1)  #Removing the Names column from X
X_test = np.delete(X_test, 7, axis=1)  #Removing Cabin no.
X_test = np.delete(X_test, 5, axis=1)  #Removing ticket
X_test = np.delete(X_test, 5, axis=1)  #Removing fare
X_test = np.delete(X_test, 5, axis=1)  #Removing Embarkation
# NO dependent variable in test set y_test = test.iloc[:, 1].values 

y_test = result.iloc[:, 1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer_train = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_train = imputer_train.fit(X_train[:, 2:3])
X_train[:, 2:3] = imputer_train.transform(X_train[:, 2:3])

imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_tr = LabelEncoder()
X_train[:, 1] = labelencoder_tr.fit_transform(X_train[:, 1])
onehotencoder_tr = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder_tr.fit_transform(X_train).toarray()

labelencoder_te = LabelEncoder()
X_test[:, 1] = labelencoder_te.fit_transform(X_test[:, 1])
onehotencoder_te = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder_te.fit_transform(X_test).toarray()

# Avoiding Dummy Variable trap
X_train = X_train[: , 1:]
X_test = X_test[: , 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

