import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
warnings.simplefilter("ignore")

# Load the train dataset
train = pd.read_csv('train.csv')
print(train.head())
print("\n")
print(train.columns.values)
train.isna().head()

print("NULL values: ")
print(train.isna().sum())
print("\n")

# Handling null values
train.fillna(train.mean(), inplace=True)
print(train.isna().sum())

train.info()
# removing the features not correlated to the target class

train = train.drop(['MSZoning'], axis=1)

# encoding the categorical features


labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train['LotShape'])
train['LotShape'] = labelEncoder.transform(train['LotShape'])
train.info()
print(train.head())

# Fitting SVM model to the data
feature_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual']


X = train[feature_cols]
y = train.SaleCondition
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm = SVC()
expected = train.SaleCondition
# Training the Model

svm.fit(X_train, y_train)
print(expected)

predicted_label = svm.predict(train[feature_cols])
print(predicted_label)
print(metrics.confusion_matrix(expected, predicted_label))
print(metrics.classification_report(expected, predicted_label))

# Evaluating the Model based
print('Accuracy of SVM classifier : {:.2f}'
     .format(svm.score(X_test, y_test)))

#calculating navie bayes model


model = GaussianNB()
model.fit(train[feature_cols], train.SaleCondition)
expected = train.SaleCondition            # Making Prediction based on X, Y
predicted = model.predict(train[feature_cols])


print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))
X_train, X_test, Y_train, Y_test = train_test_split(train[feature_cols], train.SaleCondition, test_size=0.2, random_state=0)

model.fit(X_train, Y_train)

Y_predicted = model.predict(X_test)

print("accuracy using Gaussian navie bayes Model is ", metrics.accuracy_score(Y_test, Y_predicted) * 100)

logreg = LogisticRegression()

# fit the model with data

logreg.fit(X, y)
logreg.predict(X)
y_pred = logreg.predict(X)
len(y_pred)

#knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print("accuracy using knn Model is ",metrics.accuracy_score(y, y_pred))