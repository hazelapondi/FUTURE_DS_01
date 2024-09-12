#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#load the file into a pandas DataFrame
train = pd.read_csv('datasets/titanic_train.csv')

#display the first 5 rows of the DataFrame
train.head()

#Explore our DataFrame to understand its structure
train.describe()

train.shape

#find missing values in the dataset
train.isnull()

#alternatively we can visualize using a heat map. Yellow areas represent missing values
sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')

#Countplot of Survived column
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data = train)

#survival count based on gender
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data = train, palette = 'RdBu_r')
plt.title('Survival Rate by Gender')
plt.show()

#survival count based on passenger class (Pclass)
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'rainbow')
plt.title('Survival rate by Passenger Class')
plt.show()

#Age distribtion of passengers
sns.distplot(train['Age'].dropna(), kde = False, color = 'darkred', bins = 40)
plt.title('Distribution of Age')
plt.show()

#Countplot of Siblings/Spouses aboard
sns.countplot(x = 'SibSp', data = train)

#Fare distribution
train['Fare'].hist(color = 'green', bins = 40, figsize = (8,4))
plt.title('Fare Distribution')
plt.show()

#Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#check the average Age by Passenger Class using a boxplot
plt.figure(figsize=(12,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train, palette = 'winter')

#replace missing (null) Age values
def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]

  if pd.isnull(Age):
    if Pclass == 1:
      return 37
    elif Pclass == 2:
      return 29
    else: 
      return 24
  else:
    return Age

#apply the function impute_age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)

#check heat map again to establish missing values 
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

#drop the Cabin column and the rows with value NaN
train.drop('Cabin', axis = 1, inplace = True)

train.head()

train.dropna(inplace = True)

#convert categorical features to dummy variables
train.info()

pd.get_dummies(train['Embarked'], drop_first = True).head()
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embark'], drop_first = True)

train.drop(['Sex','Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train = pd.concat([train,sex,embark], axis = 1)
train.head()

#split data into a training set and a test set
train.drop('Survived', axis = 1).head()

train['Survived'].head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trian.drop('Survived', axis = 1),
                                                    train['Survived'], test_size = 0.30,
                                                    random_state = 101)

#Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#making predictions
predictions = logmodel.predict(X_test)

#confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)

predictions
