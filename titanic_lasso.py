import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# import data
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

all = pd.concat((train.loc[:,'Pclass':'Embarked'],test.loc[:,'Pclass':'Embarked']))

# Embarked, C has high survival rate
plt.figure()
sns.countplot(x='Embarked',data=train)
sns.countplot(x='Survived',hue='Embarked',data=train)
sns.factorplot(x='Embarked',y='Survived',data=train,order=['C','Q','S'])
all.Embarked.fillna('C',inplace=True)
all.Embarked = pd.Categorical(all.Embarked).codes

# Sex, female has high survival rate
sns.factorplot(x='Sex',y='Survived',data=train)
all.Sex = pd.Categorical(all.Sex).codes

# Pclass, class1 has high survival rate
sns.countplot(x='Pclass',data=train)
sns.factorplot(x='Pclass',y='Survived',data=train)

# Parch, Sib
all['Family'] = all['SibSp'] + all['Parch']
all.Family[all.Family>0]=1
all.drop(['Parch','SibSp'],axis=1,inplace=True)

# Cabin
all.Cabin[all.Cabin.isnull()]='0'
all['CabinClass'] = all.Cabin.str[0]
all.CabinClass.value_counts()
all.CabinClass[all.CabinClass!='0']='1'
all.drop('Cabin',axis=1,inplace=True)

# Age
all.Age.fillna(train.Age.mean(),inplace=True)
age_bins = [0,10,50,1000]
all['Age_cat'] = pd.cut(all.Age, bins = age_bins)
all.Age_cat = pd.Categorical(all.Age_cat).codes

# Fare
test[(test.Pclass==3)&(test.Embarked=='S')].Fare.hist(bins=100)
test[(test.Pclass==3)&(test.Embarked=='S')].Fare.value_counts().head()
all.Fare.fillna('8.05',inplace=True)

all.drop(['Name','Ticket'], axis =1, inplace=True)

# define X,y
X_train = all[:train.shape[0]]
y = train.Survived
X_test = all[train.shape[0]:]

# model
model = RandomForestClassifier(random_state =1, n_estimators=120, min_samples_split=3,min_samples_leaf=2)
model.fit(X_train,y)
print model.score(X_train, y)
score=cross_val_score(model, X_train, y, cv=5).mean()
print score

# predict
y_pred = model.predict(X_test)
solution = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':y_pred})
solution.to_csv('titanic.csv',index=False)

