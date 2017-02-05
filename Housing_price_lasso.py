import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

def loaddata():
  """ import data """
  train = pd.read_csv('../train.csv')
  test = pd.read_csv('../test.csv')
  return train, test

def plot_import_vars(coef, n=5):
  imp_coef = pd.concat([coef.sort_values().head(n),coef.sort_values().tail(n)])
  imp_coef.plot(kind = 'bar',figsize=(10,8))
  plt.xticks(rotation=20)
  plt.show()

def fillna_cat(train):	
  cols = train.dtypes[train.dtypes =='object'].index
  for col in cols:
	if train[col].isnull().sum()!= 0:
	  train.loc[train[col].isnull(),col]='None'
  return train

def preprocess(train,test):
  alldata = pd.concat([train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']])
  
  # fill missing categorical data
  fillna_cat(alldata)
  
  # remove low variance categorical features
  cols = alldata.dtypes[alldata.dtypes =='object'].index
  for col in cols:
	tmp = ((alldata[col].value_counts()/alldata.shape[0])**2).sum()
	if tmp > 0.9:
	  del alldata[col]

  # transform numeric data with skewness larger than 0.5 to its log form
  alldata = alldata.fillna(alldata.mean())
  num = alldata.dtypes[alldata.dtypes != object].index
  num_skew = alldata[num].apply(lambda x: skew(x))
  num_skew = num_skew[num_skew > 0.5]
  num_skew = num_skew.index
  alldata[num_skew] = np.log1p(alldata[num_skew])

  # transform alldata categorical data to numerical
  alldata = pd.get_dummies(alldata)
  return alldata


if __name__ == '__main__':
  train, test = loaddata()
  alldata = preprocess(train,test)
  
  # prepare features
  X_train = alldata.iloc[:train.shape[0], :]
  X_test = alldata.iloc[train.shape[0]:, :]

  # prepare target
  train['SalePrice_log'] = np.log1p(train['SalePrice'])
  y = train.SalePrice_log

  # model
  alphas = np.linspace(0.0001,0.001,100)
  cv = 5 
  model_lasso = LassoCV(alphas=alphas, cv=5)
  res=model_lasso.fit(X_train, y)
  score=cross_val_score(model_lasso, X_train, y, cv=cv).mean()
  coef = pd.Series(model_lasso.coef_,index=X_train.columns)
  print 'Lasso has chosen alpha to be %f.' % (res.alpha_) 
  print 'The cross validation score is %f.' % (score)

  # plot the most significant 10 features
  plot_import_vars(coef, 5)

  # prediction
  preds = np.expm1(model_lasso.predict(X_test))
  solution = pd.DataFrame({'id':test.Id,'SalePrice':preds})
  solution.to_csv('house_price.csv',index=False)

# to do:
# fill missing value by distribution
# create new features
# more samples
