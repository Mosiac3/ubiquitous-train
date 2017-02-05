import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm,datasets
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# import iris data
# X shape=(150,4)
# y ~ [0,1,2]
iris = datasets.load_iris()
X = iris.data
y = iris.target

# binarize the output
y = label_binarize(y,classes=[0,1,2])
n_classes = y.shape[1]

# add noisy features to make the problem harder
# X.shape is (150,804)
random_state = np.random.RandomState(0)
n_samples,n_features = X.shape
X = np.c_[X,random_state.randn(n_samples,200*n_features)]

# shuffle and split training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.5,random_state = 0)

# predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear',probability=True,random_state=random_state))
y_score = classifier.fit(X_train,y_train).decision_function(X_test)

# compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
  fpr[i],tpr[i],_ = roc_curve(y_test[:,i],y_score[:,i])
  roc_auc[i] = auc(fpr[i],tpr[i])

# compute micro-average ROC curve and AUC
fpr['micro'],tpr['micro'],_ = roc_curve(y_test.ravel(),y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'],tpr['micro'])

# plot a ROC curve for class 2
plt.figure()
lw = 2
plt.plot(fpr[2],tpr[2],color='darkorange',lw=lw,label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiving operating characteristic example')
plt.legend(loc='lower right')

# plot ROC curves for the multiclass problem
# compute macro-average ROC curve and AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
  mean_tpr += interp(all_fpr,fpr[i],tpr[i])
 
mean_tpr /= n_classes

fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'],tpr['macro'])

# plot all ROC curves
plt.figure()
plt.plot(fpr['micro'],tpr['micro'],label='micro-average ROC curve (area = {0:0.2f})'''.format(i,roc_auc['micro']),color='deeppink',linestyle=':',linewidth=4)
plt.plot(fpr['macro'],tpr['macro'],label='macro-average ROC curve (area = {0:0.2f})'''.format(i,roc_auc['macro']),color='navy',linestyle=':',linewidth=4)
colors = cycle(['aqua','darkorange','cornflowerblue'])
for i,color in zip(range(n_classes),colors):
  plt.plot(fpr[i],tpr[i],color=color,lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i,roc_auc[i]))

plt.plot([0,1],[0,1],'k--',lw=lw)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc='lower right')
plt.show()

