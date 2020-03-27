#!/usr/bin/env python
from sklearn.svm import SVC
import itertools
from sklearn.grid_search import GridSearchCV
from sklearn import svm, grid_search
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics, cross_validation
from sklearn.model_selection import cross_validate
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import math

path_to_file = '/home/pegah/project_discovery/clinical_cleaned_dead_alive.xlsx'
df1 = pd.read_excel(path_to_file)

path_to_file = '/home/pegah/project_discovery/gene_ex.xlsx'
df2 = pd.read_excel(path_to_file)
df3 = df2.transpose()

path_to_file = '/home/pegah/project_discovery/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
df100 = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  # after trasposing a df2 column names become the first row ,thats why I renamed the column names with the firts row
df4 = df3.drop(df3.index[[0]])     # after that because by renaming I would end up with same
                                   # column names and and same row values,I droped the first row

df6=df4.reset_index()              # I wanted to have dataframe format with all index for each row
df6.columns = df100.iloc[:,0]      # After all, now its time to rename columns
result = pd.merge(df1,df6,on='METABRIC_ID')

del_NON = result.dropna(how = 'any')  # drop row if you find any NON value in it

Tar = del_NON[del_NON.columns[6]]
Descriptor = del_NON.drop(del_NON.columns[[0,1,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values


X = arr_descriptors
T = arr_Target
#Target = np.reshape(T, (609,1)) # when not using Kfold cv and using Holdout cv method
Target = np.reshape(T, (533,)) # when using kfold cross validation you need this sort of reshape
Y = Target


print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

#######################################################
#Applying svm classifier
#######################################################

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#param_grid = {'C':[0.001, 0.010, 0.100,1.000, 10, 100, 1000],'gamma':[0.001, 0.010, 0.100,1.000, 10, 100, 1000], 'kernel':['linear','rbf']}


#clf = GridSearchCV(svm.SVC(), param_grid, verbose=1)
#clf.fit(X_train, y_train)
#print('Best parameters:',clf.best_params_)

reg = svm.SVC(kernel='linear', C=0.001,gamma=0.001)
scores = cross_val_score( reg, X_train, y_train, cv=5)

print " result for svmcleand "
print scores
print("Average 4-Fold CV Score: {}".format(np.mean(scores)))
print "svm"

