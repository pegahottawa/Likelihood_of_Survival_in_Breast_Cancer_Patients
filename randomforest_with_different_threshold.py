#!/usr/bin/env python
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error, r2_score
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

#df300 = df200.transpose()
#df300.columns = df300.iloc[0]
#df400 = df300.drop(df300.index[[0]])
#df600=df400.reset_index()
#df600.columns = df2000.iloc[:,0]


df3.columns = df3.iloc[0]  # after trasposing a df2 column names become the first row ,thats why I renamed the column names
                           # with the firts row
df4 = df3.drop(df3.index[[0]])     # after that because by renaming I would end up with same
                                   # column names and and same row values,I droped the first row

df6=df4.reset_index()              # I wanted to have dataframe format with all index for each row
df6.columns = df100.iloc[:,0]      # After all, now its time to rename columns
result = pd.merge(df1,df6,on='METABRIC_ID')


NON_value = result.isnull().sum()   # how many NON value we have in each column
del_NON = result.dropna(how = 'any')  # drop row if you find any NON value in it

Tar = del_NON[del_NON.columns[6]]
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values

feature_lst = list(Descriptor.columns.values)
#print feature_list

X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (554,1))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

#######################################################
# Feature importance
#######################################################

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0 )
clf = clf.fit(X_train, y_train)

##################################################################
# New random forest with only most important variables
##################################################################

c =[ 0.0001,0.0004,0.0007,0.001,0.0013,0.0016,0.0019,0.0022,0.0025,0.0028,0.0031]

#sfm = SelectFromModel(clf, threshold=0.0034)
#sfm.fit(X_train, y_train)
#print "12"
#numberfeature=[]
#for feature_list_index in sfm.get_support(indices=True):
#   numberfeature.append(feature_list_index)
#print len(numberfeature)



for i in c:
   print "threshold", i
   sfm = SelectFromModel(clf, threshold=i)
   sfm.fit(X_train, y_train)
   X_important_train = sfm.transform(X_train)
   X_important_test = sfm.transform(X_test)
   clf_important = RandomForestClassifier(n_estimators=10,random_state=0)
   clf_important.fit(X_important_train, y_train)
   y_important_pred = clf_important.predict(X_important_test)
   print "Accuracy is ", accuracy_score(y_test, y_important_pred) * 100



