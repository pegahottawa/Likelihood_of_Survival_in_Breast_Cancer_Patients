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
column_names = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  
df4 = df3.drop(df3.index[[0]])     

df6 = df4.reset_index()              
df6.columns = column_names.iloc[:,0]      
df_merge = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = df_merge.isnull().sum()   
del_NON = df_merge.dropna(how = 'any')  

Target = del_NON[del_NON.columns[6]]
Descriptor = del_NON.drop(del_NON.columns[[0,2,6]], axis=1)
arr_descriptors = Descriptor.values
arr_Target = Target.values

feature_lst = list(Descriptor.columns.values)
#print feature_list

X = arr_descriptors
T = arr_Target
Target_reshape = np.reshape(T, (554,1))
Y = Target_reshape

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0 )
clf = clf.fit(X_train, y_train)

############################################################
# random forest with only most important variables
#############################################################

# After calculating the averge importance of all the features, we can build a list with different threshold value to choose differnt subset of features 
#and check the performance of the model

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



