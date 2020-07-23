#!/usr/bin/env python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
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
from sklearn import preprocessing

path_to_file = '/home/pegah/project_discovery/clinical_cleaned_dead_alive.xlsx'
df1 = pd.read_excel(path_to_file)

path_to_file = '/home/pegah/project_discovery/gene_ex.xlsx'
df2 = pd.read_excel(path_to_file)
df3 = df2.transpose()

path_to_file = '/home/pegah/project_discovery/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
column_names = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  
df4 = df3.drop(df3.index[[0]])     

df6=df4.reset_index()             
df6.columns = columns_names.iloc[:,0]     
df_merge = pd.merge(df1,df6,on='METABRIC_ID')
#print 'result.shape =', result.shape

del_NON = df_merge.dropna(how = 'any')  # drop rows with any NON value
#print del_NON
print 'del_NON.shape =', del_NON.shape

Target = del_NON[del_NON.columns[6]]
#print Target
Descriptor = del_NON.drop(del_NON.columns[[0,1,2,6]], axis=1)
#print Descriptor
#print 'Descriptor.shape =', Descriptor.shape
arr_descriptors=Descriptor.values
arr_Target=Target.values

X = arr_descriptors
T = arr_Target
#Target = np.reshape(T, (554,1)) # when not using Kfold cv and using Holdout cv method
Target_reshape = np.reshape(T, (533,)) 
Y = Target_reshape

X = preprocessing.scale(X)
Y = preprocessing.scale(Y)

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

#######################################################
#Applying Logistic classifier
#######################################################

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X, Y,random_state=0)   #random_state=0  obtain the
#same split everytime you run your script , dose not shuffle order of test and train set .

print 'X_train.shape =', X_train.shape
print 'y_train.shape =', y_train.shape

logreg = LogisticRegression()
#logreg.fit(X_train,y_train)
#y_pred=logreg.predict(X_test)
scores =cross_val_score( logreg, X_train, y_train, cv=5)
print scores
print("Average 5-Fold CV Score: {}".format(np.mean(scores)))
print"logistic cleaned "






