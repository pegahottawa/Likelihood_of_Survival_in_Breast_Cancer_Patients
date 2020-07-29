#!/usr/bin/env python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
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
df_merge = pd.merge(df1,df6,on ='METABRIC_ID')

NON_value = df_merge.isnull().sum()   # how many NON value we have in each column
del_NON = df_merge.dropna(how = 'any')  # drop rows with NON value

#####################################
# Target and descriptor selection
#####################################

Target = del_NON[del_NON.columns[6]]
Descriptor = del_NON.drop(del_NON.columns[[0,1,2,6]], axis=1)
arr_descriptors = Descriptor.values
arr_Target = Target.values

feature_lst = list(Descriptor.columns.values)
X = arr_descriptors
T = arr_Target
Target_reshape = np.reshape(T, (533,))
Y = Target_reshape

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

######################################################################
# Selecting features using Lasso regularisation using SelectFromModel
######################################################################

X_train,X_test,y_train,y_test=train_test_split(X, Y,random_state=0)

lambda =[1e-2,2,5,10,20,30,40,50,60,70,80,90,100,150,200]  # increasing the penalisation Lambda will increase the number of features removed

for i in lambda:
   sel= SelectFromModel(LogisticRegression(C = i, penalty ='l1'))  # L1 regularisation shrink feature coefficients to zero
   sel.fit(X_train, y_train)
   X_important_train = sel.transform(X_train)   # Transform the data to both training and test to create a new dataset containing only the most important features
   X_important_test = sel.transform(X_test)
   sel_important = LogisticRegression()
   scores =cross_val_score( sel_important , X_train, y_train, cv=5)
   print scores
   print("Average 5-Fold CV Score: {}".format(np.mean(scores)))
   print('features with coefficients shrank to zero: {}'.format(np.sum(sel.estimator_.coef_ == 0)))  # Totall number of features which coefficient was shrank to zero

# print the names of the most important features
for feature_list_index in sel.get_support(indices=True):  
   print(feature_lst[feature_list_index])
   

