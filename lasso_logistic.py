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
df100 = pd.read_excel(path_to_file)

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
Descriptor = del_NON.drop(del_NON.columns[[0,1,2,6]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values

feature_lst = list(Descriptor.columns.values)

X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (533,))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape


X_train,X_test,y_train,y_test=train_test_split(X, Y,random_state=0)


#a =[1e-2,2,5,10,20,30,40,50,60,70,80,90,100,150,200,210.220,230,240,250,260,270,290,300,320,330,340,360,370,380,390,400]

#for i in a:
#   print i
#   sel= SelectFromModel(LogisticRegression(C=i, penalty='l1'))
#   sel.fit(X_train, y_train)
#   X_important_train = sel.transform(X_train)
#   X_important_test = sel.transform(X_test)
#   sel_important = LogisticRegression()
#   scores =cross_val_score( sel_important , X_train, y_train, cv=5)
#   print scores
#   print("Average 5-Fold CV Score: {}".format(np.mean(scores)))
#   print('features with coefficients shrank to zero: {}'.format(np.sum(sel.estimator_.coef_ == 0)))

#numberfeature=[]
#for feature_list_index in sel.get_support(indices=True):
#   print(feature_lst[feature_list_index])
#   numberfeature.append(feature_list_index)
#print len(numberfeature)





sel= SelectFromModel(LogisticRegression(C=1e-2, penalty='l1'))
sel.fit(X_train, y_train)
X_important_train = sel.transform(X_train)
X_important_test = sel.transform(X_test)
sel_important = LogisticRegression()
scores =cross_val_score( sel_important , X_train, y_train, cv=5)
print scores
print("Average 5-Fold CV Score: {}".format(np.mean(scores)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel.estimator_.coef_ == 0)))


numberfeature=[]
for feature_list_index in sel.get_support(indices=True):
#   print(feature_lst[feature_list_index])
   numberfeature.append(feature_list_index)
print len(numberfeature)
print numberfeature

