#!/usr/bin/env python
#SBATCH --mem=40G
#SBATCH --time=13-24:00:00

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import math

path_to_file = '/home/peg/clinical_dataframe.xlsx'
df1 = pd.read_excel(path_to_file)

path_to_file = '/home/peg/gene_ex.xlsx'
df2 = pd.read_excel(path_to_file)
df3 = df2.transpose()

path_to_file = '/home/peg/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
column_names = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0] 
df4 = df3.drop(df3.index[[0]])     

df6 = df4.reset_index()            
df6.columns = column_names.iloc[:,0]      
df_merge = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = df_merge.isnull().sum()   
#print NON_value
del_NON = df_merge.dropna(how = 'any')  
#print del_NON

Target = del_NON[del_NON.columns[7]]
#print Tar
Descriptor = del_NON.drop(del_NON.columns[[0,7]], axis=1)
arr_descriptors = Descriptor.values
arr_Target = Target.values

X = arr_descriptors
T = arr_Target
Target_reshape = np.reshape(T, (609,))
#Target = np.reshape(T, (609,1))
Y = Target_reshape

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

###################################################################
#RFE with Logistic Regression and cross-validation(Stratified)
###################################################################

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

print 'X_train.shape =', X_train.shape
print 'y_train.shape =', y_train.shape

logreg = LogisticRegression()
#rfe = RFE(estimator = logreg , step=1)
#rfe.fit(X, Y)

rfecv = RFECV(estimator= logreg, step=1, cv=StratifiedKFold(4), scoring='accuracy')
rfecv.fit(X, Y)

print " time for Ranking "

print("Num Features: %s" % (rfecv.n_features_))
print("best Features: %s" % (rfecv.support_))
print("Feature Ranking: %s" % (rfecv.ranking_))

X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

rfecv_model = logreg.fit(X_train_rfecv ,y_train)
y_pred=logreg.predict(X_test_rfecv)

print(logreg.score(X_test_rfecv, y_test))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print 'rmse =', rmse

