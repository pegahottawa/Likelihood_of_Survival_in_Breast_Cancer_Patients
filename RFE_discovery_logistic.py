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

#path_to_file = '/home/pegah/breast_cancer/mRNA.xlsx'
#df200 = pd.read_excel(path_to_file)

#path_to_file = '/home/pegah/breast_cancer/column_mRNA.xlsx'
#df2000 = pd.read_excel(path_to_file)

path_to_file = '/home/peg/column.xlsx'  # column.xlsx file contain only column names for about 48804 gene names
df100 = pd.read_excel(path_to_file)

df3.columns = df3.iloc[0]  # after trasposing a df2 column names become the first row ,thats why I renamed the column names
                           # with the firts row
df4 = df3.drop(df3.index[[0]])     # after that because by renaming I would end up with same
                                   # column names and and same row values,I droped the first row

df6=df4.reset_index()              # I wanted to have dataframe format with all index for each row
df6.columns = df100.iloc[:,0]      # After all, now its time to rename columns
result = pd.merge(df1,df6,on='METABRIC_ID')

NON_value = result.isnull().sum()   # how many NON value we have in each column
#print NON_value
del_NON = result.dropna(how = 'any')  # drop row if you find any NON value in it
#print del_NON

Tar = del_NON[del_NON.columns[7]]
#print Tar
Descriptor = del_NON.drop(del_NON.columns[[0,7]], axis=1)
arr_descriptors=Descriptor.values
arr_Target=Tar.values

X = arr_descriptors
T = arr_Target
Target = np.reshape(T, (609,)) # when using kfold cross validation you need this sort of reshape
#Target = np.reshape(T, (609,1))
Y = Target

print 'X.shape =', X.shape
print 'Y.shape =', Y.shape

###################################################################
#Recursive feature elimination with cross-validation(Stratified)
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

print "Final result"

print(logreg.score(X_test_rfecv, y_test))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print 'rmse =', rmse

