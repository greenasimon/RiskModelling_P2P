# Databricks notebook source
import pandas as pd
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

#Dataset 
filePath1 = '/FileStore/tables/LoanstatsDefaultInnerjoin.csv' 
#Move the dataset into a spark Dataframe
loanstats = spark.read.option("header","true"). option("inferSchema","true").csv(filePath1)
#move to pandas dataframe
pd_loanstats = loanstats.toPandas() 
pd_loanstats = pd_loanstats.drop_duplicates(['LOAN_ID'])

# COMMAND ----------

##########################################################Data Preprocessing##############################################################################
#1. LOAN_ID will be in index for the data
#indexedStats = pd_loanstats.set_index('LOAN_ID')
indexedStats = pd_loanstats.drop_duplicates(['LOAN_ID'])
#2.PBAL_BEG_PERIOD id the principal amount borrowed. Renaming the column.
indexedStats.rename(columns= {'PBAL_BEG_PERIOD':'PBal_Beg'},inplace = True)
#3.Month - First payment month - Not required for probability of Default.We have Issue date already! Hence, Dropping the column.
indexedStats.drop('MONTH',axis =1, inplace = True)
#4.InterestRate = Required column in required format. Nothing to do here.
#5.IssuedDate - For simplicity, we will remove the month string to get the issued year and label encode data for each year.
indexedStats['IssuedYear'] = indexedStats['IssuedDate'].str.replace(r'[^0-9]','')
indexedStats.drop('IssuedDate', axis = 1, inplace = True)

# COMMAND ----------

pd_pca = pd_loanstats['LOAN_ID']
print(pd_pca.shape)
pd_pca.head()

# COMMAND ----------

lb_IY = LabelBinarizer()
lb_results = lb_IY.fit_transform(indexedStats['IssuedYear'])
pd_pca = pd.concat([pd_pca,pd.DataFrame(lb_results, columns=lb_IY.classes_)],axis =1)
pd_pca.head()

# COMMAND ----------

#6 MONTHLYCONTRACTAMT - format is as required. renaming the column.
indexedStats.rename(columns = {'MONTHLYCONTRACTAMT':'MonthlyAmt'}, inplace = True)
#7 dti - Debt to income ratio. format as required. No changes here.
indexedStats.head()

# COMMAND ----------

#8 State - Categorical variable with a lot of levels. Using one hot encoding and adding to PCA dataframe
lb_States = LabelBinarizer()
lb_Stresults = lb_States.fit_transform(indexedStats['State'])
pd_pca = pd.concat([pd_pca,pd.DataFrame(lb_Stresults, columns=lb_States.classes_)],axis =1)
pd_pca.head()

# COMMAND ----------

#9 HomeOwnership - Requires one hot encoding and adding to PCA dataframe
lb_HO = LabelBinarizer()
lb_HOresults = lb_HO.fit_transform(indexedStats['HomeOwnership'])
pd_pca = pd.concat([pd_pca,pd.DataFrame(lb_HOresults, columns=lb_HO.classes_)],axis =1)
pd_pca.head()

# COMMAND ----------

#10 MonthlyIncome - in required format but has missing values. filling it with mean MonthlyIncome
indexedStats['MonthlyIncome'].fillna(indexedStats['MonthlyIncome'].mean(), inplace = True)
#11 EarliestCREDITLine -  Not required as of now. Dropping the column.
indexedStats.drop('EarliestCREDITLine',axis =1, inplace = True)
#12 OpenCREDITLines - in required format but has missing values but very less. filling it with mean value
indexedStats['OpenCREDITLines'].fillna(indexedStats['OpenCREDITLines'].mean(), inplace = True)
#13 TotalCREDITLines - in required format but has missing values but very less. filling it with mean value
indexedStats['TotalCREDITLines'].fillna(indexedStats['TotalCREDITLines'].mean(), inplace = True)
#14 RevolvingCREDITBalance - in required format but has missing values but very less. filling it with mean value
indexedStats['RevolvingCREDITBalance'].fillna(indexedStats['RevolvingCREDITBalance'].mean(), inplace = True)
#15 RevolvingCREDITBalance - in required format but has missing values but very less. filling it with mean value
indexedStats['RevolvingLineUtilization'].fillna(indexedStats['RevolvingLineUtilization'].mean(), inplace = True)
#16 Inquiries6M - Need to remove unwanted values and convert it to integer
indexedStats['Inquiries6M']= indexedStats['Inquiries6M'].replace( r'[^0-9]', np.NaN)
indexedStats['Inquiries6M']= indexedStats['Inquiries6M'].replace( '*', np.NaN)
indexedStats['Inquiries6M'] = indexedStats['Inquiries6M'].astype('float')
indexedStats['Inquiries6M'].fillna(indexedStats['Inquiries6M'].mean(), inplace = True)
#17 #18 #19 #20 DQ2yrs MonthsSinceDQ PublicRec MonthsSinceLastRec - data insufficient because of missing values/90 % zero etc. Dropping the columns
indexedStats.drop(['DQ2yrs','MonthsSinceDQ','PublicRec','MonthsSinceLastRec'], axis = 1, inplace = True)

# COMMAND ----------

#21 EmploymentLength - One hot encoding and add to PCA
lb_Emplen = LabelBinarizer() # Replacing labels for EmploymentLength
lb_Emplenresults = lb_Emplen.fit_transform(indexedStats['EmploymentLength'])
pd_pca = pd.concat([pd_pca,pd.DataFrame(lb_Emplenresults, columns=lb_Emplen.classes_)],axis =1)
pd_pca.head()

# COMMAND ----------

#22 currentpolicy - in required format. No missing values. Keeping it as is.
#23 Grade - Dropping it as it is not available in the 424b3 files
indexedStats.drop('grade', axis = 1, inplace = True)

# COMMAND ----------

#24 term - Available duration of the loan. 2 type. 36 months and 60 months. Converting it to  category variable as shortTerm nd longTerm. No null values
TermAvail = indexedStats.groupby(['term'])['term'].count()
indexedStats['termUpdated'] = indexedStats['term'].replace([36,60],['ShortTerm','LongTerm'])
lb_termU = preprocessing.LabelEncoder() # Replacing labels 
indexedStats['termU_code'] = lb_termU.fit_transform(indexedStats['termUpdated'])
indexedStats[['termUpdated','termU_code']].head()

# COMMAND ----------

#25 APPL_FICO_BAND - No missing values. format is a range. taking the min value here.
indexedStats['Min_FICOBand'] = indexedStats['APPL_FICO_BAND'].str.split('-').str[0].astype('int32')
#26 Last_FICO_BAND 27 VINTAGE 28 RECEIVED_D 29 PBAL_END_PERIOD 30 Final_Term 31 CO 32 COAMT 33 PCO_RECOVERY 34 PCO_COLLECTION_FEE - Dropping as not required now.
indexedStats.drop(['Last_FICO_BAND','VINTAGE','RECEIVED_D','PBAL_END_PERIOD','Final_Term','CO','COAMT','PCO_RECOVERY','PCO_COLLECTION_FEE'], axis = 1, inplace = True)

# COMMAND ----------

#adding y variable to PCA
pd_pca = pd.concat([pd_pca, pd_loanstats['Final_Stat']],axis =1)
pd_pca.head()

# COMMAND ----------

# Y variable - 'Final_Stat'
AvailableYs = indexedStats.groupby(['Final_Stat'])['Final_Stat'].count()
#AvailableYs = [ Charged Off, Default, Fully Paid, Issued]
#Charged Off, Default => Default, 'Issued' not required. So will drop those rows.
indexedStats = indexedStats[indexedStats['Final_Stat'] != 'Issued']
pd_pca = pd_pca[pd_pca['Final_Stat'] != 'Issued']
indexedStats['Final_Stat'].replace('Charged Off','Default', inplace = True)
pd_pca['Final_Stat'].replace('Charged Off','Default', inplace = True)
lb_DefaultStatus = preprocessing.LabelEncoder() # Replacing labels 
indexedStats['DefaultStatus'] = lb_termU.fit_transform(indexedStats['Final_Stat'])
indexedStats[['Final_Stat','DefaultStatus']].head()

# COMMAND ----------

lb_Default = preprocessing.LabelEncoder() # Replacing labels 
pd_pca['Default'] = lb_Default.fit_transform(pd_pca['Final_Stat'])
pd_pca[['Final_Stat','Default']].head()
pd_pca.drop(['Final_Stat'],axis = 1,inplace = True)


# COMMAND ----------

#################Principal Component Analysis######################################################
X = pd_pca.drop(['Default'],axis = 1)
indexedX = X.set_index('LOAN_ID')
pca = PCA(n_components = 5)
principalComponents = pca.fit_transform(indexedX.values)
principalDf = pd.DataFrame(data = principalComponents)

# COMMAND ----------

print(principalDf.shape)
print(indexedStats.shape)

# COMMAND ----------

#Drop all the old related columns and keep the new category coded columns for prediction
CleanedData = indexedStats.drop(['Final_Stat','APPL_FICO_BAND','termUpdated','term','EmploymentLength',
                                 'HomeOwnership','State','IssuedYear'], axis = 1)
CleanedData.head()

# COMMAND ----------

A = CleanedData.reset_index()
B = principalDf
print(CleanedData.shape)
print(principalDf.shape)
print(A.shape)
FinalDf = pd.concat([B,A],axis =1)
print(FinalDf.shape)

# COMMAND ----------

FinalDf = FinalDf.set_index(['LOAN_ID'])
FinalDf.drop(['MonthlyAmt'],axis = 1, inplace = True)#high correlation with another variable
FinalDf.head()

# COMMAND ----------

print(FinalDf.shape)
print(FinalDf.corr().round(2))

# COMMAND ----------

print(FinalDf.describe().T)

# COMMAND ----------

FinalDfA = FinalDf[FinalDf["DefaultStatus"] == 0]
FinalDfB = FinalDf[FinalDf["DefaultStatus"] == 1] 
ab1 = FinalDfA.mean().T.round(2)
ab2 = FinalDfB.mean().T.round(2)
print(pd.concat([ab1,ab2],axis =1 ))

# COMMAND ----------

FinalDf.shape

# COMMAND ----------

##########################Split the data into input and output variables
inputX = FinalDf.values[:,0:18]
inputY = FinalDf.values[:,18]
# Standardizing the features
inputX = StandardScaler().fit_transform(inputX)
print(inputX)
print(inputY)

# COMMAND ----------

print(inputX[0])
print(FinalDf.values[0])

# COMMAND ----------

#################################################Data Modeling###########################################################

# COMMAND ----------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.25, random_state=0)

# COMMAND ----------

#################################################NEURAL NETWORK###########################################################
modelNeural = Sequential()
modelNeural.add(Dense(20, input_dim=18, activation='relu'))
modelNeural.add(Dense(20, activation='relu'))
modelNeural.add(Dense(1, activation='sigmoid'))
# Compile model
modelNeural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

# Fit the model
modelNeural.fit(X_train, y_train, epochs=100, batch_size=1000)
# evaluate the model
scores = modelNeural.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (modelNeural.metrics_names[1], scores[1]*100))

# COMMAND ----------

#################################################LogisticRegression##############################################################
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# COMMAND ----------

#logreg = LogisticRegression()
#rfe = RFE(logreg, 10) #10 will be the number of features to be selected
#rfe = rfe.fit(inputX, inputY )
#print(rfe.support_)
#print(rfe.ranking_)
#or maybe i dont need to reduce it now! i will use all 41 columns

# COMMAND ----------

logit_model=sm.Logit(inputY,inputX)
result=logit_model.fit()
print(result.summary())

# COMMAND ----------

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# COMMAND ----------

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(logreg.score(X_test, y_test)*100))

# COMMAND ----------

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()*100))

# COMMAND ----------

############################################RandomForest############################################################
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100,min_samples_split = 100,min_samples_leaf=100, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train);

# COMMAND ----------

print('Accuracy of random forest classifier on test set: {:.3f}'.format(rf.score(X_test, y_test)*100))

# COMMAND ----------


