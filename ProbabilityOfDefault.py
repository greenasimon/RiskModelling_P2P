# Databricks notebook source
import pandas as pd
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

# COMMAND ----------

#Dataset 
filePath1 = '/FileStore/tables/LoanstatsDefaultInnerjoin.csv' 
#Move the dataset into a spark Dataframe
loanstats = spark.read.option("header","true"). option("inferSchema","true").csv(filePath1)
#move to pandas dataframe
pd_loanstats = loanstats.toPandas() 

# COMMAND ----------

##########################################################Data Preprocessing##############################################################################
#1. LOAN_ID will be in index for the data
indexedStats = pd_loanstats.set_index('LOAN_ID')
#2.PBAL_BEG_PERIOD id the principal amount borrowed. Renaming the column.
indexedStats.rename(columns= {'PBAL_BEG_PERIOD':'PBal_Beg'},inplace = True)
#3.Month - First payment month - Not required for probability of Default.We have Issue date already! Hence, Dropping the column.
indexedStats.drop('MONTH',axis =1, inplace = True)
#4.InterestRate = Required column in required format. Nothing to do here.

# COMMAND ----------

#5.IssuedDate - For simplicity, we will remove the month string to get the issued year and label encode data for each year.
indexedStats['IssuedYear'] = indexedStats['IssuedDate'].str.replace(r'[^0-9]','')
indexedStats.drop('IssuedDate', axis = 1, inplace = True)
lb_IY = preprocessing.LabelEncoder()
indexedStats['IY_code'] = lb_IY.fit_transform(indexedStats['IssuedYear']) #encoding years
indexedStats[['IssuedYear','IY_code']].head()

# COMMAND ----------

#6 MONTHLYCONTRACTAMT - format is as required. renaming the column.
indexedStats.rename(columns = {'MONTHLYCONTRACTAMT':'MonthlyAmt'}, inplace = True)
#7 dti - Debt to income ratio. format as required. No changes here.
indexedStats.head()

# COMMAND ----------

#8 State - This part is a bit tricky since we have so many categories here! One method is to see number of statewise defaults and decide
StatewiseDefault = indexedStats.query("Final_Stat == 'Charged Off' or Final_Stat =='Default'").groupby(['State'])['State'].count()
StDefaultGrp1 = [key for key in StatewiseDefault.index if StatewiseDefault[key] < 2000]
StDefaultGrp2 = [key for key in StatewiseDefault.index if StatewiseDefault[key] > 2000 and StatewiseDefault[key] < 5000]
StDefaultGrp3 = [key for key in StatewiseDefault.index if StatewiseDefault[key] > 5000 and StatewiseDefault[key] < 10000]
StDefaultGrp4 = [key for key in StatewiseDefault.index if StatewiseDefault[key] > 10000 and StatewiseDefault[key] < 15000]
indexedStats['StateUpdated'] = indexedStats['State'].replace(StDefaultGrp1,'A1')
indexedStats['StateUpdated'] = indexedStats['StateUpdated'].replace(StDefaultGrp2,'A2')
indexedStats['StateUpdated'] = indexedStats['StateUpdated'].replace(StDefaultGrp3,'A3')
indexedStats['StateUpdated'] = indexedStats['StateUpdated'].replace(StDefaultGrp4,'A4')
lb_StU = preprocessing.LabelEncoder() # Replacing labels for State Groups
indexedStats['StU_code'] = lb_StU.fit_transform(indexedStats['StateUpdated'])
indexedStats[['StateUpdated','StU_code']].head()

# COMMAND ----------

#9 HomeOwnership - Requires Label Encoding
lb_HO = preprocessing.LabelEncoder()
indexedStats['HO_code'] = lb_StU.fit_transform(indexedStats['HomeOwnership'])
indexedStats[['HomeOwnership','HO_code']].head()

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

#21 EmploymentLength - similar to 'State' variable. Need to look at the default numbers
EmpLengthwiseDefault = indexedStats.query("Final_Stat == 'Charged Off' or Final_Stat =='Default'").groupby(['EmploymentLength'])['EmploymentLength'].count()
#based on the numbers categories will be '1 year or less', '2-3 years','4-9 years','10+ years'. Missing values will be in '1 year or less'
indexedStats['EmpLengthUpdated'] = indexedStats['EmploymentLength'].replace(['1 year','< 1 year','n/a'],'1 year or less')
indexedStats['EmpLengthUpdated'] = indexedStats['EmpLengthUpdated'].replace(['2 years','3 years'],'2-3 years')
indexedStats['EmpLengthUpdated'] = indexedStats['EmpLengthUpdated'].replace(['4 years','5 years','6 years','7 years','8 years','9 years'],'4-9 years')
lb_EmpU = preprocessing.LabelEncoder() # Replacing labels for EmploymentLength Groups
indexedStats['EmpU_code'] = lb_EmpU.fit_transform(indexedStats['EmpLengthUpdated'])
indexedStats[['EmpLengthUpdated','EmpU_code']].head()

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

# Y variable - 'Final_Stat'
AvailableYs = indexedStats.groupby(['Final_Stat'])['Final_Stat'].count()
#AvailableYs = [ Charged Off, Default, Fully Paid, Issued]
#Charged Off, Default => Default, 'Issued' not required. So will drop those rows.
indexedStats = indexedStats[indexedStats['Final_Stat'] != 'Issued']
indexedStats['Final_Stat'].replace('Charged Off','Default', inplace = True)
lb_DefaultStatus = preprocessing.LabelEncoder() # Replacing labels 
indexedStats['DefaultStatus'] = lb_termU.fit_transform(indexedStats['Final_Stat'])
indexedStats[['Final_Stat','DefaultStatus']].head()

# COMMAND ----------

#Drop all the old related columns and keep the new category coded columns for prediction
CleanedData = indexedStats.drop(['Final_Stat','APPL_FICO_BAND','termUpdated','term','EmpLengthUpdated','EmploymentLength',
                                 'HomeOwnership','StateUpdated','State','IssuedYear'], axis = 1)
CleanedData.head()

# COMMAND ----------

print(CleanedData.shape)
print(CleanedData.corr().round(2))
print(CleanedData.describe().T)

# COMMAND ----------

CleanedDataA = CleanedData[CleanedData["DefaultStatus"] == 0]
CleanedDataB = CleanedData[CleanedData["DefaultStatus"] == 1] 
ab1 = CleanedDataA.mean().T.round(2)
ab2 = CleanedDataB.mean().T.round(2)
print(pd.concat([ab1,ab2],axis =1 ))

# COMMAND ----------

##########################Split the data into input and output variables
inputX = CleanedData.values[:,0:17]
inputY = CleanedData.values[:,17]
print(inputX)
print(inputY)

# COMMAND ----------

print(inputX[0])
print(CleanedData.values[0])

# COMMAND ----------

#################################################Data Modeling###########################################################

# COMMAND ----------

#################################################NEURAL NETWORK###########################################################
modelNeural = Sequential()
modelNeural.add(Dense(10, input_dim=17, activation='relu'))
modelNeural.add(Dense(10, activation='relu'))
modelNeural.add(Dense(1, activation='sigmoid'))
# Compile model
modelNeural.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

# Fit the model
modelNeural.fit(inputX, inputY, epochs=100, batch_size=1000)
# evaluate the model
scores = modelNeural.evaluate(inputX, inputY)
print("\n%s: %.2f%%" % (modelNeural.metrics_names[1], scores[1]*100))

# COMMAND ----------

#################################################LogisticRegression##############################################################
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# COMMAND ----------

logreg = LogisticRegression()
rfe = RFE(logreg, 10) #10 will be the number of features to be selected
rfe = rfe.fit(inputX, inputY )
print(rfe.support_)
print(rfe.ranking_)
#or maybe i dont need to reduce it now! i will use all 17 columns

# COMMAND ----------

logit_model=sm.Logit(inputY,inputX)
result=logit_model.fit()
print(result.summary())

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# COMMAND ----------

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# COMMAND ----------

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# COMMAND ----------


