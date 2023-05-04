#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries


# For analysis and numerical functions
import numpy as np
import pandas as pd

# For Vizualization
import matplotlib.pyplot as plt
import seaborn as sns

# Extra
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# Read the csv file using pandas

df = pd.read_csv('day.csv')


# In[3]:


# Reading the Dataset
df.head()


# In[4]:


# Checking the total number of Rows and Columns.
df.shape


# In[5]:


# Checking the informations of datasets.
df.info()


# In[6]:


# Checking the Statistical informations of the dataset.
df.describe()


# In[7]:


#Cleaning Data
#Drop those columns who are of no use for analysis

# Creating a list to drop no use columns at onces
dropping_cols = ['instant','dteday','casual','registered']
df.drop(dropping_cols, axis=1, inplace=True)


# In[8]:


# Checking Data after dropping columns
df


# In[9]:


# Checking proportion of missing values in each column
df.isnull().mean()


# In[10]:


# Calulating number of unique values in each column
df.nunique()


# In[11]:


# Creating list for categorical, continuous and target column.
cat_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
cont_cols = ['temp','atemp','hum','windspeed']
target = ['cnt']
len(cat_cols)+len(cont_cols)+len(target)


# In[12]:


#univariate analysis

# Print continuous columns graph using countplot
fig, axs = plt.subplots(len(cont_cols) // 2, 2, figsize=(16, 10))
for i in range(len(cont_cols) // 2):
    for j in range(2):
        index = 2 * i + j
        sns.histplot(df[cont_cols[index]], ax=axs[i][j])
plt.show()


# In[ ]:


# Insights:

#According to 'windspeed' people would prefer to travel while windspeed is between 8 to 16 after, that the rented bike graph goes down
#On the basis 'hum' column customer rented bike when humidity is higher than 70
#According to 'temp' graph people prefer to rented bike when temprature is moderate either extreme low nor extreme high
#Based on above insights we can conclude that climate directly control shared bike business.


# In[13]:


# Print countplot graph for categorical columns
fig, axs = plt.subplots(len(cat_cols) // 2, 2, figsize=(16, 10))
for i in range(len(cat_cols) // 2):
    for j in range(2):
        index = 2 * i + j
        sns.countplot(x=cat_cols[index], data=df, ax=axs[i][j])
plt.show()


# In[14]:


#Bivariate analysis

# Boxplot for categorical column
cat_cols = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
for i in enumerate(cat_cols):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(data=df, x=i[1], y='cnt')
plt.show()


# In[16]:


# Inference:
#Here are few insights we drawn from the plots

#. Fall is the most popular season to hire bikes.
#. I've noticed an increase in demand for next year.
#. Demand is increasing month after month until season_6. The month of season_9 has the biggest demand. Demand begins to fall after season_9.
#. When there is a holiday, demand falls.
#. The weekday does not provide a clear picture of demand.


# In[17]:


# Draw box plots for indepent variables with continuous values
cols = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(18,4))

i = 1
for col in cols:
    plt.subplot(1,4,i)
    sns.boxplot(y=col, data=df)
    i+=1


# In[18]:


#Multivariate analysis

# create new dataframe with only continuous columns
cont_df = df[cont_cols]


# In[19]:


# plot pairplot
sns.pairplot(cont_df)
plt.show()


# In[20]:


#creating dummy variables

# Checking no. of unique values of categorical columns
df[cat_cols].nunique()# Only for non binary categorical columns


# In[21]:


# Creating dummy variable for 'season' column and drop first column
season_dum = pd.get_dummies(df["season"], prefix="season", drop_first=True)
season_dum.head()


# In[22]:


# Creating dummy variable for 'mnth' column and drop first column
mnth_dum=pd.get_dummies(df["mnth"], prefix="mnth", drop_first=True)
mnth_dum.head()


# In[23]:


# Creating dummy variable for 'weekday' column and drop first column
weekday_dum=pd.get_dummies(df["weekday"], prefix="weekday", drop_first=True)
weekday_dum.head()


# In[24]:


# Creating dummy variable for 'weathersit' column and drop first column
weathersit_dum=pd.get_dummies(df["weathersit"], prefix="weathersit", drop_first=True)
weathersit_dum.head()


# In[25]:


# Combining the result to the dataframe
df1 = pd.concat([season_dum, mnth_dum, weekday_dum, weathersit_dum, df], axis=1)
df1.head()


# In[28]:


# Print columns after creating dummy variuables 
df1.columns


# In[29]:


# Check if the columns exist in the dataframe before dropping them
cols_to_drop = ['season', 'mnth', 'weathersit', 'weekday']


# In[30]:


# Creating loop to drop columns
existing_cols = [col for col in cols_to_drop if col in df.columns]
df1.drop(existing_cols, axis=1, inplace=True)


# In[31]:


# Again reading data
df1.head()


# In[32]:


df1.info()


# In[33]:


# Checking columns again after dropping not necessary columns
df1.columns


# In[34]:


#Train_Test_split

# Spliting columns into two portion 
X=df1.drop(["cnt"],axis=1)   # X = features 
y=df1["cnt"]   # y = target column


# In[35]:


# Spliting data into 4 parts as X_train,X_test,y_train,y_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


# In[36]:


# Measuring X_train shape
X_train.shape


# In[37]:


# Measuring X_test shape
X_test.shape


# In[38]:


#Scaling

#Standardization
#z=x-xmean/sigma
X_train=(X_train-X_train.mean())/X_train.std()


# In[39]:


X_test=(X_test-X_test.mean())/X_test.std()


# In[41]:


#Modelling

#Feature Selection

# Building a Lienar Regression model using SKLearn for RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

estimator=LinearRegression()
selector=RFE(estimator,n_features_to_select=15)#15

selector=selector.fit(X_train,y_train)
selector.support_
selector


# In[42]:


# Columns selected by RFE
selected_features = list(X_train.columns[selector.support_])
selected_features


# In[43]:


# Assigning changes of selected_feature into X_train and X_test
X_train = X_train[selected_features]
X_test = X_test[selected_features]


# In[44]:


# Saving above changes into X_train-sm and X_test_sm
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# In[45]:


# First Model
model1 = sm.OLS(y_train,X_train_sm)
res1 = model1.fit()
res1.summary()


# In[46]:


# Checking VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"]=X_train.columns

vif_data["VIF"] = [variance_inflation_factor(X_train.values,i) for i in range(len(X_train.columns))]
vif_data


# In[47]:


# Dropping season_3 because it is very closer to p_value 0.05 or (5%).

X_train = X_train_sm.drop(["season_3"],axis=1)
X_test = X_test_sm.drop(["season_3"],axis=1)


# In[48]:


# again adding the changes in update in x_data

X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# In[49]:


# Second Model
model2 = sm.OLS(y_train,X_train_sm)
res2 = model2.fit()
res2.summary()


# In[50]:


# Checking VIF again
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"]=X_train_sm.columns

vif_data["VIF"]=[variance_inflation_factor(X_train_sm.values,i) for i in range(len(X_train_sm.columns))]
vif_data


# In[51]:


# Dropping mnth because it has high p_value of 0.121
X_train = X_train_sm.drop(["mnth_3"],axis=1)
X_test = X_test_sm.drop(["mnth_3"],axis=1)


# In[52]:


# again adding the changes in update in x_data
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# In[53]:


# Third Model
model3 = sm.OLS(y_train,X_train_sm)
res3 = model3.fit()
res3.summary()


# In[54]:


# Again checking VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"]=X_train_sm.columns

vif_data["VIF"]=[variance_inflation_factor(X_train_sm.values,i) for i in range(len(X_train_sm.columns))]
vif_data


# In[55]:


#_Inference_¶

#Here VIF seems to be almost accepted. p-value for all the features is almost 0.0 and R2 is 0.847
#Let us select Model 3 as our final as it has all important statistics high (R-square, Adjusted R-squared and F-statistic), 
#along with no insignificant variables and no multi coliinear (high VIF) variables. Difference between R-squared and 
#Adjusted R-squared values for this model is veryless, 
#which also means that there are no additional parameters that can be removed from this model.


# In[56]:


#Build a model with all columns to select features automatically
def build_model_sk(X,y):
    lr1 = LinearRegression()
    lr1.fit(X,y)
    return lr1


# In[57]:


#Let us build the finalmodel using sklearn
cols = ['season_2', 'season_4', 'mnth_8',
        'mnth_9', 'mnth_10', 'weekday_6', 
        'weathersit_2', 'weathersit_3', 
        'yr', 'workingday', 'temp', 'hum', 
        'windspeed']

#Build a model with above columns
lr = build_model_sk(X_train[cols],y_train)
print(lr.intercept_,lr.coef_)


# In[58]:


y_train_pred = lr.predict(X_train[cols])


# In[59]:


#Plot a histogram of the error terms
def plot_res_dist(act, pred):
    sns.distplot(act-pred)
    plt.title('Error Terms')
    plt.xlabel('Errors')


# In[60]:


plot_res_dist(y_train, y_train_pred)


# In[61]:


# Importing Libraries to calculate r square, train as well as test performance
from sklearn.metrics import r2_score


# In[62]:


# R-Square of Test Data
predicted_value = res3.predict(X_test_sm)
print("Test Performance:",round(r2_score(y_test, predicted_value)*100,2),"%")


# In[63]:


# R-Square of Train Data
predicted_value1 = res3.predict(X_train_sm)
print("Train Performance:",round(r2_score(y_train, predicted_value1)*100,2),"%")


# In[64]:


# Calculating Adjusted-R^2 value for the test dataset
adjusted_r2 = round((1 - (1 - r2_score(y_test, predicted_value)) * (X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)) * 100, 2)
print(f"Adjusted R-Squared: {adjusted_r2} %")


# In[65]:


#CONCLUSION

#Analysing the above model, the comapany should focus on the following features:
#Company should focus on expanding business during 'season_2' and 'season_4'.
#Company should focus on expanding business during mnth_8 to mnth_10.
#Based on previous data it is expected to have a boom in number of users once situation comes back to normal, compared to 2019.
#There would be less bookings during Light Snow or Rain, they could probably use this time to serive the bikes without having business impact.
#Hence when the situation comes back to normal, the company should come up with new offers during season_3 when the weather is pleasant and also advertise a little for mnth_8 to mnth_10 as this is when business would be at its best.


# In[ ]:




