#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


#reading the data file

data= pd.read_csv("C:\\Users\\jashj\\OneDrive\\Desktop\\Datasets\\Credit Card Defaulter Prediction.csv")


# In[6]:


data


# In[7]:


#Looking for any null values

data.isna().sum()


# In[8]:


#Finding the data types

data.dtypes


# In[9]:


#For addtional information 

data.info()


# In[10]:


#Identifying the dispersion in the dataset

data.boxplot(figsize = (30,10))


# In[11]:


# Checking for outliers in the data

Q1=data['LIMIT_BAL'].quantile(0.25)
Q2=data['LIMIT_BAL'].quantile(0.5)
Q3=data['LIMIT_BAL'].quantile(0.75)
Q4=data['LIMIT_BAL'].quantile(0.99)
print (Q4)
IQR=Q3-Q1
print (IQR)
OUT1=Q3+1.5*IQR
print(OUT1)
OUT2=Q1-1.5*IQR
print(OUT2)


# In[12]:


#Treatment of outlier

#For upper outlier
data['LIMIT_BAL'].max()
Upper_Out=data['LIMIT_BAL'].quantile(0.99)
data.loc[data.LIMIT_BAL > Upper_Out,'LIMIT_BAL']=Upper_Out
data['LIMIT_BAL'].max()

#For lower outlier
data['LIMIT_BAL'].min()
Lower_Out=data['LIMIT_BAL'].quantile(0.99)
data.loc[data.LIMIT_BAL < Upper_Out,'LIMIT_BAL']=Lower_Out
data['LIMIT_BAL'].min()


# In[13]:


data


# In[14]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# In[18]:


#Creating dummy variable for object type

cat_vars=['SEX','EDUCATION','MARRIAGE']
dummy_x=pd.get_dummies(data,columns=cat_vars,drop_first=True)


# In[19]:


#Dropping unwanted variable

x=dummy_x.drop('ID',axis=1)
x=x.drop('default ',axis=1)


# In[20]:


x.head()


# In[21]:


data


# In[36]:


####code for Treating multiple column outlier together

def cut_data(df):
    for column in df.columns:
        print("cutting the", column)
        if(((df[column].dtype)=='float64')|((df[column].dtype)=='int64')):
            percentiles = df[column].quantile([0.01,0.99]).values
            df[column][df[column] <= percentiles[0]] = percentiles[0]
            df[column][df[column] >= percentiles[1]] = percentiles[1]
        else:
            df[column] = df[column]
    return df

data= cut_data(data)


# In[39]:


#checking if the Treatment has occured properly

data['AGE'].max()


# In[40]:


data['AGE'].min()


# In[24]:


#Seperating x and y variables

y= np.where(data['default ']=='Y', 1,0)


# In[25]:


y


# In[26]:


#Using Train test split to train 80% of the model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape


# In[27]:


from sklearn.linear_model import LogisticRegression
#creating on object of LinearRegression class

LR= LogisticRegression()
#fitting the training data
#.fit is model creation function

LR.fit(x_train,y_train)


# In[28]:


LR.intercept_


# In[29]:


LR.coef_


# In[30]:


#Predicting the y variable

y_pred_test=LR.predict(x_test)
LR.score(x_test,y_test,sample_weight=None)
y_pred_test


# In[31]:


#Printing the important values

from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred_test)
print('The accuracy is: ',acc)
acc=metrics.precision_score(y_test,y_pred_test)
print('The precision is: ',acc)
acc=metrics.recall_score(y_test,y_pred_test)
print('The Recall Rate: ',acc)
acc=metrics.roc_auc_score(y_test,y_pred_test)
print('The roc_and_score: ',acc)


# In[34]:


#Printing Regression Results 

import statsmodels.api as sm
x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train).fit()
#predictions = model.predict(x)
print_model = model.summary()
print(print_model)


# In[ ]:





# In[ ]:




