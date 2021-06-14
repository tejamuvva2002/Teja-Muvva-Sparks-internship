#!/usr/bin/env python
# coding: utf-8

# # Task 1 - prediction using Supervised ML

# # Author - Teja Muvva 

# In[35]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


# In[36]:


# Reading the Data
data=pd.read_csv('http://bit.ly/w-data')


# In[37]:


data


# In[38]:


data.head(10)


# In[39]:


# Checking if there are any null values in the Dataset
data.isnull == True


# There are no null values in the Dataset so now we can visualize the data

# In[40]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()


# From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied', Lets plot a regression line to confirm the correlation.

# In[41]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# From the Graph we can see that there is a positive relationship between 'Marks Percentage' and 'Hours Studied'.

# In[42]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[43]:


# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[44]:


# Fitting the Data into the module
regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# In[45]:


# Predicting the Percentage of Marks
pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[46]:


# Comparing the Predicted Marks with the Actual Marks
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[47]:


# Visually Comparing the Predicted Marks with the Actual Marks
plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[48]:


# Evaluating the Model
# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# Small value of Mean absolute error states that the chances of error forecasting through the model are very less.

# In[49]:


# What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?
hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))


# According to the regression model if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.

# In[ ]:




