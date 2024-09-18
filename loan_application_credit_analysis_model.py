#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_excel("Loan applicant's Risk Segmentation-Dataset.xlsx")


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


df.describe()


# # Display scatter plot wbetween age & Total Work Experience

# In[15]:


plt.scatter(df['Age'],df['Total Work Experience'])


# # Display box plot for age

# In[18]:


plt.boxplot(df['Age'])


# # Display box plot for Cibil score

# In[19]:


plt.boxplot(df['Cibil score'])


# # Create target and feature data

# In[20]:


X = df.drop('Total bounces past12months',axis=1)
y = df['Total bounces past12months']


# In[21]:


X


# In[22]:


y


# # working with model

# # 1. Split data into training and testing

# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier


# # 2. Create Knn classfier

# In[25]:


clf = KNeighborsClassifier()
clf.fit(X_train,y_train)


# # 3.  Testing score

# In[26]:


clf.score(X_test,y_test)


# # 4. Training score

# In[27]:


clf.score(X_train,y_train)


# In[28]:


y_pred = clf.predict(X_test)


# In[29]:


from sklearn.metrics import accuracy_score


# # 5. Accuracy score

# In[30]:


accuracy_score(y_test,y_pred)


# # 6.  Try 1 to 14 k values for classifier

# In[35]:


from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    model = KNeighborsClassifier(i)
    model.fit(X_train,y_train)
    
    train_scores.append(model.score(X_train,y_train))
    test_scores.append(model.score(X_test,y_test))


# # 7. Display training and testing score for that 1 to 14 k values 

# In[34]:


import seaborn as sns
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[ ]:




