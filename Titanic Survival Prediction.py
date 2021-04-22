#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
from sklearn.ensemble import RandomForestClassifier
r=RandomForestClassifier()
from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
from sklearn .neighbors import KNeighborsClassifier
k=KNeighborsClassifier()
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
g=GaussianNB()
b=BernoulliNB()
from xgboost import XGBClassifier
xgbc=XGBClassifier()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


df1=pd.read_csv("train.csv")


# In[4]:


df1.shape


# In[5]:


df2=pd.read_csv("test.csv")


# In[6]:


df2.shape # test dosyasında hayatta kalanlar yok çünkü bunları tahmin edeceğiz


# In[7]:


df=df1.append(df2) # iki dataframe birleştirdik 


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df.Embarked.value_counts() # hangi limandan kaç kişi binmiş


# In[11]:


df.Embarked.value_counts(normalize=True) # limandan binenlerin yüzdesini gösterir


# In[12]:


df["Name"]


# In[13]:


df["Title"]=df["Name"].str.extract(" ([A-Za-z]+)\.",expand=False)
df["Title"].value_counts()


# In[14]:


df["Title"]=df["Title"].replace(["Ms","Mlle"],"Miss")
df["Title"]=df["Title"].replace(["Mme","Countess","Lady","Dona"],"Mrs")
df["Title"]=df["Title"].replace(["Dr","Major","Col","Sir","Rev","Jonkheer","Capt","Don"],"Mr")


# In[15]:


df["Title"].unique()


# In[16]:


df.info()


# In[17]:


del df["Cabin"] # cabin verisi az olduğu için onu sildik


# In[18]:


df["Fare"].fillna(df["Fare"].mean(),inplace=True) # Fare bölümünde 1 eksik veriyi ortalama veri ile doldurduk
# " fillna " boşlukları doldur demek


# In[19]:


df["Age"].fillna(df["Age"].median(),inplace=True)


# In[20]:


df["Family"]=df["SibSp"]+df["Parch"]+1


# In[21]:


df.Embarked.value_counts(dropna=False).plot(kind="bar");


# In[22]:


df["Embarked"]=df["Embarked"].fillna("S")


# In[23]:


sns.countplot(x="Embarked",hue="Survived",data=df);


# In[24]:


df.Age.plot(kind="hist",bins=50);


# In[25]:


df.Family.value_counts() # bir ailenin kaç kişi olduğunu gösterir örneğin 7 kişilik 16 aile varmış


# In[26]:


df["Single"]=df.Family<2
df["Small"]=(df.Family>1)&(df.Family<5)
df["Medium"]=(df.Family>4)&(df.Family<7)
df["Large"]=(df.Family>6)


# In[27]:


df.drop(["PassengerId","Ticket"],axis=1,inplace=True) # bilet numarsı vb. işimize yaramaycağını düşündük sildik


# In[28]:


df_dummies=pd.get_dummies(df,drop_first=True) # string değerleri sayısal değişken olarak tutuyoruz


# In[29]:


df_train=df_dummies[:891]
df_test=df_dummies[891:] # veriyi tekrar böldük bir kısmını test için ayırdık


# In[30]:


df_train.shape,df_test.shape


# In[31]:


del df_train["Survived"]


# In[32]:


y=df1["Survived"]


# In[33]:


algorithms=[g,b,k,log,gbc,r,d,xgbc]
names=["GaussianNB","BernoulliNB","K Nearest","Logistic","GradientBoosting","RandomForest","Decision Tree","Xgbc"]


# In[37]:


def algo_test(X,y,algorithms=algorithms,names=names):
    for i in range(len(algorithms)):
        algorithms[i]=algorithms[i].fit(X,y)
    
    accuracy=[]
    precision=[]
    recall=[]
    f1=[]
    for i in range(len(algorithms)):
        accuracy.append(accuracy_score(y,algorithms[i].predict(X)))
        precision.append(precision_score(y,algorithms[i].predict(X)))
        recall.append(recall_score(y,algorithms[i].predict(X)))
        f1.append(f1_score(y,algorithms[i].predict(X)))
    metrics=pd.DataFrame(columns=["Accuracy","Precision","Recall","F1"],index=names)
    metrics["Accuracy"]=accuracy
    metrics["Precision"]=precision
    metrics["Recall"]=recall
    metrics["F1"]=f1
    return metrics.sort_values("F1",ascending=False)    


# In[38]:


algo_test(df_train,y)


# In[36]:


df_train.isnull().sum()


# In[39]:


del df_test["Survived"]


# In[66]:


tahmin=k.predict(df_test)


# In[58]:


tahmin


# In[67]:


sonuc5=df2[["PassengerId"]]


# In[68]:


sonuc5["Survived"]=tahmin


# In[55]:


sonuc1.head()


# In[69]:


sonuc5.to_csv("sonuc5.csv",index=False)


# In[ ]:




