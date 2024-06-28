#!/usr/bin/env python
# coding: utf-8

# In[55]:


#Instagram Reach Analysis using Python - Data Science and Machine Learning


# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.linear_model import PassiveAggressiveRegressor


# In[110]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier


# In[111]:


#Read Data


# In[112]:


data = pd.read_csv("/Users/noorehera/Downloads/Instagram_dataset.csv", encoding = "latin1")
data.head(5)


# In[113]:


data.isnull().sum()


# In[114]:


data = data.dropna()


# In[115]:


data.info()


# In[116]:


data


# In[117]:


plt.figure(figsize=(10, 8))
plt.style.use("fivethirtyeight")
plt.title("Distro of Impression From Home")
sns.displot(data["From Home"])
plt.show


# In[118]:


plt.figure(figsize = (10, 8))
plt.title("Distro of Impressions From Hashtags")
sns.displot(data["From Hashtags"])
plt.show()


# In[119]:


plt.figure(figsize = (10, 8))
plt.title("Distro of Impressions From Explore")
sns.displot(data["From Explore"])
plt.show()


# In[120]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()
labels = ["From Home", "From Hashtags", "From Explore", "Other"]
values = [home, hashtags, explore, other]
fig = px.pie(data, names=labels, values=values, title="Impressions of Instagram posts from various sources")
fig.show()


# In[121]:


#Checking most used words


# In[122]:


text = "".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.style.use("classic")
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show


# In[123]:


#Checking most used Hashtags 


# In[66]:


text = "".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.style.use("classic")
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show


# In[ ]:


#Analyzing relationship between likes and impressions


# In[124]:


figure = px.scatter(data_frame=data,  x="Impressions", y="Likes", size="Likes", trendline="ols", title="Relationship Between Likes and Impressions")
figure.show()


# In[125]:


#Analyzing relationship between comments and total impressions


# In[68]:


figure = px.scatter(data_frame = data, x="Impressions", y="Comments", size="Comments", trendline="ols", title="Relationship Between Comments and Total Impressions")
figure.show()


# In[ ]:


#Analyzing relationship between shares and total impressions


# In[69]:


figure = px.scatter(data_frame = data, x="Impressions", y="Shares", size="Shares", trendline="ols", title="Relationship Between Shares and Total Impressions")
figure.show()


# In[ ]:


#Analyzing relationship between saves and total imoressions


# In[70]:


figure = px.scatter(data_frame = data, x="Impressions", y="Saves", size="Saves", trendline="ols", title="Relationship Between Saves and Total Impressions")
figure.show()


# In[76]:


data.columns


# In[ ]:


#Correlation matrix heatmap


# In[126]:


df= data.drop(['Caption','Hashtags'],axis=1)
correlation=df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[127]:


numeric_data = data.select_dtypes(include=[np.number])


# In[128]:


correlation = numeric_data.corr()
print(correlation)


# In[73]:


#converion rate = (Follows / profile Visits) *100


# In[129]:


conversion_rate=(data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# In[130]:


# 41% conversion rate


# In[131]:


#Looking at the relationship between total profile visits and number of followers gained from all profile visits


# In[105]:


figure = px.scatter(data_frame = data, x="Profile Visits", y="Follows", size="Follows", trendline="ols", title="Relationship Between Profile Visits and Followers Gained")
figure.show()


# In[ ]:


#Instagram reach prediction model


# In[132]:


model = PassiveAggressiveRegressor(max_iter=1000, random_state=42)
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(f"Model Score: {score}")


# In[133]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])


# In[134]:


#features = "Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"


# In[135]:


model.predict(features)

