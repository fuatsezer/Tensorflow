import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
x = [0,1,2]
y=[100,200,300]
plt.plot(x,y)
plt.show()
#%%
housing = pd.DataFrame({"rooms":[1,1,2,2,2,3,3,3],
                        "price":[100,120,190,200,230,310,330,305]})
#%%
print(housing.head(10))
#%%
plt.scatter("rooms","price",data = housing,marker="x")
plt.title("Title")
plt.xlabel("X label")
plt.xlim(0,2)
plt.ylim(0,200)
plt.show()
#%%
df = pd.read_csv("Heart.csv")
#%%
plt.figure(figsize=(10,10))
sns.distplot(df["Age"],bins=50,color="red")
plt.show()
#%%
sns.countplot(x="Sex",data=df,hue="Ca",palette="terrain")
plt.show()
#%%
sns.boxplot(x="Sex",y="Age",data=df,hue="Ca")
plt.show()
#%%
sns.scatterplot(x="Chol",y="RestBP",data=df,hue="Sex")
#%%
sns.regplot(x="Chol",y="RestBP",data=df)
#%%
sns.pairplot(df,hue="Ca")