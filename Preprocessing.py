# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import Data and print.
df = pd.read_csv("House_Price.csv", header=0)
print(df)

# Run EDD
print(df.describe())

#plot

sns.jointplot(x="n_hot_rooms",y="price",data=df)
plt.show()

sns.countplot(x='airport',data=df)
plt.show()

# Outliers

df.info()
# Upper Limit
uv = np.percentile(df.n_hot_rooms,[99])[0]
print(uv)
df.n_hot_rooms[df.n_hot_rooms>3*uv]=3*uv

# Lower Limit

lv= np.percentile(df.rainfall,[1])[0]
print(lv)
df[df.rainfall<lv]
df.rainfall[(df.rainfall<0.3*lv)]=0.3*lv

sns.jointplot(x="crime_rate",y="price",data=df)
plt.show()

# Treating null values
df.info()
df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())
df.info()

sns.jointplot(x="crime_rate",y="price", data=df)
plt.show()

df.crime_rate=np.log(1+df.crime_rate)
sns.jointplot(x="crime_rate",y="price", data=df)
plt.show()

# Average
df['avg_dist'] = (df.dist1+df.dist2+df.dist3+df.dist4)/4
print(df.describe())

# # Delete the dist
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
del df['bus_ter']
print(df.describe())

df = pd.get_dummies(df)
del df['airport_NO']
del df['waterbody_None']
print(df.head())

# Corelation
print(df.corr())
del df['parks']
print(df.corr())

