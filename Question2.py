import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
import seaborn as sns
import warnings

warnings.simplefilter("ignore")

# load the data
df = pd.read_csv('College.csv')
print(df.head())
print("\n")

print(df.columns.values)

# For identifying no.of null values in the customer set
df.isna().head()

print("NULL values in the train ")
print(df.isna().sum())
print("\n")

# Fill missing or null values with mean column values in the customer set
df.fillna(df.mean(), inplace=True)

df.info()
print("\n")
#
# # Encoding categorical data
# from sklearn.preprocessing import LabelEncoder
# from sklearn import preprocessing
#
# labelEncoder = preprocessing.LabelEncoder()
# labelEncoder.fit(df['Gender'])
# df['Gender'] = labelEncoder.transform(df['Gender'])
#
# #removing the features not correlated to the data
# df.drop(["CustomerID"], axis=1, inplace=True)
#
# df.info()
# print("\n")

#using Age, Annual Income and Spending Score for clustering customers
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("white")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Apps, df["Accept"], df["Enroll"], c='blue', s=60)
ax.view_init(30, 185)
plt.xlabel("Apps")
plt.ylabel("Accept")
ax.set_zlabel('Enroll')
plt.show()

#appliying k-means and calucuating silhoutte score
from sklearn.cluster import KMeans

wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:, [2, 4]])
    wcss.append(kmeans.inertia_)
    score = silhouette_score(df.iloc[:,[2,4]], kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(k, score))

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(2, 11), wcss, linewidth=2, color="red", marker="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1, 11, 1))
plt.ylabel("WCSS")
plt.show()

#kmeans clusturring
km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,[2,4]])

df["label"] = clusters
#cluturing represntation of
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Apps[df.label == 0], df["Accept"][df.label == 0], df["Enroll"][df.label == 0], c='blue', s=60)
ax.scatter(df.Apps[df.label == 1], df["Accept"][df.label == 1], df["Enroll"][df.label == 1], c='red', s=60)
ax.scatter(df.Apps[df.label == 2], df["Accept"][df.label == 2], df["Enroll"][df.label == 2], c='green', s=60)
ax.scatter(df.Apps[df.label == 3], df["Accept"][df.label == 3], df["Enroll"][df.label == 3], c='orange', s=60)
ax.scatter(df.Apps[df.label == 4], df["Accept"][df.label == 4], df["Enroll"][df.label == 4],  c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Apps")
plt.ylabel("Accept")
ax.set_zlabel('Enroll')
plt.show()