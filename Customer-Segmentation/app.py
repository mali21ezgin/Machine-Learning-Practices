# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Set the path to the file you'd like to load
file_path = "Mall_Customers.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "vjchoudhary7/customer-segmentation-tutorial-in-python",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

db = df.values


plt.figure()
plt.scatter(db[:,3], db[:,4])
plt.xlabel("Gelir")
plt.ylabel("Harcama skoru")
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=8)
X = db[:, [3,4]]
kmeans_pred = kmeans.fit_predict(X)

plt.figure()
plt.title("Kmeans")
plt.scatter(db[:,3], db[:,4], c = kmeans_pred)
plt.xlabel("Gelir")
plt.ylabel("Harcama skoru")
plt.legend()
plt.show()


hierarchical = AgglomerativeClustering(n_clusters=5)
hier_pred = hierarchical.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=hier_pred, cmap="viridis", s=50)
plt.xlabel("Gelir")
plt.ylabel("Harcama skoru")
plt.title("Hiyerarşik Kümeleme (Agglomerative) Sonuçları")
plt.show()


dbscan = DBSCAN(eps=5.0, min_samples=5)
dbscan_pred = dbscan.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_pred, cmap="viridis", s=50)
plt.xlabel("Gelir")
plt.ylabel("Harcama skoru")
plt.title("DBSCAN Müşteri Segmentasyonu (Aykırı Değerler Dahil)")
plt.show()