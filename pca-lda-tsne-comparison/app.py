from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("akshaydattatraykhare/car-details-dataset")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "CAR DETAILS FROM CAR DEKHO.csv")

db = pd.read_csv((csv_file))


y = pd.qcut( db["selling_price"],q =3, labels = [0,1,2])



X = db.drop(["selling_price","name"] , axis = 1)
X["name"] = db.name.str.split().str[0]
X = pd.get_dummies(X,drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



pca = PCA(n_components=2)  # 3 adet temel bilesen (PC)
X_pca = pca.fit_transform(X_scaled)

lda = LinearDiscriminantAnalysis(n_components= 2)
X_lda = lda.fit_transform(X_scaled,y)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca[:,0],X_pca[:,1],c = y, cmap = "tab10", alpha = 0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Car Dataset Dataset")
plt.legend()

plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],c = y, cmap = "tab10", alpha = 0.6)
plt.title("LDA of Car Dataset dataset")

plt.xlabel("LD1")
plt.ylabel("LD2")

plt.figure()
plt.scatter(X_tsne[:,0],X_tsne[:,1],c = y, cmap = "tab10", alpha = 0.6)
plt.title("T-SNE Dimension ")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")





