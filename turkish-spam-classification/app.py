import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from kagglehub import KaggleDatasetAdapter


import numpy as np
import matplotlib.pyplot as plt

# Set the path to the file you'd like to load
file_path = "trspam.csv"

# Load the latest version with pandas_kwargs to handle formatting/bad lines
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "cuneytdemir/turkish-spam-dataset",
    file_path,
    pandas_kwargs={
        "on_bad_lines": "skip",  # Bozuk/kaymış satırları atlar
        "encoding": "utf-8"       # Türkçe karakter sorununu önler
    }
)
print("Veri setinin toplam satır ve sütun sayısı:", df.shape)
print("First 5 records:")
print(df.head())


df = df[["0.0", "0.0.1"]]
bosMu = df.isna().any().any()
print("datasetde boş veriler var mı ? ", bosMu)

df = df.dropna()
print("boş verilere silindikten sonra kalan veri sayısı: ", df.shape)

X = df["0.0"]
y = df["0.0.1"]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

vektorizer = TfidfVectorizer()
X_train_vekt = vektorizer.fit_transform(X_train)
X_test_vekt = vektorizer.transform(X_test)

tree = DecisionTreeClassifier()

kf = KFold(n_splits= 5)

tree_param_grid = {"criterion": ["gini", "entropy"],
                   "max_depth": [5,10,15],
                   "min_samples_split": [3,6,9],
                   "min_samples_leaf": [1,5,7]
                   }
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv= kf, n_jobs=-1)

tree_grid_search.fit(X_train_vekt, y_train)

print("bet tree modal params : ",tree_grid_search.best_params_ )
print ("best tree model accuracy score: ",tree_grid_search.best_score_ )


tree_best = tree_grid_search.best_estimator_
y_pred_best = tree_best.predict(X_test_vekt)


fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    tree_best, 
    X_test_vekt, 
    y_test, 
    display_labels=["Normal", "Spam"],
    cmap=plt.cm.Blues,
    ax=ax
)

plt.title("En İyi Karar Ağacı - Karmaşıklık Matrisi", fontsize=12)
plt.xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=10)
plt.ylabel("Gerçek Sınıf (True)", fontsize=10)
plt.show()

tree_best_class =classification_report(y_test,y_pred_best ) 
print(tree_best_class)

#random forest 

rf = RandomForestClassifier()


rf_param_grid = {"n_estimators": [50,100,200],
                   "max_depth": [5,10,15],
                   "min_samples_split": [3,6,9],
                   "min_samples_leaf": [1,5,7]
                   }
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv= kf, n_jobs=-1)

rf_grid_search.fit(X_train_vekt, y_train)

print("bet rf model params : ",rf_grid_search.best_params_ )
print ("best rf model accuracy score: ",rf_grid_search.best_score_ )


rf_best = rf_grid_search.best_estimator_
y_pred_best_rf = rf_best.predict(X_test_vekt)


fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    rf_best, 
    X_test_vekt, 
    y_test, 
    display_labels=["Normal", "Spam"],
    cmap=plt.cm.Blues,
    ax=ax
)

plt.title("En İyi Random Forest - Karmaşıklık Matrisi", fontsize=12)
plt.xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=10)
plt.ylabel("Gerçek Sınıf (True)", fontsize=10)
plt.show()

rf_best_class =classification_report(y_test,y_pred_best_rf ) 
print(rf_best_class)

# nb

nb = MultinomialNB()


nb.fit(X_train_vekt, y_train)


y_pred_best_nb = nb.predict(X_test_vekt)


fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    nb, 
    X_test_vekt, 
    y_test, 
    display_labels=["Normal", "Spam"],
    cmap=plt.cm.Blues,
    ax=ax
)

plt.title("En İyi Navie Bayes - Karmaşıklık Matrisi", fontsize=12)
plt.xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=10)
plt.ylabel("Gerçek Sınıf (True)", fontsize=10)
plt.show()

nb_best_class =classification_report(y_test,y_pred_best_nb) 
print(nb_best_class)

#logistic regression

log_reg = LogisticRegression(solver='saga')


log_reg_param_grid = {"penalty": ["l1","l2"],
                   "C": [1,0.1,10],
                   "max_iter": [50,100,150,300]
                   }
log_reg_grid_search = GridSearchCV(log_reg, log_reg_param_grid, cv= kf, n_jobs=-1)

log_reg_grid_search.fit(X_train_vekt, y_train)

print("bet logistic regression model params : ",log_reg_grid_search.best_params_ )
print ("best logistic regression model accuracy score: ",log_reg_grid_search.best_score_ )


log_reg_best = log_reg_grid_search.best_estimator_
y_pred_best_log_reg = log_reg_best.predict(X_test_vekt)


fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    log_reg_best, 
    X_test_vekt, 
    y_test, 
    display_labels=["Normal", "Spam"],
    cmap=plt.cm.Blues,
    ax=ax
)

plt.title("En İyi Logistic Regression - Karmaşıklık Matrisi", fontsize=12)
plt.xlabel("Tahmin Edilen Sınıf (Predicted)", fontsize=10)
plt.ylabel("Gerçek Sınıf (True)", fontsize=10)
plt.show()

log_reg_best_class =classification_report(y_test,y_pred_best_log_reg ) 
print(log_reg_best_class)







