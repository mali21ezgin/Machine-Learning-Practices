import kagglehub
import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Download latest version
path = kagglehub.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "Titanic-Dataset.csv")
                        
df = pd.read_csv(csv_file)

numpy_array = df.values

y = df["Survived"]
X = df.drop(["Survived","PassengerId","Name", "Ticket","Cabin"], axis = 1)
X = pd.get_dummies(X,drop_first=True)


print(X.isna().sum())


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state=42 )

train_median_age = X_train["Age"].median()
X_train["Age"] = X_train["Age"].fillna(train_median_age)
X_test ["Age"] = X_test["Age"].fillna(train_median_age)

print(X_train.isna().sum())


scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

ns = []
knn_accuracys = []

for n in range (1,11):
    knn = KNeighborsClassifier(n_neighbors=n, p=2)

    

    knn.fit(X_train_sc, y_train)
    knn_y_pred = knn.predict(X_test_sc)

    knn_accuracy_score = accuracy_score(y_test, knn_y_pred)
    ns.append(n)
    knn_accuracys.append(knn_accuracy_score)
    
max_accuracy_knn = max(knn_accuracys)
max_index = knn_accuracys.index(max_accuracy_knn)
max_n = ns[max_index]

print(f" Best knn models accuracy score: {max_accuracy_knn}, best n value: {max_n} ")

knn = KNeighborsClassifier(n_neighbors=max_n, p=2)
knn.fit(X_train_sc, y_train)
knn_y_pred = knn.predict(X_test_sc)
print("Knn models performance:")
print(classification_report(y_test, knn_y_pred))

depths = []
depth_accuracys = []
for n in range(1,11):
    dec_tree = DecisionTreeClassifier(max_depth=n)
    dec_tree.fit(X_train, y_train)
    dec_y_pred = dec_tree.predict(X_test)

    depth_accruacy_score = accuracy_score(y_test, dec_y_pred)
    depths.append(n)
    depth_accuracys.append(depth_accruacy_score)
    
max_accuracy_depth = max(depth_accuracys)
max_index_depth = depth_accuracys.index(max_accuracy_depth)
max_depth = depths[max_index_depth]

print(f" Best depth: {max_depth}, accuracy :{max_accuracy_depth}  ")



splits = []
split_accuracys = []
for n in range(2,11):
    dec_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split= n)
    dec_tree.fit(X_train, y_train)
    dec_y_pred = dec_tree.predict(X_test)

    split_accruacy_score = accuracy_score(y_test, dec_y_pred)
    splits.append(n)
    split_accuracys.append(split_accruacy_score)
    
max_accuracy_split = max(split_accuracys)
max_index_split = split_accuracys.index(max_accuracy_split)
max_split = splits[max_index_split]

print(f" Best split: {max_split}, accuracy :{max_accuracy_split}  ")


leafs = []
leaf_accuracys = []
for n in range(1,11):
    dec_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split= max_split, min_samples_leaf=n)
    dec_tree.fit(X_train, y_train)
    dec_y_pred = dec_tree.predict(X_test)

    leaf_accruacy_score = accuracy_score(y_test, dec_y_pred)
    leafs.append(n)
    leaf_accuracys.append(leaf_accruacy_score)
    
max_accuracy_leaf = max(leaf_accuracys)
max_index_leaf = leaf_accuracys.index(max_accuracy_leaf)
max_leaf = leafs[max_index_leaf]

print(f" Best leaf: {max_leaf}, accuracy :{max_accuracy_leaf}  ")


dec_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split= max_split, min_samples_leaf=max_leaf)
dec_tree.fit(X_train, y_train)
dec_y_pred = dec_tree.predict(X_test)

dec_accuracy= accuracy_score(y_test, dec_y_pred)

print(f" Best decision tree models accuracy score: {dec_accuracy}")
print("decision tree models performance:")
print(classification_report(y_test, dec_y_pred))

estimators = []
estimator_accuracys = []
for n in range(50,300,50):
    rf_clf = RandomForestClassifier(n_estimators = n, max_depth = 5, min_samples_split=2)
    rf_clf.fit(X_train, y_train)
    rf_y_pred = rf_clf.predict(X_test)

    estimator_accruacy_score = accuracy_score(y_test, rf_y_pred)
    estimators.append(n)
    estimator_accuracys.append(estimator_accruacy_score)
    
max_accuracy_estimator = max(estimator_accuracys)
max_index_estimator = estimator_accuracys.index(max_accuracy_estimator)
max_estimator = estimators[max_index_estimator]

print(f" Best estimators: {max_estimator}, accuracy :{max_accuracy_estimator}")


rf_depths = []
rf_depth_accuracys = []
for n in range(1,11):
    rf_dec_tree = RandomForestClassifier(n_estimators =max_estimator, max_depth=n)
    rf_dec_tree.fit(X_train, y_train)
    rf_dec_y_pred = rf_dec_tree.predict(X_test)

    rf_depth_accruacy_score = accuracy_score(y_test, rf_dec_y_pred)
    rf_depths.append(n)
    rf_depth_accuracys.append(rf_depth_accruacy_score)
    
rf_max_accuracy_depth = max(rf_depth_accuracys)
rf_max_index_depth = rf_depth_accuracys.index(rf_max_accuracy_depth)
rf_max_depth = rf_depths[rf_max_index_depth]

print(f" rf_ Best depth: {rf_max_depth}, rf_accuracy :{rf_max_accuracy_depth}  ")

rf_dec_tree = RandomForestClassifier(n_estimators =max_estimator, max_depth= rf_max_depth)
rf_dec_tree.fit(X_train, y_train)
rf_dec_y_pred = rf_dec_tree.predict(X_test)
print(f" rf_ Best accuracy: {rf_max_accuracy_depth} ")
print(f"rf prformance:")
print(classification_report(y_test, rf_dec_y_pred))


svm_c_accuracys = []
C = [0.1, 1, 10]
for c in C:
    svm_c = SVC(kernel = "rbf", C=c) 
    svm_c.fit(X_train_sc, y_train)
    swm_y_pred =  svm_c.predict(X_test_sc)
    
    svm_accuracy_score = accuracy_score(y_test, swm_y_pred)
    svm_c_accuracys.append(svm_accuracy_score)
    
svm_max_accuracy_c = max(svm_c_accuracys)
svm_max_index_c = svm_c_accuracys.index(svm_max_accuracy_c)
svm_max_c = C[svm_max_index_c]


print(f" Best svm models accuracy score: {svm_max_accuracy_c}, best c value: {svm_max_c} ")


svm_c = SVC(kernel = "rbf", C=svm_max_c) 
svm_c.fit(X_train_sc, y_train)
swm_y_pred =  svm_c.predict(X_test_sc)

print("svm models performance:")
print(classification_report(y_test, swm_y_pred))

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_y_pred = nb.predict(X_test)
print(f"navie bayes models accuracy : {accuracy_score(y_test, nb_y_pred)}")
print("bayes models performance:")
print(classification_report(y_test, nb_y_pred))










































