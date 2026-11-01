import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error



db = pd.read_csv("Housing.csv")

print (db.head(4))


y = db["price"]
X = db.drop("price", axis = 1)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)



rmse_list = []
depth_list = []
for depth in range(1,11):
    decisionTreeRegr = DecisionTreeRegressor(max_depth = depth)
    decisionTreeRegr.fit(X_train, y_train)
    decisionTreeRegr_pred = decisionTreeRegr.predict(X_test)
    rmse_decisionTreeRegr = root_mean_squared_error(y_test,decisionTreeRegr_pred)
    rmse_list.append(rmse_decisionTreeRegr)
    depth_list.append(depth)
    


min_rmse_depth = min(rmse_list)
index_rmse_depth =rmse_list.index(min_rmse_depth)
best_depth = depth_list[index_rmse_depth]

print(f"Karar ağaçlarının en iyi depth'deki değeri : {best_depth} ve rmse = {min_rmse_depth}")

rmse_list = []
split_list = []
for split in range(2,11):
    decisionTreeRegr = DecisionTreeRegressor(max_depth = best_depth, min_samples_split=split)
    decisionTreeRegr.fit(X_train, y_train)
    decisionTreeRegr_pred = decisionTreeRegr.predict(X_test)
    rmse_decisionTreeRegr = root_mean_squared_error(y_test,decisionTreeRegr_pred)
    rmse_list.append(rmse_decisionTreeRegr)
    split_list.append(split)
    


min_rmse_split = min(rmse_list)
index_rmse_split =rmse_list.index(min_rmse_split)
best_split = split_list[index_rmse_split]

print(f"Karar ağaçlarının en iyi split'deki değeri : {best_split} ve rmse = {min_rmse_split}")


rmse_list = []
leaf_list = []
for leaf in range(2,11):
    decisionTreeRegr = DecisionTreeRegressor(max_depth = best_depth, min_samples_split=best_split, min_samples_leaf=leaf)
    decisionTreeRegr.fit(X_train, y_train)
    decisionTreeRegr_pred = decisionTreeRegr.predict(X_test)
    rmse_decisionTreeRegr = root_mean_squared_error(y_test,decisionTreeRegr_pred)
    rmse_list.append(rmse_decisionTreeRegr)
    leaf_list.append(leaf)
    


min_rmse_leaf = min(rmse_list)
index_rmse_leaf =rmse_list.index(min_rmse_leaf)
best_leaf = leaf_list[index_rmse_leaf]

print(f"Karar ağaçlarının en iyi leaf'deki değeri : {best_leaf} ve rmse = {min_rmse_leaf}")


decisionTreeRegr = DecisionTreeRegressor(max_depth = best_depth, min_samples_split=best_split, min_samples_leaf=best_leaf)
decisionTreeRegr.fit(X_train, y_train)
decisionTreeRegr_pred = decisionTreeRegr.predict(X_test)
rmse_decisionTreeRegr = root_mean_squared_error(y_test,decisionTreeRegr_pred)
mse_decisionTreeRegr = mean_squared_error(y_test,decisionTreeRegr_pred) 

print(f"Karar ağaçlarında en iyi rmse: {rmse_decisionTreeRegr} en iyi mse : {mse_decisionTreeRegr}")





k_list = []
rmse_list = []
for k in range(1,11):
  
    kNN = KNeighborsRegressor(n_neighbors=k)
    kNN.fit(X_train_sc, y_train)
    kNN_pred = kNN.predict(X_test_sc)
    rmse_KNN = root_mean_squared_error(y_test,kNN_pred)
    rmse_list.append(rmse_KNN)
    k_list.append(k)
    
min_rmse_k = min(rmse_list)
index_rmse_k = rmse_list.index(min_rmse_k)
best_k = k_list[index_rmse_k]

print(f"KNN için en iyi k değeri : {best_k} ve en iyi rmse = {min_rmse_k}")

kNN = KNeighborsRegressor(n_neighbors=best_k)
kNN.fit(X_train_sc, y_train)
kNN_pred = kNN.predict(X_test_sc)
rmse_KNN = root_mean_squared_error(y_test,kNN_pred)
mse_KNN = mean_squared_error(y_test,kNN_pred)

print(f"KNN için  en iyi rmse: {rmse_KNN} en iyi mse : {mse_KNN}")


    





    















