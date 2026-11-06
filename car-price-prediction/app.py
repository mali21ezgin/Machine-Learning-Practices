import kagglehub
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Download latest version
path = kagglehub.dataset_download("akshaydattatraykhare/car-details-dataset")

print("Path to dataset files:", path)

csv_file = os.path.join(path, "CAR DETAILS FROM CAR DEKHO.csv")

db = pd.read_csv((csv_file))

y = db["selling_price"]

X = db.drop(["selling_price","name"] , axis = 1)
X["name"] = db.name.str.split().str[0]
X = pd.get_dummies(X,drop_first=True)

a =  db.isna().any().any()

print(a)

c_lin_reg = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


c_lin_reg.fit(X_train, y_train)

y_pred_c_lin = c_lin_reg.predict(X_test)

rmse_c_l = root_mean_squared_error(y_test, y_pred_c_lin)

mse_c_l = mean_squared_error(y_test, y_pred_c_lin)


print (f"çoklu lineer regresyon rmse = {rmse_c_l}, mse = {mse_c_l}")


degres = []
rmses = []

for i in range(2,3):

    poly = PolynomialFeatures(degree= i)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    p_lin_reg = LinearRegression()
    p_lin_reg.fit(X_train_poly, y_train)
    p_pred =  p_lin_reg.predict(X_test_poly)
    
    
  
    rmse = root_mean_squared_error(y_test, p_pred)

    degres.append(i)
    rmses.append(rmse)
    
min_rmse = min(rmses)
min_index = rmses.index(min_rmse)
min_degre = degres[min_index]

print (f"polinomal lineer regresyon rmse = {min_rmse}, best degree= {min_degre}")

