from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")

X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
# print(ls)

# ls.plot(kind='scatter', grid=True ,
#         x = "GDP per capita (USD)", y = "Life satisfaction")
# plt.axis([23500, 62500, 4 , 9])
# plt.show()
#
model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=3) # k = 3
model.fit(X,y)

X_new = [[31721.3]]
print(model.predict(X_new))
# LinearRegression 5.90
# KNeighborsRegressor 5.79