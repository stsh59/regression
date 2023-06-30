import numpy as np
import pandas as pd

dataframe = pd.read_csv("polynomial_svr_reg_dataset.csv")
x = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,1].values


#import matplotlib.pyplot as plt

#plt.scatter(x, y)
#plt.show()

# since the datasets produce a graph as a non linear line or a curve.
# so we use polynomial regression here

# implementing polynomial regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)

# Creating the graph using matplotlib

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(x, lin_reg.predict(x_poly), color='blue')


# predicting the salary for different levels

predicted_val = lin_reg.predict(poly_reg.fit_transform([[7]]))
print(predicted_val)