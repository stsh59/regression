import numpy as np
import pandas as pd

dataframe = pd.read_csv("polynomial_svr_reg_dataset.csv")

x = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,1].values


from sklearn.tree import DecisionTreeRegressor

dtr_obj = DecisionTreeRegressor(random_state=0)
dtr_obj.fit(x,y)


import matplotlib.pyplot as plt

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y)
plt.plot(x_grid, dtr_obj.predict(x_grid), color='red')
plt.title('Level vs Salary graph')
plt.xlabel('Level')
plt.ylabel('Salary')

predict_val = dtr_obj.predict(np.reshape(6.5, (-1,1)))
print(predict_val)