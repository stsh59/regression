import numpy as np
import pandas as pd

dataframe = pd.read_csv("polynomial_svr_reg_dataset.csv")

x = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,1].values

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
std_x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
std_y = scaler_y.fit_transform(np.reshape(y,(-1,1)))

# applying svr algorithm to the standarized dataset

from sklearn.svm import SVR

sv_reg = SVR(kernel='rbf')
sv_reg.fit(std_x, np.ravel(std_y))

# viewing the plot diagram

import matplotlib.pyplot as plt

plt.scatter(std_x, std_y) 
plt.plot(std_x, sv_reg.predict(std_x), color='red')
plt.title('Level vs Salary graph')
plt.xlabel('Level')
plt.ylabel('Salary')

# Creating the prediction on the basis of random value for each value of x

final_predicted_value = np.array([])

for i in x: 
  predicted_value = sv_reg.predict(scaler_x.transform(np.reshape(i,(1,1))))
  print(predicted_value)

# Converting the standardarized predicted value into our own readable value
  
  final_predicted_value = np.append(final_predicted_value, np.array(scaler_y.inverse_transform(predicted_value)))
  print(final_predicted_value)




