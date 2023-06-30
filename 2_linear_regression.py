import pandas as pd

dataset = pd.read_csv("linear_regression_dataset.csv")

# divide the whole dataset into dependent and independent variables

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# split both dependent and independent variables datas into training and testing data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)

# linear_regression

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)

# plot the scatter plot graph using matplotlib

import matplotlib.pyplot as plt

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='red')
plt.title("experience vs salary graph")
plt.xlabel("experience of employeee")
plt.ylabel("salary of employee")
plt.show()

plt.scatter(x_test, y_test)
plt.plot(x_test, reg.predict(x_test), color='red')
plt.title("experience vs salary graph")
plt.xlabel("experience of employeee")
plt.ylabel("salary of employee")
plt.show()

