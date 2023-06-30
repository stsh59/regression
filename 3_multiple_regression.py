import pandas as pd

dataframe = pd.read_csv("Multiple_Regression_dataset.csv")

# divide the dataframe into dependent and independent variable

x = dataframe.iloc[:,:-1].values
y = dataframe.iloc[:,3].values

# dividing the datasets into trainind data and test data sets

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(x, test_size=0.3, random_state=0)
y_train, y_test = train_test_split(y, test_size=0.3, random_state=0)

# implementing multiple regression
# same code used for linear regression can be executed for mltiple regresssion as well

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y_train)

# use the fitted dataset for prediction along x_test

y_predict = reg.predict(x_test)





