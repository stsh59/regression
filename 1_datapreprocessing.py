import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("datapreprocessing_dataset.csv")

# here we divide our dataset into two parts
# our purpose is to predict wheather the developer is married or not based on top three parameters

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

# Handling the missing values in the dataset
# we use scikit learn for doing this

from sklearn.impute import SimpleImputer

s_imputer= SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
s_imputer=s_imputer.fit(x[:,1:3])
x[:,1:3]=s_imputer.transform(x[:,1:3])


# Converting the string data into numbers in x variable and y variable
# also creating dummy variable through one hot encoding

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

columntrans_x = ColumnTransformer(transformers= [('one_hot_encoder_x', OneHotEncoder(), [0])], remainder='passthrough')
x = columntrans_x.fit_transform(x)


# Splitting the dependent and independent datas into training data and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# standardizing the training data or normalization as the data in multiple columns are highly dispersed

from sklearn.preprocessing import StandardScaler
st_sc_x = StandardScaler()
x_train = st_sc_x.fit_transform(x_train)
x_test = st_sc_x.fit_transform(x_test)

 


