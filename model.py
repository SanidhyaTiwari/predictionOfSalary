import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

x = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x.values, y.values)

pickle.dump(regressor, open('sandy.pkl','wb'))


model = pickle.load(open('sandy.pkl','rb'))
print(model.predict([[2, 9, 6]]))


