import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
df = pd.read_csv('challenge_dataset.txt', names=['X','Y'])
x_values = np.asarray(df['X'])
y_values = np.asarray(df['Y'])

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values.value.reshape(-1, 1), y_values.value.reshape(-1, 1))

#visualize results
plt.scatter(x_values.value.reshape(-1, 1), y_values.value.reshape(-1, 1))
plt.plot(x_values.value.reshape(-1, 1), body_reg.predict(x_values.value.reshape(-1, 1)))
plt.show()
