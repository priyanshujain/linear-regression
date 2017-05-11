import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
df = pd.read_csv('challenge_dataset.txt', names=['X','Y'])
x_values = df[['X']]
y_values = df[['Y']]

#train model on data
reg = linear_model.LinearRegression()
reg.fit(x_values, y_values)

#results
print reg.score(x_values,y_values)

# The coefficients
print('Coefficients: ', reg.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((reg.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(x_values, y_values))

#Visualize Results
plt.scatter(x_values, y_values)
plt.plot(x_values, reg.predict(x_values))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Challenge Dataset')
plt.show()
