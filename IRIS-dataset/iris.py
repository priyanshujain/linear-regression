import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first three features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


#train model on data
reg = linear_model.LinearRegression()
reg.fit(X, Y)

#results
print reg.score(X, Y)



#Visualize Results
fig = plt.figure()
fig.set_size_inches(12.5,7.5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], Y, c='g', marker= 'o')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Orignal Dataset')
#ax.view_init(10, -45)

fig1 = plt.figure()
fig1.set_size_inches(12.5,7.5)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], reg.predict(X), c='r', marker= 'o')
#ax.plot_surface(X[:,0],X[:,1], reg.predict(X), cmap=plt.cm.hot, color='b', alpha=0.2);
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Predicted Dataset')
plt.show()
