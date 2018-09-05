import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

line = plt.figure()
x = np.random.normal(0, 10, 1000)
y = 3 * x + np.random.normal(0, 10, 1000)

plt.scatter(x,y,s=5)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

pca = PCA()
npIn = np.column_stack([x,y])

#Uses the data to obtain the parameters of PCA (eigenvectors and eigenvalues)
pca.fit(npIn)

variancePCA = pca.explained_variance_ratio_
		
pca.n_components = 2
	
aux1_x_reduced = pca.fit(npIn).transform(npIn)

plt.figure(3)
plt.scatter(aux1_x_reduced[:,0],aux1_x_reduced[:,1], s=5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
