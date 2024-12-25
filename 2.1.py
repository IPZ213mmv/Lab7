import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Завантаження даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Визначення кількості кластерів
num_clusters = 5

# Створення моделі K-means та навчання
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(X)

# Передбачення кластерів для кожного об'єкта
y_kmeans = kmeans.predict(X)

# Візуалізація результатів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.title('Кластеризація K-means')
plt.show()