import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import adjusted_rand_score

# Завантаження набору даних Iris
iris = load_iris()
X = iris.data  # Атрибути (довжина/ширина чашолистка та пелюстки)
y = iris.target  # Цільова змінна (класи квітів)

# Визначення параметрів K-середніх
n_clusters = 3  # Кількість кластерів

# Створення та навчання моделі K-середніх
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Передбачення кластерів для даних
y_pred = kmeans.predict(X)

# Візуалізація результатів у 3D
fig = plt.figure(1, figsize=(8, 6))  # Збільшено розмір фігури для кращого вигляду
ax = fig.add_subplot(111, projection='3d')  # Створення 3D-графіка
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y_pred, edgecolor='k', cmap='viridis')  # Використання кольорової карти для кластерів

# Налаштування підписів осей
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')

# Додавання заголовка
plt.title("K Means clustering on Iris dataset")
plt.show()

# Оцінка якості кластеризації за Adjusted Rand Index
print("Adjusted Rand Index:", adjusted_rand_score(y, y_pred))
