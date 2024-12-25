# Завдання 2.2. Кластеризація K-середніх для набору даних Iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Завантаження набору даних Iris
iris = load_iris()
X = iris['data']  # Атрибути (довжина та ширина чашолистка і пелюстки)
y = iris['target']  # Цільова змінна (класи квітів)

# Ініціалізація та налаштування моделі K-середніх
# Створюємо об'єкт кластеризації з 3 кластерами (за кількістю класів у наборі даних Iris)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=42)

# Навчання моделі на даних
kmeans.fit(X)

# Отримання передбачених кластерів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації
# Малюємо точки даних, розфарбовуючи їх відповідно до кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Відображення центрів кластерів
centers = kmeans.cluster_centers_  # Координати центрів кластерів
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, label='Centers')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Додаткова функція для пошуку кластерів
# Реалізація алгоритму K-середніх вручну для розуміння
from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    """Знаходження кластерів вручну"""
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # Визначення міток на основі найближчих центрів
        labels = pairwise_distances_argmin(X, centers)
        # Обчислення нових центрів як середнього значення точок у кожному кластері
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # Перевірка на зупинку, якщо центри більше не змінюються
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# Використання функції для знаходження 3 кластерів
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Custom K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
