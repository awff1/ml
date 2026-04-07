import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def init_centers(X, k):
    n = X.shape[0]
    indexes = np.random.choice(n, k, replace=False)
    centers = X[indexes]
    return centers



def compute_distances(X, centers):
    return np.sqrt(np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))


def assign_clusters(distances):
    return np.argmin(distances, axis=1)


def update_centers(X, labels, k):
    new_centers = []
    
    for i in range(k):
        points = X[labels == i]
        
        if len(points) == 0:
            new_centers.append(X[np.random.randint(0, X.shape[0])])
        else:
            new_centers.append(points.mean(axis=0))
    
    return np.array(new_centers)


def kmeans(X, k, max_iterations=100):
    centers = init_centers(X, k)
    
    for i in range(max_iterations):
        
        distances = compute_distances(X, centers)
        
        labels = assign_clusters(distances)
        
        new_centers = update_centers(X, labels, k)
        
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    return labels, centers


if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    X = df.values
    X = normalize(X)

    labels, centers = kmeans(X,5)


    X_plot = X[:, [2, 6]]
    for i in range(5):
        points = X_plot[labels == i]
        plt.scatter(points[:, 1], points[:, 0], label=f"Cluster {i}")

    centers_plot = centers[:, [2, 6]]
    plt.scatter(centers_plot[:, 1], centers_plot[:, 0],
                marker='x', s=200, label='Centers')

    plt.xlabel("7")
    plt.ylabel("3")
    plt.title("Кластеры")
    plt.legend()
    plt.show()