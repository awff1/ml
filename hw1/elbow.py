import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, normalize

def compute_wcss(X, labels, centers):
    wcss = 0
    
    for i in range(len(X)):
        center = centers[labels[i]]
        diff = X[i] - center
        wcss += np.sum(diff ** 2)
    
    return wcss

def elbow_method(X, max_k=10):
    wcss_values = []

    ks = range(1, max_k + 1)

    for k in ks:
        labels, centers = kmeans(X, k)
        wcss = compute_wcss(X, labels, centers)
        wcss_values.append(wcss)

    plt.plot(ks, wcss_values, marker='o')
    plt.xlabel("k")
    plt.ylabel("WCSS")
    plt.title("Метод локтя")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    X = df.values
    X = normalize(X)

    elbow_method(X, 20)