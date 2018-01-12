import numpy as np

def hist_intersection_kernel(x, y):
    n_samples_1, n_features = x.shape
    n_samples_2, _ = y.shape

    K = np.zeros((n_samples_1, n_samples_2))

    for i in range(n_samples_1):
        for j in range(n_samples_2):
            K[i, j] = np.minimum(x[i, :], y[j, :]).sum()

    return K