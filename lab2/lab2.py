import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

N = 100
BLUE_CLR = '#1f77b4'
ORANGE_CLR = '#ff7f0e'

def generate_circle_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    noise = 0

    center_angles = np.random.uniform(0, 2 * np.pi, N)
    center_radius = np.random.uniform(0, 0.5, N)
    center_x = center_radius * np.cos(center_angles) + np.random.normal(0, noise, N)
    center_y = center_radius * np.sin(center_angles) + np.random.normal(0, noise, N)
    center_labels = np.full(N, BLUE_CLR)

    boundary_angles = np.random.uniform(0, 2 * np.pi, N)
    boundary_radius = np.random.uniform(1, 1.5, N)
    boundary_x = boundary_radius * np.cos(boundary_angles) + np.random.normal(0, noise, N)
    boundary_y = boundary_radius * np.sin(boundary_angles) + np.random.normal(0, noise, N)
    boundary_labels = np.full(N, ORANGE_CLR)

    labels = np.concatenate((center_labels, boundary_labels))
    samples = np.vstack((np.column_stack((center_x, center_y)), np.column_stack((boundary_x, boundary_y))))

    return samples, labels

def generate_xor_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-6, 6, N)  # Генерируем случайные точки в квадрате [0, 1] x [0, 1]
    y = np.random.uniform(-6, 6, N)
    samples = np.column_stack((x, y))
    values = np.logical_xor(samples[:, 0] > 0, samples[:, 1] > 0).astype(int)  # XOR логика
    labels = np.where(values > 0, ORANGE_CLR, BLUE_CLR)
    print(samples)
    return samples, labels

def generate_gaussian_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    mean1 = [0, 0]
    cov1 = [[0.1, 0], [0, 0.1]]
    mean2 = [1, 1]
    cov2 = [[0.1, 0], [0, 0.1]]
    class1 = np.random.multivariate_normal(mean1, cov1, N)

    class2 = np.random.multivariate_normal(mean2, cov2, N)

    x = np.vstack((class1, class2))

    labels = np.hstack((np.full(N, ORANGE_CLR), np.full(N, BLUE_CLR)))  # 0 для первого класса, 1 для второго

    return x, labels

def generate_spiral_data():
    pass

# CIRCLE
# circle_samples, circle_labels = generate_circle_data(N)
# plt.figure(figsize=(12, 8))
# plt.subplot(221)
# plt.scatter(circle_samples[:, 0], circle_samples[:, 1], c=circle_labels)
# plt.title('Circle')
# plt.savefig("circle.png")

# XOR
# xor_samples, xor_labels = generate_xor_data(N)
# plt.scatter(xor_samples[:, 0], xor_samples[:, 1], c=xor_labels)
# plt.title('Exclusive or')
# plt.savefig("xor.png")

# GAUSSIAN
gaussian_samples, gaussian_labels = generate_gaussian_data(N)
plt.scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], c=gaussian_labels)
plt.title('Gaussian')
plt.savefig("gaussian.png")