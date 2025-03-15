import numpy as np
import matplotlib.pyplot as plt

N = 100
BLUE_CLR = '#1f77b4'
ORANGE_CLR = '#ff7f0e'

def generate_circle_data(N):
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

def generate_xor_data():
    pass

def generate_gaussian_data():
    pass

def generate_spiral_data():
    pass


circle_samples, circle_labels = generate_circle_data(N)

plt.figure(figsize=(12, 8))


plt.subplot(221)

plt.scatter(circle_samples[:, 0], circle_samples[:, 1], c=circle_labels)

plt.title('Circle')

plt.savefig("circle.png")