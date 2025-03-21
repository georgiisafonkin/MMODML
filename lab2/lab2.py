import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from torch import nn, optim
import torch

from sklearn.metrics import confusion_matrix
import seaborn as sns


TRAINING_N = 80
VALIDATION_N = 80
TEST_N = 80

BLUE_CLR = '#1f77b4'
ORANGE_CLR = '#ff7f0e'

def generate_circle_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    noise = 0.2

    center_angles = np.random.uniform(0, 2 * np.pi, N)
    center_radius = np.random.uniform(0, 0.5, N)
    center_x1 = center_radius * np.cos(center_angles) + np.random.normal(0, noise, N)
    center_x2 = center_radius * np.sin(center_angles) + np.random.normal(0, noise, N)
    center_labels = np.full(N, 0)

    boundary_angles = np.random.uniform(0, 2 * np.pi, N)
    boundary_radius = np.random.uniform(1, 1.5, N)
    boundary_x1 = boundary_radius * np.cos(boundary_angles) + np.random.normal(0, noise, N)
    boundary_x2 = boundary_radius * np.sin(boundary_angles) + np.random.normal(0, noise, N)
    boundary_labels = np.full(N, 1)

    y = np.concatenate((center_labels, boundary_labels))
    x = np.vstack((np.column_stack((center_x1, center_x2)), np.column_stack((boundary_x1, boundary_x2))))

    return x, y

def generate_xor_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    x1 = np.random.uniform(-6, 6, N)  # Генерируем случайные точки в квадрате [0, 1] x [0, 1]
    x2 = np.random.uniform(-6, 6, N)
    x = np.column_stack((x1, x2))
    values = np.logical_xor(x[:, 0] > 0, x[:, 1] > 0).astype(int)  # XOR логика
    y = np.where(values > 0, 0, 1)
    return x, y

def generate_gaussian_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    N //= 2
    
    mean1 = [0, 0]
    cov1 = [[0.1, 0], [0, 0.1]]
    
    mean2 = [1, 1]
    cov2 = [[0.1, 0], [0, 0.1]]
    
    class1 = np.random.multivariate_normal(mean1, cov1, N)
    
    class2 = np.random.multivariate_normal(mean2, cov2, N)

    x = np.vstack((class1, class2))

    y = np.hstack((np.full(N, 0), np.full(N, 1)))  # 0 для первого класса, 1 для второго

    return x, y

def generate_spiral_data(N: int) -> Tuple[np.ndarray, np.ndarray]:
    noise = 0.1
    
    theta = np.linspace(-4 * np.pi, 0, N)  # Угол
    r = theta  # Радиус
    
    first_x1 = r * np.cos(theta) + np.random.normal(0, noise, N)
    first_x2 = r * np.sin(theta) + np.random.normal(0, noise, N)

    second_x1 = r * np.cos(theta + np.pi) + np.random.normal(0, noise, N)
    second_x2 = r * np.sin(theta + np.pi) + np.random.normal(0, noise, N)

    x = np.vstack((np.column_stack((first_x1, first_x2)), np.column_stack((second_x1, second_x2))))
    y = np.hstack((np.full(N, ORANGE_CLR), np.full(N, BLUE_CLR)))

    return x, y

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
# gaussian_samples, gaussian_labels = generate_gaussian_data(N)
# plt.scatter(gaussian_samples[:, 0], gaussian_samples[:, 1], c=gaussian_labels)
# plt.title('Gaussian')
# plt.savefig("gaussian.png")

# SPIRAL
# spiral_samples, spiral_labels = generate_spiral_data(N)
# plt.scatter(spiral_samples[:, 0], spiral_samples[:, 1], c=spiral_labels)
# plt.title("Spiral")
# plt.savefig("spiral.png")

class GIPerceptron(): # Geometric intuition algorithm
    def __init__(self):
        super(GIPerceptron, self).__init__()
        self.weights = torch.randn(2) * 0.01
        self.bias = torch.randn(1) * 0.01
        self.learn_rate = 0.1
        self.loss_func = nn.BCELoss()


    def binary_step(self, x):
        return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))
    
    def predict(self, x):
        return self.binary_step(x @ self.weights + self.bias)

    def train(self, x: torch.Tensor, y, epochs=2000):
        for epoch in range(epochs):
            errors = 0
            for i in range(x.shape[0]):
                x_i = x[i]
                y_i = y[i]

                predict_y = self.predict(x_i)

                if predict_y != y_i.item():
                    self.weights += self.learn_rate * (y_i - predict_y) * x_i
                    self.bias += self.learn_rate * (y_i - predict_y)
                    errors += 1

            predictions = self.predict(x)
            loss = self.loss_func(predictions.view(-1, 1), y)
            print(f"Epoch {epoch + 1}, Errors: {errors}, Loss: {loss.item()}")
            if errors == 0:
                break


class Perceptron2(nn.Module):
    def __init__(self):
        super(Perceptron2, self).__init__()
        self.fully_connected = nn.Linear(2, 1) # (x1, x2) --> y
        self.activation_function = nn.Sigmoid()

    def forward(self, x):
        return self.activation_function(self.fully_connected(x))


train_x_samples, train_y_samples = generate_gaussian_data(TRAINING_N)
test_x_samples, test_y_samples = generate_gaussian_data(TEST_N)

train_y_samples = torch.tensor(train_y_samples, dtype=torch.float32).view(-1, 1)
test_y_samples = torch.tensor(test_y_samples, dtype=torch.float32).view(-1, 1)

plt.scatter(x=train_x_samples[:,0], y=train_x_samples[:,1], c=train_y_samples)
plt.title("xor model test")
plt.savefig("xortest.png")

# Преобразуем numpy массивы в тензоры PyTorch
train_x_samples = torch.tensor(train_x_samples, dtype=torch.float32)
test_x_samples = torch.tensor(test_x_samples, dtype=torch.float32)

gi_model = GIPerceptron()
gi_model.train(train_x_samples, train_y_samples)

preds = gi_model.predict(test_x_samples)

np_preds = np.array(preds)
np_y = np.array(test_y_samples)

cm = confusion_matrix(np_preds, np_y)

# Визуализируем матрицу ошибок

plt.figure(figsize=(10, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(np_y), yticklabels=np.unique(np_y))

plt.ylabel('Истинные метки')

plt.xlabel('Предсказанные метки')

plt.title('Матрица ошибок')

plt.show()

# # Создание модели, функции потерь и оптимизатора
# model = Perceptron2()
# criterion = nn.BCELoss()  # Функция ошибки, в данном случаем бинарная кросс-энтропия
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# # Обучение модели
# epochs = 1000
# for epoch in range(epochs):
#     optimizer.zero_grad() # Сбрасываем значения градиента
#     outputs = model(train_x_samples) # Вычисляем выходные значения по выборке на текущей итерации
#     loss = criterion(outputs, train_y_samples) # Вычисляем функцию ошибки
#     loss.backward() #Обратное распространенние ошибки
#     optimizer.step() #Снова вычисляем градиент
    
#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# # Тестирование модели
# with torch.no_grad():
#     predictions = model(test_x_samples)
#     loss = criterion(predictions, test_y_samples)
#     print(f"Test samples loss function: {loss.item():.4f}")
#     # print("Predictions:", predictions.numpy())