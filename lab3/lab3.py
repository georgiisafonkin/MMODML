import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from itertools import product
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

TRAIN_N = 1024
TEST_N = 2048


# --------- Генерация данных ---------
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
    N //= 2
    noise = 0.1
    
    theta = np.linspace(-4 * np.pi, 0, N)  # Угол
    r = theta  # Радиус
    
    first_x1 = r * np.cos(theta) + np.random.normal(0, noise, N)
    first_x2 = r * np.sin(theta) + np.random.normal(0, noise, N)

    second_x1 = r * np.cos(theta + np.pi) + np.random.normal(0, noise, N)
    second_x2 = r * np.sin(theta + np.pi) + np.random.normal(0, noise, N)

    x = np.vstack((np.column_stack((first_x1, first_x2)), np.column_stack((second_x1, second_x2))))
    y = np.hstack((np.full(N, 0), np.full(N, 1)))

    return x, y


# --------- Sigmoid, 2 layers,  ---------
class SigmoidMLP(nn.Module):
    def __init__(self, epochs=100, lr=0.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 5),
            nn.Sigmoid(),
            nn.Linear(5, 2),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(2, 1)
        self.criterion = nn.BCELoss()
        
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

    def train_model(self, x_train, y_train):
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y.float())
                loss.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, x_test, y_test):
        self.eval()
        with torch.no_grad():
            outputs = self(x_test)
            predicted = (outputs >= 0.5).int().squeeze()  # Превращаем в 0 или 1
            y_test_int = y_test.int().squeeze()

            correct = (predicted == y_test_int).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

        print(f"Accuracy: {accuracy * 100:.2f}%")

        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()
        preds_np = predicted.cpu().numpy()

        # Цвет — по настоящим меткам, marker edge — по предсказанию
        plt.figure(figsize=(6, 6))
        for i in range(len(x_np)):
            plt.scatter(
                x_np[i, 0], x_np[i, 1],
                c='blue' if y_np[i] == 1 else 'red',
                marker='o' if preds_np[i] == y_np[i] else 'x',
                edgecolor='black',
                s=100,
                alpha=0.6
            )

        plt.title("Модельные предсказания (цвет — истинная метка, форма — правильность)")
        plt.xlabel("X₁")
        plt.ylabel("X₂")
        plt.grid(True)
        plt.show()

        return accuracy

# --------- Tanh, 3 layers,  ---------
class TanhMLP(nn.Module):
    def __init__(self, epochs=100, lr=0.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 4),
            nn.Tanh(),
            nn.Linear(4,3),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(3, 2)
        self.criterion = nn.CrossEntropyLoss()
        
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)
        return x

    def train_model(self, x_train, y_train):
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, x_test, y_test):
        self.eval()
        with torch.no_grad():
            outputs = self(x_test)
            # Получаем индекс максимального значения, который соответствует классу
            predicted = torch.argmax(outputs, dim=1)  # Для бинарной классификации
            y_test_int = y_test.int().squeeze()

            correct = (predicted == y_test_int).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

        print(f"Accuracy: {accuracy * 100:.2f}%")

        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()
        preds_np = predicted.cpu().numpy()

        # Цвет — по настоящим меткам, marker edge — по предсказанию
        plt.figure(figsize=(6, 6))
        for i in range(len(x_np)):
            plt.scatter(
                x_np[i, 0], x_np[i, 1],
                c='blue' if y_np[i] == 1 else 'red',
                marker='o' if preds_np[i] == y_np[i] else 'x',
                edgecolor='black',
                s=100,
                alpha=0.6
            )

        plt.title("Модельные предсказания (цвет — истинная метка, форма — правильность)")
        plt.xlabel("X₁")
        plt.ylabel("X₂")
        plt.grid(True)
        plt.show()

        return accuracy


# --------- ReLU, 4 layers,  ---------
class ReLUMLP(nn.Module):
    def __init__(self, epochs=100, lr=0.01):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(4, 2)  # Два нейрона на выходе для бинарной классификации
        self.criterion = nn.CrossEntropyLoss()  # Используем CrossEntropyLoss для бинарной классификации
        
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_layer(x)  # Логиты с двумя выходными нейронами
        return x  # Возвращаем логиты (без softmax)

    def train_model(self, x_train, y_train):
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_y)  # CrossEntropyLoss автоматически применяет softmax
                loss.backward()
                self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, x_test, y_test):
        self.eval()
        with torch.no_grad():
            outputs = self(x_test)
            # Получаем индекс максимального значения, который соответствует классу
            predicted = torch.argmax(outputs, dim=1)  # Для бинарной классификации
            y_test_int = y_test.int().squeeze()

            correct = (predicted == y_test_int).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

        print(f"Accuracy: {accuracy * 100:.2f}%")

        x_np = x_test.cpu().numpy()
        y_np = y_test.cpu().numpy()
        preds_np = predicted.cpu().numpy()

        # Цвет — по настоящим меткам, marker edge — по предсказанию
        plt.figure(figsize=(6, 6))
        for i in range(len(x_np)):
            plt.scatter(
                x_np[i, 0], x_np[i, 1],
                c='blue' if y_np[i] == 1 else 'red',
                marker='o' if preds_np[i] == y_np[i] else 'x',
                edgecolor='black',
                s=100,
                alpha=0.6
            )

        plt.title("Модельные предсказания (цвет — истинная метка, форма — правильность)")
        plt.xlabel("X₁")
        plt.ylabel("X₂")
        plt.grid(True)
        plt.show()

        return accuracy


# --------- Кастомная модель ---------
class CustomMLP(nn.Module):
    def __init__(self, input_size, layer_config, output_size, activation_fn):
        super(CustomMLP, self).__init__()
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, layer_config[0]))
        for i in range(1, len(layer_config)):
            self.layers.append(nn.Linear(layer_config[i - 1], layer_config[i]))

        self.output_layer = nn.Linear(layer_config[-1], output_size)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return self.output_layer(x)


# --------- Обучение с кросс-валидацией ---------
def cross_validate(X, Y, layer_configs, activations, epochs=256, k_folds=16):
    best_score = 0.0
    best_params = None

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.long)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for layer_config, (act_name, act_fn) in product(layer_configs, activations.items()):
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            model = CustomMLP(2, layer_config, 2, act_fn)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
            x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]

            for _ in range(epochs):
                model.train()
                optimizer.zero_grad()
                logits = model(x_train)
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                preds = torch.argmax(val_logits, dim=1)
                acc = accuracy_score(y_val.numpy(), preds.numpy())
                fold_scores.append(acc)

        avg_acc = np.mean(fold_scores)
        print(f"Config: {layer_config}, Activation: {act_name}, Accuracy: {avg_acc:.4f}")

        if avg_acc > best_score:
            best_score = avg_acc
            best_params = (layer_config, act_name, act_fn)

    return best_params, best_score


# --------- Визуализация предсказаний ---------
def visualize(model, x_data, y_data):
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        preds = torch.argmax(logits, dim=1).numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x_data[:, 0], x_data[:, 1], c=preds, cmap="coolwarm", alpha=0.6)
    plt.title("Предсказания модели на всей выборке")
    plt.show()


# --------- Запуск ---------
def main():
    
    # model = ReLUMLP(epochs=500)
    # X, Y = generate_spiral_data(TRAIN_N)
    # x_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(Y, dtype=torch.long)
    # X_VAL, Y_VAL = generate_spiral_data(TEST_N)
    # x_val_tensor = torch.tensor(X_VAL, dtype=torch.float32)
    # y_val_tensor = torch.tensor(Y_VAL, dtype=torch.long)
    # model.train_model(x_tensor, y_tensor)
    # model.predict(x_val_tensor, y_val_tensor)
    


    # Данные
    X, Y = generate_xor_data(TRAIN_N)

    # Гиперпараметры
    layer_configs = [
        [3, 3],
        [4, 3, 2],
        [5, 5, 3, 2],
        [5, 4, 5, 4]
    ]
    activations = {
        'sigmoid': nn.Sigmoid(),
        'relu': nn.ReLU(),
        'tanh': nn.Tanh()
    }

    # Поиск лучших параметров
    best_params, best_score = cross_validate(X, Y, layer_configs, activations)

    print("\n🎯 Лучшая конфигурация:")
    print("Слои:", best_params[0])
    print("Активация:", best_params[1])
    print("Точность:", best_score)

    # Обучим модель с лучшими параметрами на всех данных
    final_model = CustomMLP(2, best_params[0], 2, best_params[2])
    optimizer = optim.Adam(final_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.long)

    for _ in range(500):
        final_model.train()
        optimizer.zero_grad()
        logits = final_model(x_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    # Визуализация
    visualize(final_model, X, Y)


if __name__ == "__main__":
    main()
