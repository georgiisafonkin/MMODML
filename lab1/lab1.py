import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import scipy.stats


A = scipy.stats.uniform.rvs(loc=-3, scale=6)
B = scipy.stats.uniform.rvs(loc=-3, scale=6)
C = scipy.stats.uniform.rvs(loc=-3, scale=6)
D = scipy.stats.uniform.rvs(loc=-3, scale=6)
EPSILON_0 = 1
POINTS_NUMBER = 25
POINTS_NUMBER_FOR_FULL = 100


def F1(x : float) -> float:
    return A*np.pow(x, 3) + B*np.pow(x, 2) + C*x + D


def F2(x : float) -> float:
    return x * math.sin(math.pi * x)


uniform_mistakes = np.random.uniform(-EPSILON_0, EPSILON_0, POINTS_NUMBER)
normal_mistakes = np.random.normal(0, EPSILON_0, POINTS_NUMBER)

# Сетка для формирования выборки
grid = np.linspace(-1, 1, POINTS_NUMBER)

# Сетка для построения графиков функций
full_points = np.linspace(-1, 1, POINTS_NUMBER_FOR_FULL)

# Формирование данных 
y = [F1(full_points[i] )for i in range(POINTS_NUMBER_FOR_FULL)]

n_data = list()
u_data = list()


for i in range(len(grid)):
    n_data.append(F1(grid[i]) + normal_mistakes[i])
    u_data.append(F1(grid[i]) + uniform_mistakes[i])

normal_sample = np.array(n_data)
uniform_sample = np.array(u_data)


# Полиномиальная регрессия (метод наименьших квадратов)
coefficients = np.polyfit(grid, n_data, 3)
poly = np.poly1d(coefficients)


plt.plot(full_points, y, linestyle='-', color='b', label=f"Исходная функция")  # Синие точки + линия
plt.scatter(grid, uniform_sample, marker='o', linestyle='-', color='r', label=f"Точки выборки с ошибкой")  # Синие точки + линия
plt.plot(full_points, poly(full_points), color='g', label=f'Восстановленная функциональная зависимость')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Пример работы полиномиальной регрессии")
plt.legend()
plt.grid()
plt.savefig("plot.png")