import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

A = scipy.stats.uniform.rvs(loc=-3, scale=6)
B = scipy.stats.uniform.rvs(loc=-3, scale=6)
C = scipy.stats.uniform.rvs(loc=-3, scale=6)
D = scipy.stats.uniform.rvs(loc=-3, scale=6)
EPSILON_0 = 1
POINTS_NUMBER = 64
POINTS_NUMBER_FOR_FULL = 128
POLY_DEGREE = 5
F1_STR = "A*x^3 + B*x^2 + C*x + D"
F2_STR = "x*sin(2*pi*x)"

def f1(x : float) -> float:
    return A*np.pow(x, 3) + B*np.pow(x, 2) + C*x + D

def f2(x : float) -> float:
    return x * math.sin(2 * math.pi * x)

uniform_mistakes = np.random.uniform(-EPSILON_0, EPSILON_0, POINTS_NUMBER)
normal_mistakes = np.random.normal(0, EPSILON_0, POINTS_NUMBER)

# Сетка для формирования выборки
grid = np.random.uniform(-1, 1, POINTS_NUMBER)

# Сетка для построения графиков функций
full_points = np.linspace(-1, 1, POINTS_NUMBER_FOR_FULL)

# Получение точек исходной функции
y = [f2(full_points[i]) for i in range(POINTS_NUMBER_FOR_FULL)]

n_data = []
u_data = []

# Получение точек выборки для двух распределений ошибки
for i, point in enumerate(grid):
    n_data.append(f2(point) + normal_mistakes[i])
    u_data.append(f2(point) + uniform_mistakes[i])

normal_sample = np.array(n_data)
uniform_sample = np.array(u_data)

# Полиномиальная регрессия (метод наименьших квадратов)
coefficients = np.polyfit(grid, n_data, POLY_DEGREE)
poly = np.poly1d(coefficients)

# Отрисовка
plt.plot(full_points, y, linestyle='-', color='b', label=f"Исходная функция {F2_STR}")
plt.scatter(grid, normal_sample, marker='o', linestyle='-', color='r', label=f"Точки выборки")
plt.plot(full_points, poly(full_points), color='g', label=f"Восстановленный полином {POLY_DEGREE} степени")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Полиномиальная регрессия. Хорошее предсказание.")
plt.legend()
plt.grid()
plt.ylim(-6, 6)
plt.savefig(f"polyregr_xsin2pix_optimal.png")