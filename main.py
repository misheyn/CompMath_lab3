import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt


def f1(x):
    return np.arcsin(1.2 * x + 0.2) - x


def f2(x):
    return np.sqrt(1 - x * x)


# исходная функция 1
def func1(x, y):
    return np.sin(x + y) - 1.2 * x - 0.2


# исходная функция 2
def func2(x, y):
    return x * x + y * y - 1


# производная по X 1
def func11(x, y):
    return np.cos(x + y) - 1.2


# производная по Y 1
def func12(x, y):
    return np.cos(x + y)


# производная по X 2
def func21(x, y):
    return 2 * x


# производная по Y 2
def func22(x, y):
    return 2 * y


def alpha(x, y):
    return 1 / (func12(x, y) * func21(x, y) / func22(x, y) - func11(x, y))


def beta(x, y):
    return func12(x, y) / (func11(x, y) * func22(x, y) - func12(x, y) * func21(x, y))


def gamma(x, y):
    return 1 / (func11(x, y) * func22(x, y) / func21(x, y) - func12(x, y))


def delta(x, y):
    return func11(x, y) / (func21(x, y) * func12(x, y) - func11(x, y) * func22(x, y))


def func(p):
    x, y = p
    return np.sin(x + y) - 1.2 * x - 0.2, x * x + y * y - 1


def iter_func1(x, y):
    return x + alpha(x, y) * func1(x, y) + beta(x, y) * func2(x, y)


def iter_func2(x, y):
    return y + gamma(x, y) * func1(x, y) + delta(x, y) * func2(x, y)


def newton_func1(x, y):
    return x - np.linalg.det(A1(x, y)) / np.linalg.det(jacobi(x, y))


def newton_func2(x, y):
    return y - np.linalg.det(A2(x, y)) / np.linalg.det(jacobi(x, y))


def A1(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func1(x, y)
    matrix[1][0] = func2(x, y)
    matrix[0][1] = func12(x, y)
    matrix[1][1] = func22(x, y)
    return matrix


def A2(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func11(x, y)
    matrix[1][0] = func21(x, y)
    matrix[0][1] = func1(x, y)
    matrix[1][1] = func2(x, y)
    return matrix


def jacobi(x, y):
    matrix = [[0] * 2 for _ in range(2)]
    matrix[0][0] = func11(x, y)
    matrix[1][0] = func21(x, y)
    matrix[0][1] = func12(x, y)
    matrix[1][1] = func22(x, y)
    return matrix


def iteration(x, y, iter_f1, iter_f2, eps):
    it = 0
    while math.sqrt((iter_f1(x, y) - x) ** 2 + (iter_f2(x, y) - y) ** 2) >= eps:
        it += 1
        x = iter_f1(x, y)
        y = iter_f2(x, y)
    return x, y, it


def seidel(x, y, iter_f1, iter_f2, eps):
    it = 0
    while math.sqrt((iter_f1(x, y) - x) ** 2 + (iter_f2(iter_f1(x, y), y) - y) ** 2) >= eps:
        it += 1
        x = iter_f1(x, y)
        y = iter_f2(iter_f1(x, y), y)
    return x, y, it


def newton(x, y, newton_f1, newton_f2, eps):
    it = 0
    while math.sqrt((newton_f1(x, y) - x) ** 2 + (newton_f2(x, y) - y) ** 2) >= eps:
        it += 1
        x = newton_f1(x, y)
        y = newton_f2(x, y)
    return x, y, it


def result_print(res):
    print("x = %.5f y = %.5f" % (res[0], res[1]))
    print("Number of iteration = %d" % res[2])


e = 10e-5
x0 = -0.9
y0 = -0.2

print("Initial approximation: x0 = %.1f y0 = %.1f" % (x0, y0))

print("\nIteration method:")
res1 = iteration(x0, y0, iter_func1, iter_func2, e)
result_print(res1)

print("\nSeidel method:")
res2 = seidel(x0, y0, iter_func1, iter_func2, e)
result_print(res2)

print("\nNewton method:")
res3 = newton(x0, y0, newton_func1, newton_func2, e)
result_print(res3)

print("\nCheck with use SciPy:")
X = opt.fsolve(func, (x0, y0))[0]
print("x = %0.5f y = %0.5f" % (X, f1(X)))

x_gr = np.arange(-0.999, 0, 0.01)
plt.grid(True)
plt.plot(x_gr, f1(x_gr), lw=2, color="green")
plt.plot(x_gr, -f2(x_gr), lw=2, color="red")
plt.scatter(X, f1(X))
plt.show()
