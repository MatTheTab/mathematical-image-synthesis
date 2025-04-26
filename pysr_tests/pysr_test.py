import pysr
import numpy as np

t = np.linspace(0, np.pi, 5000).reshape(-1, 1)  # (5000, 1) shape

X = (4/9) * np.sin(2*t[:,0]) + (1/3) * (np.sin(t[:,0]))**8 * np.cos(3*t[:,0]) + (1/8) * np.sin(2*t[:,0]) * (np.cos(247*t[:,0]))**4
Y = np.sin(t[:,0]) + (1/3) * (np.sin(t[:,0]))**8 * np.sin(3*t[:,0]) + (1/8) * np.sin(2*t[:,0]) * (np.sin(247*t[:,0]))**4

targets = np.stack([X, Y], axis=1)  # (5000, 2)

model = pysr.PySRRegressor(
    model_selection="best",
    niterations=200,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "sin", "cos", "square", "cube",
    ],
    extra_sympy_mappings={"square": lambda x: x**2, "cube": lambda x: x**3},
    loss="loss(x, y) = (x - y)^2",
    populations=10,
    verbosity=1,
    procs=4
)

model.fit(t, targets)
