import numpy as np
import matplotlib.pyplot as plt
import torch

# create tensors with requires_grad = true
x = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(-2.0, requires_grad=True)
w = torch.tensor(-10.0, requires_grad=True)
q = torch.tensor(10.0, requires_grad=True)


def func(x):
    y = (x**2+2*x+5)
    #y = x**4+2*x**3+5
    return y


y = x**4+2*x**3+5

# Compute gradients using backward function for y
y.backward()

# Access the gradients using x.grad
dx = x.grad
print("x.grad :", dx)

# compute gradient with epsilon method


def epsilonSlope(x, eps):
    x1 = x+eps
    x2 = x-eps
    y1 = func(x1)
    y2 = func(x2)
    grad2 = (y1-y2)/(2*eps)
    return grad2


epsilonSlope(x, 1e-1)
epsilonSlope(x, 1e-2)
epsilonSlope(x, 1e-3)
epsilonSlope(x, 1e-4)
epsilonSlope(x, 1e-5)
epsilonSlope(x, 1e-6)
epsilonSlope(x, 1e-7)
epsilonSlope(x, 1e-8)
epsilonSlope(x, 1e-9)

# define function for tangent at x with grad2 as slope


def linApprox(x, a):
    # compute value of start point
    h0 = func(x)
    # compute slope (with help of epsilon circle)
    slope = epsilonSlope(x, 1e-4).item()
    tang = []
    for ai in a:
        # make a list of y values for the tangent
        y = slope*(ai-x)+h0
        tang.append(y.item())
    return tang


PATH = '/var/tmp/osane/code/bachelorarbeit/'


def plot_lin_approx():
    a = np.linspace(-5, 5, 100)
    plt.plot(a, func(a))
    plt.plot(a, linApprox(x, a))
    plt.plot(a, linApprox(v, a))
    plt.legend(['function', 'lin approx at x=2', 'lin approx at v=-2'])
    plt.show()
    plt.savefig(PATH+'lin_approx3.png')


plot_lin_approx()
