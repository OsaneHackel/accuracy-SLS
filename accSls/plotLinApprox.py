import matplotlib as plt

#

def linApprox(x, a):
    # compute value of start point
    h0 = func(x)
    # compute slope (with help of epsilon circle)
    slope = epsilonSlope(x, 1e-4).item()
    tang = []
    for ai in a:
        # make a list of y values for the tangent
        y = slope*(ai+x)+h0
        tang.append(y.item())
    return tang

def plot_lin_approx(model, weights, lr):
    a = np.linspace(-5, 5, 100)
    plt.plot(a, func(a))
    plt.plot(a, linApprox(x, a))
    plt.plot(a, linApprox(v, a))
    plt.legend(['function', 'lin approx at x=2', 'lin approx at v=-2'])
    plt.show()
    plt.savefig(PATH+'lin_approx3.png')