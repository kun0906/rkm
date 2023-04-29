"""

https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/


"""
import numpy as np
import matplotlib.pyplot as plt

random_state = 42
n = 200
loc = [0, 0, 0]
radius = 1
d = len(loc)
X = np.zeros((n, d))
r = np.random.RandomState(seed=random_state)
dists = []
PI = 3.1415926


def method1(loc = [0, 0], d=2):
    """
    *Method 1. Polar
     This should be very clear to everyone, and is the foundation for many of the other methods.
    Returns
    -------

    """
    if d!=2:
        raise NotImplementedError(d)
    X = np.zeros((n, d))
    dists = []
    for i in range(n):
        theta = 2 * PI * r.random()
        x = np.cos(theta)
        y = np.sin(theta)
        x = [x, y]
        X[i] = x
        dists.append(np.sqrt(sum([(x[j] - loc[j]) ** 2 for j in range(d)])))

    return X, dists

def method2(d=2, loc=[0,0]):
    if d!=2:
        raise NotImplementedError(d)

    X = np.zeros((n, d))
    dists = []
    i = 0
    while i < n:
        """
        Select u,v ~ U(-1,1)
        d2 = u^2+v^2
        If d2 >1 then
            reject and go back to step 1
        Else
            x = (u^2-v^2)/d2
            y = (2*u*v)/d2
        Return (x,y)
        """
        u, v = r.uniform(-1, 1, size=2)
        d2 = u**2 + v**2
        if d2 > 1:
            continue
        else:
            x = (u**2 - v**2)/d2
            y = (2*u*v)/d2
            x = [x, y]
        X[i] = x
        dists.append(np.sqrt(sum([(x[j] - loc[j]) ** 2 for j in range(d)])))
        i += 1

    return X, dists
def bin_():
    d = 10
    n_clusters = 5
    true_centroids = np.zeros((d, d))
    r = np.random.RandomState(seed=random_state)
    for i in range(n_clusters):
        # get 0 or 1 with 0.5 probability for each coordinate
        true_centroids[i] = [0 if v < 0.5 else 1 for v in r.uniform(0, 1, size=d)]
        print(i, true_centroids[i])


def f2():
    d = 2
    n = 1000
    X = np.zeros((n, d))
    for i in range(n):
        u = np.random.normal(0, 1, d + 2)  # an array of (d+2) normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        u = u / norm
        x = u[0:d]  # take the first d coordinates
        X[i] = x


def f3():
    d = len(loc)
    X = np.zeros((n, d))
    dists = []
    for i in range(n):
        u = np.random.normal(0, 1, d + 2)  # an array of (d+2) normally distributed random variables
        norm = np.sum(u ** 2) ** (0.5)
        u = u / norm
        x = (u[0:d] * radius + loc)  # take the first d coordinates
        dists.append(np.sqrt(sum([ (x[j] - loc[j]) ** 2 for j in range(d)])))
        X[i] = x


def f4():
    is_sphere = True
    for i in range(n):
        if is_sphere:
            u = r.normal(0, 1, d)
            v = r.normal(0, 1, d)
            w = r.normal(0, 1, d)
            ds = np.sum(u ** 2) ** (0.5)
            (x, y, z) = (u, v, w) / ds
            x = x
        else:
            u = r.normal(0, 1, d + 2)  # an array of (d+2) normally distributed random variables
            print(i, u)
            norm = np.sum(u ** 2) ** (0.5)
            u = u / norm
            x = (u[0:d] * radius + loc)  # take the first d coordinates

        dists.append(np.sqrt(sum([(x[j] - loc[j]) ** 2 for j in range(d)])))
        X[i] = x


def main():
    # X, dists = method1(d=2, loc=[0,0])
    X, dists = method2(d=2, loc=[0, 0])

    if X.shape[1] == 3:
        ig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones_like(phi))
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.3)

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=100, c='r', zorder=10)
    else:
        plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    print(dists)
    plt.plot(dists)
    plt.title('$\sqrt{(x-\mu)^2}$')
    plt.show()


if __name__ == '__main__':
    main()
