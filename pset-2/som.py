import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np


class SOM:
    def __init__(self, N, M):
        self.N, self.M = N, M

        self.G = np.zeros((N, M, 2))
        self.neigh = {}

    def init_G(self, prior_func):
        """
        Initialize weights given prior
        """
        for i in range(self.N):
            for j in range(self.M):
                self.G[i][j] = prior_func(i, j)

    def init_neigh(self, is_neigh):
        """
        Updates neighbors given a function that determines if a node is a neighbor.
        """
        for i in range(self.N):
            for j in range(self.M):
                self.neigh[(i, j)] = set(
                    [
                        (k, l)
                        for k in range(self.N)
                        for l in range(self.M)
                        if is_neigh((i, j), (k, l))
                    ]
                )

    def update(self, x, epoch, lr_func):
        """
        Update all weights based on input
        """

        # Finds closest node to x
        sums = np.sum((self.G - x) ** 2, axis=-1)
        closest = np.unravel_index(np.argmin(sums), sums.shape)

        # Define lr
        lr = lr_func(epoch)

        for i in range(self.N):
            for j in range(self.M):
                delta = x - self.G[i][j]
                self.G[i][j] += lr * delta * self.update_indiv((i, j), closest, epoch)

    def update_indiv(self, cur, cur_close, epoch):
        """
        Returns update to weight of the current node based neighborhood
        """
        if not cur in self.neigh[cur_close]:
            return 0

        # Uses gaussian dropoff
        dist = np.sum((self.G[cur_close] - self.G[cur]) ** 2) ** 0.5
        return np.exp(-(epoch ** 0.7 / 10) * dist ** 2)

    def render(self):
        """
        Draws points and connects them with lines
        """
        global is_neigh_grid

        lines = []
        for i in range(self.N):
            for j in range(self.M):
                for neigh in self.neigh[(i, j)]:
                    if is_neigh_grid((i, j), neigh):
                        lines.append([tuple(self.G[i][j]), tuple(self.G[neigh])])

        _, ax = plt.subplots()
        ax.add_collection(mc.LineCollection(lines, linewidths=3))
        ax.autoscale()
        plt.show()


def is_neigh_grid(a, b):
    """
    Grid-like neighborhood relations
    """
    dr, dc = [0, 1, 0, -1], [1, 0, -1, 0]
    for k in range(4):
        if a[0] + dr[k] == b[0] and a[1] + dc[k] == b[1]:
            return True


def is_neigh_full(i, j):
    """
    Fully connects all nodes
    """
    return True


def get_uniform_prior(n, m):
    """
    Uniform prior
    """

    def f(i, j):
        return np.array([i / (n - 1), j / (m - 1)])

    return f


def get_1d_uniform(n):
    """
    Uniform prior for 1d maps (custom for the PCA dataset)
    """

    def f(i, _):
        return np.array([4 * i / (n - 1) - 4, 0])

    return f


def get_lr_func(a, b, c=50):
    """
    Exponentially decaying learning rate
    """

    def lr_func(epochs):
        scaling = np.exp(-epochs / c)
        return a * scaling + b * (1 - scaling)

    return lr_func
