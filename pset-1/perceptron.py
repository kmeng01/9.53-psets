import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


class Perceptron:
    def __init__(self, d):
        np.random.seed(999)
        self.d = d
        self.w = np.random.rand(self.d)

    def predict(self, x):
        y_hat = np.sign(np.dot(self.w, x))
        return 1 if y_hat == 0 else y_hat

    def get_accuracy(self, x, y):
        y_hat = np.sign(x @ np.expand_dims(self.w, axis=1)).flatten()
        return np.mean(y_hat == y)

    def train(self, x, y, pca, epochs=10, lr=0.1, plot_interval=1, plot_progress=True):
        """
        y should be a vector of {-1, 1}
        """

        np.random.seed(999)
        self.w = np.random.rand(self.d)

        to_plot = []
        tot_epochs = 0

        for e in range(epochs):
            # Train
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.predict(x_i)
                if y_hat_i != y_i:
                    self.w += lr * (-y_hat_i * x_i)

                if i % plot_interval == 0:
                    to_plot.append(self.get_accuracy(x, y))

            e += 1
            if self.get_accuracy(x, y) == 1:
                break

        if plot_progress:
            plt.plot(to_plot)
            plt.xlabel("Training step (per example)")
            plt.ylabel("Accuracy")
            plt.title("Perceptron training accuracy over time")
            plt.show()

        return e

    def get_parallel_w(self, pca):
        theta = 90
        r = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))

        weight_normal = pca.transform(np.expand_dims(self.w, axis=0))
        weight_parallel = (r @ weight_normal.T).squeeze()

        return weight_parallel
