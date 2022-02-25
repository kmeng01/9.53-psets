import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate


class Perceptron:
    """
    Convention:
    w is a weight matrix of shape 1 x d
    x is a data matrix of shape d x n
    y is a target matrix of shape 1 x n
    """

    def __init__(self, d, is_adaline=False):
        self.d, self.w, self.is_adaline = d, None, is_adaline
        self.init_w()

    def init_w(self):
        np.random.seed(999)
        self.w = np.random.rand(1, self.d)

    def predict_train(self, x):
        if not self.is_adaline:
            y_hat = np.sign(self.w @ x)
            y_hat[y_hat == 0] = 1
            return y_hat
        else:
            return self.w @ x

    def predict_test(self, x):
        """
        Different from training because we need to threshold the
        continuous output for ADALINE.
        """
        y_hat = np.sign(self.w @ x)
        y_hat[y_hat == 0] = 1
        return y_hat

    def get_accuracy(self, x, y):
        return np.mean(self.predict_test(x) == y)

    def get_loss(self, x, y):
        return np.mean((self.predict_train(x) - y) ** 2)

    def train(
        self,
        x,
        y,
        pca,
        batch_size=1,
        epochs=10,
        lr=0.1,
        min_correct=False,
        plot_interval=1,
        plot_progress=True,
        acc_lim=1,
    ):
        """
        y should be a vector of {-1, 1}
        """

        self.init_w()

        plot_acc, plot_loss = [], []
        tot_epochs = epochs

        for e in range(epochs):
            # Train
            updates, updates_cnt = np.zeros(self.w.shape), 0
            for i in range(x.shape[1]):
                x_i, y_i = x[:, i : i + 1], y[:, i : i + 1]
                y_hat_i = self.predict_train(x_i)
                if y_hat_i != y_i:
                    if not self.is_adaline:
                        if not min_correct:
                            cur_lr = lr
                        else:
                            cur_lr = ((self.w @ x_i) / (x_i.T @ x_i)).squeeze()
                        updates += np.abs(cur_lr) * (y_i * x_i.T)
                    else:
                        updates += lr * (-y_hat_i + y_i) * x_i.T
                    updates_cnt += 1

                if updates_cnt == batch_size or i == x.shape[1] - 1:
                    self.w += updates
                    updates, updates_cnt = np.zeros(self.w.shape), 0

                if i % plot_interval == 0:
                    plot_acc.append(self.get_accuracy(x, y))
                    plot_loss.append(self.get_loss(x, y))

            if self.get_accuracy(x, y) >= acc_lim - 1e-5 and tot_epochs == epochs:
                tot_epochs = e

        if plot_progress:
            plt.plot(plot_acc)
            plt.xlabel("Training step (per example)")
            plt.ylabel("Accuracy")
            plt.title("Perceptron training accuracy over time")
            plt.show()

            plt.plot(plot_loss)
            plt.xlabel("Training step (per example)")
            plt.ylabel("MSE")
            plt.title("Perceptron training loss over time")
            plt.show()

        return tot_epochs

    def get_parallel_w(self, pca):
        """
        Returns a vector parallel to the decision boundary
        in PCA space. For visualization purposes.
        """

        theta = 90
        r = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))

        weight_normal = pca.transform(self.w)
        weight_parallel = (r @ weight_normal.T).squeeze()

        return weight_parallel
