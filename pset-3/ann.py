import numpy as np


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.loss = MSELoss()

    def fit(self, train_x, train_y, iters, lr):
        """
        Runs stochastic gradient descent, where each iteration
        samples a random point from the training dataset.
        """

        for it in range(iters):
            idx = np.random.randint(train_x.shape[0])
            x, y = train_x[idx:idx + 1, :], train_y[idx: idx + 1, :]

            y_pred = self.forward(x)
            loss = self.loss.forward(y_pred, y).sum()
            self.backward()

            for layer in self.layers:
                layer.update(lr)

            if it % (iters // 20) == 0:
                y_pred = self.forward(train_x)
                loss = self.loss.forward(y_pred, train_y).sum()
                acc = (y_pred.argmax(axis=1) == train_y.argmax(axis=1)).mean()
                print(f"Epoch {it}: loss = {loss} | acc = {acc}")

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        """
        Backpropagation algorithm: computes gradients for each layer.
        """

        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


class Dense:
    INIT_FN = {
        "uniform": lambda in_dim, out_dim: np.random.randn(in_dim, out_dim),
        "normal": lambda in_dim, out_dim: np.random.normal(0, 1, (in_dim, out_dim)),
    }

    def __init__(self, in_dim: int, out_dim: int, init="uniform"):
        self.w = Dense.INIT_FN[init](in_dim, out_dim)
        self.b = np.zeros(out_dim)

        self.inp, self.w_grad, self.b_grad = None, None, None

    def forward(self, x):
        self.inp = x
        return self.inp @ self.w + self.b

    def backward(self, prev_grad):
        """
        Stores dL/dw and dL/db for the current layer (used to update weights).
        Returns dL/dx for the previous layer's chain rule computation.
        """

        self.w_grad = self.inp.T @ prev_grad
        self.b_grad = prev_grad.sum(axis=0)

        return prev_grad @ self.w.T

    def update(self, lr):
        """
        Updates weights and biases using the computed gradients.
        """

        self.w -= lr * self.w_grad
        self.b -= lr * self.b_grad


class Sigmoid:
    def __init__(self):
        self.inp, self.grad = None, None

    def forward(self, x):
        self.inp = x
        return 1 / (1 + np.exp(-self.inp))

    def backward(self, prev_grad):
        """
        Returns dL/dx for the previous layer's chain rule computation.
        """

        out = self.forward(self.inp)
        cur_grad = out * (1 - out)
        return cur_grad * prev_grad

    def update(self, *args, **kwargs):
        pass


class MSELoss:
    def __init__(self):
        self.y_pred, self.y_true = None, None
        self.grad = None

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = y_pred, y_true
        self.out = (y_pred - y_true) ** 2
        return self.out

    def backward(self):
        """
        Returns dL/dx.
        """

        self.grad = 2 * (self.y_pred - self.y_true)
        return self.grad
