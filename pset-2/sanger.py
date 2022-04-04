import numpy as np


class SangerNet:
    def __init__(self) -> None:
        self.w = np.random.uniform(low=-1, high=1, size=(4, 4))
        self.w_norm = None

    def learn(self, x: np.ndarray, lr: float = 1e-4, iters: int = 15000) -> None:
        """
        Applies Sanger's Rule to update the weights of the network.
        """
        for it in range(iters):
            y = self.w @ x
            delta = (y @ x.T) - np.tril(y @ y.T) @ self.w
            delta *= lr
            if it % 1000 == 0:
                print(f"Iteration {it}: delta norm {np.linalg.norm(delta)}")
            self.w += delta

        self.w_norm = self.normalize(self.w)

    def sim(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the similarity between the input and the network.
        """
        return self.w @ x

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes the input to have unit norm.
        """
        return x / np.linalg.norm(x, axis=1)[:, None]
