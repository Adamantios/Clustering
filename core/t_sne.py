from itertools import combinations
from random import shuffle

import numpy as np
from sklearn import manifold
from sklearn.datasets import make_classification

from helpers import Plotter


class TSNE(object):
    def __init__(self, n_components: int = 2, perplexity: int = 5, sigma: float = 1, n_iter: int = 500,
                 learning_rate: int = 200, step_size: float = 0.1, mass: float = 0.9, show_progress: bool = True,
                 error_threshold: float = 1e-7, minibatch_size: int = 200, random_state=0):
        np.random.seed(random_state)
        self.minibatch_size = minibatch_size
        self.error_threshold = error_threshold
        self.show_progress = show_progress
        self.n_samples: int = 0
        self.n_components = max(n_components, 2)
        self.mass = self._check_mass(mass)
        self.step_size = step_size
        self.learning_rate = max(learning_rate, 100)
        self.n_iter = max(n_iter, 1)
        self.sigma = sigma
        self.perplexity: int = perplexity

    @staticmethod
    def _check_mass(mass):
        if 0.1 <= mass <= 0.99:
            return mass
        else:
            raise ValueError('Mass should be between 0.1 and 0.99. Got {} instead'.format(mass))

    def _k_neighbors_dists(self, x: np.ndarray, index: int) -> np.ndarray:
        """
        Calculates and returns the k neighbors distances of x[index].

        :param x: the samples.
        :param index: the neighbor seeking sample's index.
        :return: The neighbors and their distances.
        """
        # Get the sample which seeks for neighbors.
        neighbor_seeker = x[index]
        # Initialise a numpy array for the neighbor distances with zeros, not empty because we need to sort later.
        distances = np.zeros(self.n_samples)

        for i in range(self.n_samples):
            # If not the neighbor_seeker's index.
            if i != index:
                # Calculate and store distance between neighbor_seeker and current sample.
                distance = np.linalg.norm(neighbor_seeker - x[i])
                distances[i] = distance

        # Sort the distances array and return the needed number of bigger distances,
        # except for the first one, which is zero (the distance with itself).
        distances.sort()
        return np.delete(distances[:self.perplexity], 0)

    def _pij(self, x: np.ndarray, i: int, j: int) -> float:
        """
        Calculates the condition probability of j to be chosen as neighbor of i in the original space.

        :param x: the samples in the original space.
        :param i: the probability's sample index.
        :param j: the probability condition's sample index.
        :return: the condition probability.
        """
        # Probability is zero if i is j.
        if i == j:
            return 0

        nominator = np.exp(-np.linalg.norm(x[i] - x[j]) ** 2 / (2 * self.sigma ** 2))
        denominator = 0
        for distance in self._k_neighbors_dists(x, i):
            denominator += np.exp(distance ** 2 / (2 * self.sigma ** 2))

        return nominator / denominator

    def _p(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the initial space's pairwise probabilities matrix.

        :param x: the samples in the original space.
        :return: the p matrix.
        """
        p = np.zeros((self.n_samples, self.n_samples))
        for i, j in combinations(range(self.n_samples), 2):
            if i != j:
                pij = self._pij(x, i, j)
                pji = self._pij(x, j, i)
                # Ensure that all the pairwise probabilities sum to 1.
                pair_value = (pij + pji) / (2 * self.n_samples)
                p[i, j] = pair_value
                p[j, i] = pair_value
        return p

    def _qij(self, y: np.ndarray, i: int, j: int) -> float:
        """
        Calculates the condition probability of j to be chosen as neighbor of i in the final space.

        :param y: the samples in the final space.
        :param i: the probability's sample index.
        :param j: the probability condition's sample index.
        :return: the condition probability.
        """
        # Probability is zero if i is j.
        if i == j:
            return 0

        nominator = (1 + np.linalg.norm(y[i] - y[j]) ** 2) ** -1
        denominator = 0
        for distance in self._k_neighbors_dists(y, i):
            denominator += (1 + distance ** 2) ** -1

        return nominator / denominator

    def _q(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates the final space's pairwise probabilities matrix.

        :param y: the samples in the final space.
        :return: the q matrix.
        """
        q = np.zeros((self.n_samples, self.n_samples))
        for i, j in combinations(range(self.n_samples), 2):
            if i != j:
                q[i, j] = self._qij(y, i, j)
                q[j, i] = self._qij(y, j, i)
        return q

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the cost between p and q matrices, using the Kullback Leibler divergence.

        :param p: the original space's pairwise probabilities matrix.
        :param q: the final space's pairwise probabilities matrix.
        :return: the cost.
        """
        cost = 0
        for i, j in combinations(range(self.n_samples), 2):
            if i < j:
                cost += p[i, j] * np.log(p[i, j] / q[i, j])

        return cost

    # def get_minibatches(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     Breaks the passed array to minibatches.
    #
    #     :param x: the array to be minibatched.
    #     :return: array of minibatch arrays.
    #     """
    #     # Init a list for the minibatches.
    #     minibatches = []
    #     # Shuffle the passed array.
    #     shuffle(x)
    #
    #     for i in range(0, x.shape[0], self.minibatch_size):
    #         # Get a mini sample of the initial array, of size self.minibatch_size.
    #         x_mini = x[i:i + self.minibatch_size]
    #         # Add the mini sample to the list
    #         minibatches.append(x_mini)
    #
    #     # Return the minibatches as a numpy array.
    #     return np.asarray(minibatches)

    @staticmethod
    def _gradient(y: np.ndarray, p: np.ndarray, q: np.ndarray, i: int, j: int) -> float:
        """
        Calculates the derivative of the kl_divergence with respect to yi.

        :param y: the samples in the final space.
        :param p: the original space's pairwise probabilities matrix.
        :param q: the final space's pairwise probabilities matrix.
        :param i: the i index.
        :param j: the j index.
        :return: the derivative.
        """
        return (p[i, j] - q[i, j]) * (y[i] - y[j]) * (1 + np.linalg.norm(y[i] - y[j] ** 2)) ** -1

    # def _sgd(self, y: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    #     """
    #     Implements stochastic gradient descent with momentum, specialised for t-SNE.
    #
    #     :param y: the samples in the final space.
    #     :param p: the original space's pairwise probabilities matrix.
    #     :param q: the final space's pairwise probabilities matrix.
    #     :return: the updated y.
    #     """
    #     velocity = np.zeros((self.minibatch_size, self.n_components, round(self.n_samples / self.minibatch_size)))
    #     for iteration in range(self.n_iter):
    #         minibatches = self.get_minibatches(y)
    #         rand_idx = np.random.randint(0, len(minibatches))
    #         y_mini = minibatches[rand_idx]
    #
    #         for i in range(y_mini.shape[0]):
    #             gradient = 0
    #             for j in range(y_mini.shape[0]):
    #                 if i == j: pass
    #                 gradient += self._gradient(y_mini, p, q, i, j)
    #
    #             gradient *= 4
    #
    #             for coordinate in range(len(gradient)):
    #                 y[i] -= self.learning_rate * gradient + self.mass
    #
    #         q = self._q(y)
    #
    #         if iteration % 100 == 0 and self.show_progress:
    #             error = self._kl_divergence(p, q)
    #             print(error)
    #             # if error <= self.error_threshold: break
    #
    #     return y

    def _gradient_descent(self, y: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Implements gradient descent, specialised for t-SNE.

        :param y: the samples in the final space.
        :param p: the original space's pairwise probabilities matrix.
        :param q: the final space's pairwise probabilities matrix.
        :return: the updated y.
        """
        history = np.zeros((p.shape[0], 2, y.shape[1]))
        for iteration in range(self.n_iter):
            for i in range(y.shape[0]):
                sum_value = 0
                for j in range(y.shape[0]):
                    sum_value += self._gradient(y, p, q, i, j)
                y[i] -= 4 * self.learning_rate * sum_value + self.mass * (history[i, 1] - history[i, 0])
                history[i, 0] = history[i, 1]
                history[i, 1] = y[i]
            if iteration % 100 == 0:
                q = self._q(y)
                print(self._kl_divergence(p, q))

        return y

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms the given samples to a new dimensional space.

        :param x: the samples to be transformed.
        :return: the samples transformed.
        """
        # Store the number of samples.
        self.n_samples = x.shape[0]
        # Calculate p.
        p = self._p(x)
        # Randomly initialize the new space y.
        y = np.random.rand(self.n_samples, self.n_components)
        # Calculate q.
        q = self._q(y)

        return self._gradient_descent(y, p, q)


def main():
    # Make a classification for testing.
    x, y = make_classification(n_samples=10, n_features=200, random_state=0)
    plotter = Plotter()

    tsne = TSNE()
    x = tsne.fit_transform(x)
    plotter.scatter(x[:, :2], y)

    tsne = manifold.TSNE()
    x = tsne.fit_transform(x)
    plotter.scatter(x[:, :2], y)


if __name__ == '__main__':
    main()
