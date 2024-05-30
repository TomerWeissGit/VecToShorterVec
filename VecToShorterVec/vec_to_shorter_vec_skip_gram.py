import random

import numpy as np
import pandas as pd

from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

from word_to_vec_skip_gram import SkipGramModel


class Vec2ShorterVecSkipGram:
    def __init__(self, df: pd.DataFrame, k: int, dim_desired: int):
        """
        The Vec2ShorterVecSkipGram class is ment to reduce data dimension using the skip-gram algorithm.

        :param df: raw pd.DataFrame for designated for dimension reduction.
        :param k: number of desired "targets" for each observation (for each variable)
        :param dim_desired: The desired dimension to reduce to.
        """
        self.data = df
        self.k = k
        self.dim_desired = dim_desired
        self.n_obs = df.shape[0]
        self.obs_to_idx = {obs: i for i, obs in enumerate(self.data.index.values)}
        self.idx_to_obs = {i: obs for i, obs in enumerate(self.data.index.values)}
        self.model = SkipGramModel(self.n_obs, self.dim_desired)

    def train(self, epochs: int = 100, learning_rate: float = 0.01) -> None:
        """
        Given number of epochs and learning rate will train a neural network for dimension reduction.
        :param epochs: number of epochs .
        :param learning_rate: desired Learning rate.
        """

        training_data = self._find_closest_observations()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.NLLLoss()
        for epoch in range(epochs):
            total_loss = 0
            for context_obs, target_obs in training_data:
                context_idx = torch.tensor([context_obs], dtype=torch.long)
                targets_idx = torch.tensor(target_obs, dtype=torch.long)
                optimizer.zero_grad()
                log_probs = self.model(context_idx)

                # Accumulate loss for all targets
                loss = 0
                loss_fn = nn.NLLLoss()
                for target_idx in targets_idx:
                    loss += loss_fn(log_probs, target=target_idx.unsqueeze(0))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def get_obs_vector(self, obs):
        """
        Function used to get a single observation vector
        :param obs: index of desired observation.
        :return: numpy array containing the weights of the desired observation.
        """
        obs_idx = torch.tensor([self.obs_to_idx[obs]], dtype=torch.long)
        return self.model.embeddings(obs_idx).detach().numpy()

    def _find_closest_observations(self, randomize: bool = True) -> List:
        """
        Using the raw data, this function finds the K closest observation for each observation in the data set for each
        feature-this will be the equivalent to context-target words in Word2Vec.
        :param randomize: This flag is used to determent if the function uses randomization(adding a random number
         between 0 and 1 to each observation)
         in case that the data is integer, so that more variance will occur when selecting neighbour

        :return: list of tuples each containing the index of the context word and the list of the corresponding targets
        indexes
        """
        df = self.data.copy()
        if randomize:
            random_numbers = np.random.rand(*df.shape)
            df = df + random_numbers

        def _get_tuple_of_neighbour(col, i):

            observation = df.iloc[i]
            distances = np.abs(df[col].values - observation[col])  # Calculate absolute differences
            sorted_indices = np.argsort(distances)  # Get indices of sorted distances
            closest_indices = sorted_indices[1:self.k + 1]  # Select k closest (excluding the observation itself)

            return i, list(closest_indices)

        list_of_neighbours = [_get_tuple_of_neighbour(col, i) for col in df.columns for i in range(len(df))]
        random.shuffle(list_of_neighbours)

        return list_of_neighbours


if __name__ == "__main__":
    from keras.datasets import mnist
    from matplotlib import pyplot as plt

    # Mnist toy problem - can it cluster?
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1)
    mnist_data = pd.DataFrame(train_X)
    mnist_data = mnist_data.loc[:, mnist_data.sum() != 0]
    # Running on 100 samples
    np.random.seed(42)
    k_example = 2
    desired_dimension = 3
    mnist_sampled = mnist_data.loc[(train_y == 7) | (train_y == 0)].sample(100)
    obs2vec = Vec2ShorterVecSkipGram(mnist_sampled.loc[:, mnist_sampled.var() > mnist_sampled.var().quantile(0.4)],
                                     k_example, desired_dimension)
    obs2vec.train(epochs=10, learning_rate=0.02)

    res = pd.concat([pd.DataFrame(obs2vec.get_obs_vector(idx), index=[idx]) for idx in mnist_sampled.index.values])
    plt.scatter(x=res.iloc[:, 0], y=res.iloc[:, 1], c=pd.Series(train_y).iloc[res.index])
    plt.show()
