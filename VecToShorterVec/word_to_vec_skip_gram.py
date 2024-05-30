import re
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow import Tensor


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        '''
        The skip gram model used as a base model for word2vec and for vec2ShorterVec algorithms.
        :param vocab_size: The number of unique words in word2vec and n observations in vec2shorterVec
        :param embed_dim: Desired dimension for reduction
        '''
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_word) -> Tensor:
        """
        Given the contex word(input word) the probability of predicting the target words is return (log softmax).
        This is used in the loss function.
        :param input_word: index of context word
        :return: log_softmax Tensor Object for all words given the context word.
        """
        embeds = self.embeddings(input_word)
        output = self.output_layer(embeds)
        log_probs = self.log_softmax(output)
        return log_probs


class Word2VecSkipGram:

    def __init__(self, text, k, N):
        """
        The basic Word2Vec (with skip-gram) algorithm.
        :param text: Raw text for which we want to create Word2Vec Data
        :param k: Number of target words for each word.
        :param N: Dimension (number of weights) wanted.
        """
        self.text = text
        self.k = k
        self.N = N
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_word = {i: word for i, word in enumerate(self.vocab)}
        self.model = SkipGramModel(self.vocab_size, self.N)

    def train(self, epochs=100, learning_rate=0.01):
        training_data = self._generate_training_data()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.NLLLoss()

        for epoch in range(epochs):
            total_loss = 0
            for context_word, target_words in training_data:
                context_idx = torch.tensor([self.word_to_idx[context_word]], dtype=torch.long)
                target_idxs = torch.tensor([self.word_to_idx[target] for target in target_words], dtype=torch.long)

                optimizer.zero_grad()
                log_probs = self.model(context_idx)

                # Accumulate loss for all targets
                loss = 0
                for target_idx in target_idxs:
                    loss += loss_fn(log_probs, target_idx.unsqueeze(0))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Given a word returns the vector of weights for this word.
        :param word: a word which has to exist in the vocabulary (raw text)
        :return: numpy array containing the weights of the word after model was trained.
        """
        word_idx = torch.tensor([self.word_to_idx[word]], dtype=torch.long)
        return self.model.embeddings(word_idx).detach().numpy()

    def _build_vocab(self) -> List[str]:
        """
        Using the raw text, this function creates a list of unique words in text.
        :return: list of unique words
        """
        # Tokenize text and build vocabulary
        words = re.findall(r'\b\w+\b', self.text.lower())
        vocab = sorted(set(words))
        return vocab

    def _generate_training_data(self):
        words = re.findall(r'\b\w+\b', self.text.lower())
        context_target_pairs = []

        for i, word in enumerate(words):
            context = word
            targets = words[max(i - self.k, 0): i] + words[i + 1: min(i + self.k + 1, len(words))]
            context_target_pairs.append((context, targets))

        return context_target_pairs


# Example of usuage
if __name__ == '__main__':
    # Example usage
    text = "The quick brown fox jumps over the lazy dog"
    k = 2
    N = 6

    word2vec = Word2VecSkipGram(text, k, N)
    word2vec.train(epochs=100, learning_rate=0.01)

    # Get vector for a word
    word_vector = word2vec.get_word_vector('fox')
    print(f"Vector for 'fox': {word_vector}")
