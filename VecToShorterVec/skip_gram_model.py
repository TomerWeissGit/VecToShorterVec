import torch.nn as nn
from tensorflow import Tensor


class SkipGramModel(nn.Module):
    def __init__(self, n, dimension_desired):
        """
        The skip gram model used as a base model for Word2Vec and for vec2ShorterVec algorithms.
        :param n: The number of unique words in Word2Vec and n observations in vec2shorterVec
        :param dimension_desired: Desired dimension for reduction
        """
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(n, dimension_desired)
        self.output_layer = nn.Linear(dimension_desired, n)
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
