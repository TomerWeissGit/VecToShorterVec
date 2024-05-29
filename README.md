# VecToShorterVec
A cool implementation based on word2vec algorithm for dimension reduction of tabular data, the algorithm extension has been developed by myself.

If you are not familier with Word2Vec algorithm, specifically with the skip-gram model, I sugguest reading about it first (at least for basic understanding of how it works).

My extension works as follows: 
Assume youv'e got Tabular float/integer data from an High dimension which you wish to visualize/cluster, lets assume this data set is called X and it is from the dimension of nXm.(where n is number of observations and m features). For example , the Mnist training dataset (which is used in the example files) is consisted of n=60,000 and m=24*24=576.
The way that the algorithm works in short is by applying the word2vec algorthm (skip-gram) to tabular data, the only problem needed to be solved is how to create the right explaining observation-target(context) observations pairing. 

We start by one-hot encode each observation by its index.
we later check for each column(feature):
  for each observation find the k(parameter) closest observations - this makes sure each observation in the dataset is paired m times to different(or not different) targer observations(context).
we than enter those exactly like we do in word2vec and get fitted weight matrix(of the size nXN, where N is the number of weights(parameter))- and thats it! we have reduced the dimension from nXm to nXN where N<<m. 
