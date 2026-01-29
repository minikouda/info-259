import itertools
from collections import Counter
import os
import numpy as np
import sys
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from typing import Optional

nltk.download("punkt_tab")
class FileDataLoader():
    def __init__(self, filepath, negative_sample_alpha=0.75, min_threshold=5):
        self.negative_sample_alpha = negative_sample_alpha
        self.min_threshold = min_threshold

        self.tokenized_documents = self.load_data(filepath)
        self.word_freqs = self.get_word_freqs(self.tokenized_documents)
        self.V = len(self.word_freqs)

        # replace words that appear fewer than min_threshold times with an [UNK] token
        for word, freq in list(self.word_freqs.items()):
            if freq < min_threshold:
                self.word_freqs["[UNK]"] += freq
                del self.word_freqs[word]

        self.idx2vocab = list(self.word_freqs.keys())
        self.vocab2idx = {word: index for index, word in enumerate(self.idx2vocab)}

        # set up a random number generator we can use for sampling
        self.rng = np.random.default_rng(159259)
        self.sample_weights = self.negative_sample_weights(alpha=negative_sample_alpha)

        ...

    def tokenize_and_lowercase(self, doc):
        """Tokenize a doc and lowercase all the words."""
        return [word.lower() for word in word_tokenize(doc)]

    def get_word_freqs(self, tokenized_documents):
        """Return a dictionary mapping each word to its frequency."""
        return Counter(itertools.chain.from_iterable(tokenized_documents))

    def load_data(self, filepath):
        return [self.tokenize_and_lowercase(doc) for doc in tqdm(open(corpus_path, "r").readlines())]

    def negative_sample_weights(self, alpha) -> Optional[np.ndarray] :
        """Calculate the weighted probabilities of each word.

        Return a (v,)-shaped numpy array, where v is the size of the vocabulary.
        """
        # TODO: implement this function
        freqs_arr = np.array(list(self.word_freqs.values()), dtype=np.float64)
        freqs_power_arr = freqs_arr ** alpha
        total_weight = np.sum(freqs_power_arr)
        self.adjust_freqs_arr = freqs_power_arr / total_weight
        return self.adjust_freqs_arr

    def negative_sample(self, target_word_idx, num_samples):
        """Sample num_samples noise words from the lexicon that is not the target word.
    
        The sample probabilities should be proportional to their weighted unigram probability if the target word probability is set to 0.

        Return a (num_samples,)-shaped numpy array of sampled indices.
        """
        # TODO: implement this function
        ...

    def sample_contexts(self, window_size, sample_k):
        for doc in self.tokenized_documents:
            if len(doc) < (2 * window_size) + 1:
                # the doc is too short for our desired window size; we skip it
                continue
            for word_idx in range(window_size, len(doc) - window_size):
                target_word_idx = self.vocab2idx[doc[word_idx]] if doc[word_idx] in self.vocab2idx else self.vocab2idx["[UNK]"]
                # sample positive words from the window
                positive_word_idxs = np.array([
                    self.vocab2idx[word] if word in self.vocab2idx else self.vocab2idx["[UNK]"] for word in doc[word_idx - window_size:word_idx] + doc[word_idx + 1:word_idx + 1 + window_size]
                    
                ])
                # sample len(positive_word_idxs) * sample_k number of negative words
                negative_word_idxs = self.negative_sample(target_word_idx, sample_k * len(positive_word_idxs))
                yield (target_word_idx, positive_word_idxs, negative_word_idxs)



print(os.getcwd())
os.chdir('./self/HW/hw1_embeddings')
corpus_path = "./en_wiki_sample.txt"
dataloader = FileDataLoader(corpus_path)

np.newaxis

# print(dataloader.idx2vocab[:10])
# alpha=.75
# dataloader.negative_sample_weights(alpha=alpha)
