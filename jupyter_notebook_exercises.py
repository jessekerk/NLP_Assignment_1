import nltk
from nltk.corpus import brown
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import math


# NLTK data
nltk.download("brown")  # Downloads the Brown Corpus files

# Dataset loading
sents = brown.sents(categories="news")

# TODO: Train validation split
train_sents, test_sents = train_test_split(sents, test_size=0.1, train_size=0.9, random_state=int(42), shuffle=True)

#print(f"Training Sentences: {len(train_sents)}")
#print(f"Testing Sentences: {len(test_sents)}")

class TrigramLanguageModel:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.vocab = set()

    def preprocess(self, sentence):
        # Add start/end tokens for trigram context
        return ["<s>"] + [w.lower() for w in sentence] + ["</s>"]

    def train(self, sentences):
        print("Training Trigram Model...")
        for sentence in sentences:
            tokens = self.preprocess(sentence)
            self.vocab.update(tokens)

            # Count Bigrams and Trigrams
            for i in range(2, len(tokens)):
                w_prev2 = tokens[i - 2]
                w_prev1 = tokens[i - 1]
                w_curr = tokens[i]

                self.bigram_counts[(w_prev2, w_prev1)] += 1
                self.trigram_counts[(w_prev2, w_prev1, w_curr)] += 1

        self.vocab_size = len(self.vocab)
        print(f"Training complete. Vocab size: {self.vocab_size}")

    def get_prob(self, w_prev2, w_prev1, w_curr, smoothing_k=1.0):
        """
        Calculates P(w_curr | w_prev2, w_prev1) with Add-k smoothing.
        """
        for sentence in sentences:
            tokens = self.preprocess(sentence)
            self.vocab.update(tokens)
            
            for i in range(2, len(tokens)):
                w_prev2 = tokens[i - 2]
                w_prev1 = tokens[i - 1]
                w_curr = tokens[i]
                
                self.bigram_counts[(w_prev2, w_prev1)] += 1
                self.trigram_counts[(w_prev2, w_prev1, w_curr)] += 1                
        # TODO: Count the trigrams and bigrams

        # TODO: Apply smoothing formula

        return ...