import sys
from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np

class LanguageModel:
    """Language model class"""

    def __init__(self, n_gram, is_laplace_smoothing, backoff = None):
        """Language model init"""

        self.smoothing = is_laplace_smoothing
        self.ngrams = n_gram
        self.corpus = None
        self.words = None
        self.freqs = None
        self.num_tokens = None

    def train(self, training_file_path):
        """Trains language model given training file"""

        in_file = open(training_file_path, 'r')
        self.corpus = in_file.read()
        self.words = self.corpus.split()
        self.freqs = Counter(self.words)
        self.freqs['<unk>'] = 0

        for k, v in list(self.freqs.items()):
            if v == 1:
                self.freqs['<unk>'] += 1    # increment the frequency of <unk>
                del self.freqs[k]   # delete all tokens with a frequency of 1

        self.num_tokens = sum(self.freqs.values())

    def unigram_probability(self, word):
        """Computes unigram probability"""

        return self.freqs[word] / self.num_tokens

    def product(self, nums):
        """Computes list product"""

        product = 1
        for num in nums: 
            product *= num
        return product

    def unigram_model(self, sentence):
        """Unigram model"""

        sentence = sentence.split()
        for i in range(len(sentence)):
            if sentence[i] not in self.freqs:
                sentence[i] = '<unk>'
        return self.product(self.unigram_probability(word) for word in sentence)

    def bigram_probability(self, bigram):
        """Computes bigram probability"""

        if self.smoothing:
            return (self.corpus.count('{} {}'.format(bigram[0], bigram[1])) + 1)  / (self.corpus.count(bigram[0]) + len(self.freqs))
        else:
            return self.corpus.count('{} {}'.format(bigram[0], bigram[1]))  / self.corpus.count(bigram[0])

    def bigram_model(self, sentence):
        """Bigram model"""

        bigrams = []
        sentence = sentence.split()
        for i in range(len(sentence)-2):
            bigrams.append((sentence[i], sentence[i+1]))

        return self.product(self.bigram_probability(bigram) for bigram in bigrams)

    """
    def generate1(self num_sentences):
        sentences = []

        if self.ngrams == 1:
            None 
        elif self.ngrams == 2:
            None
    """

    def generate(self, num_sentences):
        """Generates num_sentences number of sentences using the shannon method"""

        sentences = []

        for i in range(num_sentences):
            start = '<s>'
            sentence = ''

            while start != '</s>':
                bigram_freqs = {}

                for i in list(self.freqs.keys()):
                    if i != '<s>':
                        bigram_freqs[(start,i)] = self.bigram_probability((start,i)) # dict of all possible bigrams

                total = sum(bigram_freqs.values())

                for i in bigram_freqs:
                    bigram_freqs[i] /= total # normalize the frequencies

                bigram = random.choices(list(bigram_freqs.keys()), list(bigram_freqs.values()))

                sentence += start + ' '
                start = bigram[0][1]

            sentences.append(sentence + '</s>')

        return sentences

    def score(self, sentence):
        """Score"""

        if self.ngrams == 1:
            return self.unigram_model(sentence)
        elif self.ngrams == 2:
            return self.bigram_model(sentence)
        return None

def main():
    # Enough arguments check
    if len(sys.argv) < 3:
        raise ValueError('Not enough arguments.')

    # Arguments
    training_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    my_test_file_path = sys.argv[3]

    # Unigram probability
    unigram_lm = LanguageModel(1, True)
    unigram_lm.train(training_file_path)
    unigram_probs = []
    test_file = open(test_file_path, 'r')
    test_sentences = test_file.read().split('\n')
    unigram_prob_file = open('hw2-unigram-out.txt', 'w')
    for sentence in test_sentences:
        unigram_prob_file.write(str(unigram_lm.score(sentence)) + '\n')
        unigram_probs.append(unigram_lm.score(sentence))

    # Bigram probability
    bigram_lm = LanguageModel(2, True)
    bigram_lm.train(training_file_path)
    bigram_probs = []
    bigram_prob_file = open('hw2-bigram-out.txt', 'w')
    for sentence in test_sentences:
        bigram_prob_file.write(str(bigram_lm.score(sentence)) + '\n')
        bigram_probs.append(bigram_lm.score(sentence))

    # Histograms
    my_test_file = open(my_test_file_path, 'r')
    my_test_sentences = my_test_file.read().split('\n')
    my_test_unigram_probs = []
    my_test_bigram_probs = []

    for sentence in my_test_sentences:
        my_test_unigram_probs.append(unigram_lm.score(sentence))
        my_test_bigram_probs.append(bigram_lm.score(sentence))

    unigram_min_exponent = np.floor(np.log10(np.abs(min(unigram_probs + my_test_unigram_probs))))
    bigram_min_exponent = np.floor(np.log10(np.abs(min(bigram_probs + my_test_bigram_probs))))

    # Unigram histogram
    plt.hist([unigram_probs, my_test_unigram_probs], bins=np.logspace(np.log10(10**unigram_min_exponent),np.log10(1.0)), label = ["Test Unigram Probabilities", "My Test Unigram Probabilities"], stacked = True)
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.savefig('hw2-unigram-histogram.pdf', bbox_inches='tight')

    # Bigram histogram
    plt.hist([bigram_probs, my_test_bigram_probs], bins=np.logspace(np.log10(10**bigram_min_exponent),np.log10(1.0)), label = ["Test Bigram Probabilities", "My Test Bigram Probabilities"], stacked = True)
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.savefig('hw2-bigram-histogram.pdf', bbox_inches='tight')

    # Generate sentences
    num_sentences = 100

    generated_unigram_sentences = unigram_lm.generate(num_sentences)
    generated_bigram_sentences = bigram_lm.generate(num_sentences)

    generated_unigram_file = open('hw2-unigram-generated.txt', 'w')
    generated_bigram_file = open('hw2-bigram-generated.txt', 'w')

    for sentence in generated_unigram_sentences:
        generated_unigram_file.write(sentence + '\n')

    for sentence in generated_bigram_sentences:
        generated_bigram_file.write(sentence + '\n')

if __name__ == "__main__":
    main()