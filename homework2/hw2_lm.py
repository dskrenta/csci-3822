
class LanguageModel:

    def __init__(self, n_gram, is_laplace_smoothing, backoff = None):

        # initializes your ngram model

        # (Update, 2/7) n_gram is an integer, 1 indicating unigrams, 2 bigrams, etc

        # (Update, 2/7) is_laplace_smoothing is a boolean value. True for yes, False for no

    def train(self, training_file_path):

        # doesn't have a return

    def generate(self, num_sentences):

        # return a list of strings generated using Shannon's method of length num_sentences

        return None

    def score(self, sentence):

        # (Update, 2/7) you may assume that the words in the sentence are deliminated by spaces

        # return a probability for the given sentence

        return None