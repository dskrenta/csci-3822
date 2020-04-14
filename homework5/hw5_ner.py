import sys

from collections import Counter


"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

def generate_tuples_from_file(file_path):
  """
  Implemented for you. 

  counts on file being formatted like:
  1 Comparison  O
  2 with  O
  3 alkaline  B
  4 phosphatases  I
  5 and O
  6 5 B
  7 - I
  8 nucleotidase  I
  9 . O

  1 Pharmacologic O
  2 aspects O
  3 of  O
  4 neonatal  O
  5 hyperbilirubinemia  O
  6 . O

  params:
    file_path - string location of the data
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
  current = []
  f = open(file_path, "r", encoding="utf8")
  examples = []
  for line in f:
    if len(line.strip()) == 0 and len(current) > 0:
      examples.append(current)
      current = []
    else:
      pieces = line.strip().split()
      current.append(tuple(pieces[1:]))
  if len(current) > 0:
    examples.append(current)
  f.close()
  return examples

def get_words_from_tuples(examples):
  """
  You may find this useful for testing on your development data.

  params:
    examples - a list of tuples in the format [[(token, label), (token, label)...], ....]
  return:
    a list of lists of tokens
  """
  return [[t[0] for t in example] for example in examples]


def decode(data, probability_table, pointer_table):
  """
  TODO: implement
  params: 
    data - a list of tokens
    probability_table - a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
    pointer_table - a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
  pass


def precision(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of precision at the entity level
  """
  pass


def recall(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of recall at the entity level
  """
  pass

def f1(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of f1 at the entity level
  """
  pass

def pretty_print_table(data, list_of_dicts):
  """
  Pretty-prints probability and backpointer lists of dicts as nice tables.
  Truncates column header words after 10 characters.
  params:
    data - list of words to serve as column headers
    list_of_dicts - list of dicts with len(data) dicts and the same set of
      keys inside each dict
  return: None
  """
  # ensure that each dict has the same set of keys
  keys = None
  for d in list_of_dicts:
    if keys is None:
      keys = d.keys()
    else:
      if d.keys() != keys:
        print("Error! not all dicts have the same keys!")
        return
  header = "\t" + "\t".join(['{:11.10s}']*len(data))
  header = header.format(*data)
  rows = []
  for k in keys:
    r = k + "\t"
    for d in list_of_dicts:
      if type(d[k]) is float:
        r += '{:.9f}'.format(d[k]) + "\t"
      else:
        r += '{:10.9s}'.format(str(d[k])) + "\t"
    rows.append(r)
  print(header)
  for row in rows:
    print(row)

"""
Implement any other non-required functions here
"""

"""
Implement the following class
"""
class NamedEntityRecognitionHMM:
  
  def __init__(self):
    # TODO: implment as needed
    pass

  def train(self, examples):
    """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    return: None
    """
    pass

  def generate_probabilities(self, data):
    """
    params: data - a list of tokens
    return: two lists of dictionaries --
      - first a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
      - second a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
    """
    pass

  def __str__(self):
    return "HMM"

"""
Implement the following class
"""
class NamedEntityRecognitionMEMM:
  def __init__(self):
    # implement as needed
    pass

  def train(self, examples):
    """
    Trains this model based on the given input data
    params: examples - a list of lists of (token, label) tuples
    return: None
    """
    pass

  def featurize(self):
    """
    CHOOSE YOUR OWN PARAMS FOR THIS FUNCTION
    CHOOSE YOUR OWN RETURN VALUE FOR THIS FUNCTION
    """
    pass

  def generate_probabilities(self, data):
    pass

  def __str__(self):
    return "MEMM"


if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage:", "python hw5_ner.py training-file.txt testing-file.txt")
    sys.exit(1)

  training = sys.argv[1]
  testing = sys.argv[2]
  training_examples = generate_tuples_from_file(training)
  testing_examples = generate_tuples_from_file(testing)

  # instantiate each class, train it on the training data, and 
  # evaluate it on the testing data

  
  

