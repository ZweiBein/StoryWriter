import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import cleaning as clean


file_path = R'Books\Alice.txt'
training_text = open(file_path).read()
training_text = training_text.lower()
training_text = clean.remove_non_ascii(training_text)

unique_chars = sorted(list(set(training_text)))
chars_as_int = dict((c, i) for i, c in enumerate(unique_chars))

n_chars = len(training_text)
n_vocab = len(chars_as_int)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)