import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import h5py
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

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = training_text[i:i + seq_length]
    seq_out = training_text[i + seq_length]
    dataX.append([chars_as_int[char] for char in seq_in])
    dataY.append(chars_as_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
# filename = R"Model\weights-improvement-20-1.8463.hdf5"
# model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath=R"Model\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)