import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation
import requests

# downloading dataset
FILEPATHS = np.array([["http://www.gutenberg.org/cache/epub/11/pg11.txt", "https://www.gutenberg.org/files/68161/68161-0.txt",
                       "https://www.gutenberg.org/files/1342/1342-0.txt", "https://www.gutenberg.org/files/1661/1661-0.txt",
                       "https://www.gutenberg.org/files/64317/64317-0.txt", "https://www.gutenberg.org/files/345/345-0.txt",
                       "https://www.gutenberg.org/files/98/98-0.txt", "https://www.gutenberg.org/files/174/174-0.txt",
                       "https://www.gutenberg.org/files/1400/1400-0.txt", "https://www.gutenberg.org/files/76/76-0.txt"],
                      ["data/wonderland.txt", "data/trouble.txt", "data/prejudice.txt", "data/sherlock.txt", "data/gatsby.txt",
                       "data/dracula.txt", "data/cities.txt", "data/dorian.txt", "data/expectations.txt", "data/hucklberry.txt"]])

with open('data/gutenberg.txt', 'w') as outFile:
    for i in range(len(FILEPATHS)):
        content = requests.get(FILEPATHS[0][i]).text
        open(FILEPATHS[1][i], "w", encoding="utf-8").write(content)
        FILE_PATH = FILEPATHS[1][i]
        outFile.write(open(FILE_PATH).read())
        outFile.write("\n")

# constants
sequence_length = 100  # length of sequence we will take from text
BATCH_SIZE = 128
EPOCHS = 30  # training loops
# cleaning the dataset
# start by making the dataset file path
FILE_PATH = "data/gutenberg.txt"
BASENAME = os.path.basename(FILE_PATH)
# read the data
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps, comment this code if you want uppercase characters as well
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))

# test by printing some stats
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# making dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# making dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}
# save these dictionaries for later generation in seperate file
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))
# converting all text into integers for making dataset
encoded_text = np.array([char2int[c] for c in text])

# construct tf.data.Dataset object that contains encoded text
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
# Test by printing first 5 characters and corresponding integer value
for char in char_dataset.take(8):
    print(char.numpy(), int2char[char.numpy()])

# construct sequences by batching (combining consecutive elements of dataset into batches)
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)
# test by printing sequences
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))


def split_sample(sample):
    # ds encodes sequences in parts as integers
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample) - 1) // 2):
        input_ = sample[i: i + sequence_length]
        target = sample[i + sequence_length]
        # extend the dataset with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds


# prepare inputs and targets
dataset = sequences.flat_map(split_sample)


def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)


dataset = dataset.map(one_hot_samples)

# test by printing first 2 samples
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

# repeat, shuffle and batch the dataset to remove too small of samples
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

# building model with two LSTM layers with 128 units
# output layer consists of 39 units, one for each unique char
model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    # prevents over-fitting by dropping some neurons in hidden/visible layers
    Dropout(0.3),  # 3/10 of input units dropped
    LSTM(256),  # number of units can be arbitrary but make the look back period longer
    # makes each hidden neuron receive input from all previous layer's neurons
    Dense(n_unique_chars, activation="softmax"),
])
# define the model path
model_weights_path = f"results/{BASENAME}-{sequence_length}.h5"
# prints a summary of the model shape and components
model.summary()
# compile the model using adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# make results folder if it does not exist yet to save model to
if not os.path.isdir("results"):
    os.mkdir("results")
# train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
# save the model to results folder
model.save(model_weights_path)
