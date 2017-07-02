"""
Utilities for parsing text files (load raw data, create
vocabulary, convert words to ids, generate batches).
"""

import collections
import os

import tensorflow as tf
import numpy as np
import pandas as pd

import preprocessing as pp


# varying length sequences need to be padded when batched together
I_PAD = 0
PAD = '<pad>'

# end of sequence (i.e. end of line in text file)
I_EOS = 1
EOS = '<eos>'

# beginning of sequence
I_BOS = 2
BOS = '<bos>'

# unknown/rare word
I_UNK = 3
UNK = 'UNK'

# maximum words in sequences used during training (cutoff after MAX_LEN)
MAX_LEN = 300

# file with all movie script lines
MOVIES_FILE = 'data/movies/movies.txt'
# file with entire Seinfeld script.
SEINFELD_FILE = 'data/seinfeld/text/seinfeld.txt'


def _read_words(filename):
    """Read text file and split words."""
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', ' ').split()


def _build_vocab(filename, args, word_freq=1):
    """Create vocabulary of all words with unique id.

    Reads main text file and creates vocabulary containing all words. If
    pre-training was done, appends any additional words in that text corpus as
    UNK in the word_to_id dictionary.
    """
    data = _read_words(filename)
    counter = collections.Counter(data)
    # create tuples with (word, word frequency)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # create word_to_id dict, save space for special ids at start (+4)
    word_to_id = {e[0]: (i + 4) for (i, e) in enumerate(count_pairs)
                  if e[1] > word_freq}

    # set special words (begin/end of sequence, pad, unknown/rare words)
    word_to_id[EOS] = I_EOS
    word_to_id[PAD] = I_PAD
    word_to_id[BOS] = I_BOS
    word_to_id[UNK] = I_UNK

    for i, (x, y) in enumerate(count_pairs):
        # set unknown words to the UNK indicator
        if x not in word_to_id:
            word_to_id[x] = I_UNK

    if args.speakers[0] == 'movies' or args.pretrain_data_path == MOVIES_FILE:
        # add words in movies file to word_to_id dict if they do not exist
        # in Seinfeld scripts
        data = _read_words(MOVIES_FILE)
        oov = {word: I_UNK for word in data if word not in word_to_id.keys()}
        word_to_id.update(oov)

    return word_to_id


def _df_to_word_ids(df, word_to_id):
    """Convert words in text file to list of word ids"""
    file_ids = []
    for index, row in df.iterrows():
        # text is first (0) element in after row.tolist()
        words = row.tolist()[0].split()
        word_ids = [word_to_id[word] for word in words]
        # add end of sequence id to each sequence
        word_ids.append(I_EOS)
        # only keep sequences greater than 5 words long
        if len(word_ids) > 5:
            file_ids += word_ids
    return file_ids


def _load_raw_data(args, seed=42):
    """Load raw data from data directory (data_path) into pandas DataFrames,
    split train into train/val, create word_to_id dictionary, convert
    data frame words to ids.
    """
    # create dictionary mapping words to number id
    word_to_id = _build_vocab(SEINFELD_FILE, args)

    train_raw_data, valid_raw_data, test_raw_data = [], [], []
    for speaker in args.speakers:
        train_path = os.path.join(args.data_path, speaker + '_train.txt')
        test_path = os.path.join(args.data_path, speaker + '_test.txt')

        # read text files into pandas DataFrames
        df = pd.read_csv(train_path, header=None)
        # split train into train/val pandas data frames
        df_train, df_val = pp.train_test_split(df, train_pct=0.85, seed=seed)
        df_test = pd.read_csv(test_path)

        # append each character's text to raw data lists
        train_raw_data.append(_df_to_word_ids(df_train, word_to_id))
        valid_raw_data.append(_df_to_word_ids(df_val, word_to_id))
        test_raw_data.append(_df_to_word_ids(df_test, word_to_id))

    return train_raw_data, valid_raw_data, test_raw_data, word_to_id


def _raw_to_sequences(raw_data):
    """Convert text file with word ids to list of sequences."""
    sequences, sequence = [], []
    for i in raw_data:
        sequence.append(i)
        if i == I_EOS:
            sequences.append(sequence)
            sequence = []

    return sequences


def get_data(args, seed=42):
    """Read data from text file and convert words in text files to word ids"""
    # get train/valid/test sets and word_to_id dict
    raw_data = _load_raw_data(args, seed)
    train_raw_data, valid_raw_data, test_raw_data, word_to_id = raw_data
    # create reverse dict for id_to_word
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    id_to_word[I_UNK] = UNK

    train_data, valid_data, test_data = [], [], []
    for i in range(len(args.speakers)):
        # convert raw files into lists of sequences
        train_data.append(_raw_to_sequences(train_raw_data[i]))
        valid_data.append(_raw_to_sequences(valid_raw_data[i]))
        test_data.append(_raw_to_sequences(test_raw_data[i]))

    return train_data, valid_data, test_data, id_to_word


def batch_iterator(data, batch_size):
    """Iterate on the raw data using generator.

    This chunks up data into batches of examples, where each batch has the
    same length. Sequences that are shorter than the max length of the batch
    are padded with zeros.
    """
    n_sequences = len(data) - 1
    epoch_size = n_sequences // batch_size

    idx = np.arange(epoch_size)
    # get random ids for random feeding of batches while training
    np.random.shuffle(idx)

    for i in range(epoch_size):
        ii = idx[i]
        # get sequences for batch starting in location ii
        batch_sequences = data[batch_size*ii:batch_size*(ii+1)]
        # max_len of batch, which is either the longest sequence in the batch
        # or max_len. subtract 2 for EOS marker and y (target) being
        # 1 after each x (input) value
        max_len = min(max([len(s) for s in batch_sequences])-2, MAX_LEN)
        # initialize x, y with 0's as this acts as padding at end of sequence
        x = np.zeros([batch_size, max_len])
        y = np.zeros([batch_size, max_len])

        for j in range(batch_size):
            # create batches with actual word ids for each sequence
            # will be padded with 0's as this is how x, y were initialized
            s = batch_sequences[j]
            l = min(MAX_LEN, len(s))
            x[j][:l-2] = s[:l-2]
            y[j][:l-2] = s[1:l-1]

        yield (x, y)
