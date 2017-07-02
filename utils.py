"""
Miscellaneous functions used in training/testing model.
"""

import copy
import random

import tensorflow as tf
import numpy as np


class TensorBoardSummaries(object):
    def __init__(self):
        """Create variables for summaries in TensorBoard. Use these variable
        # across models so we can view summaries on same graphs."""
        self.ppl_summary = tf.Variable(0.0, name='ppl_summary')
        # self.lr_summary = tf.Variable(0.0, name='lr_summary')

    def create_ops(self):
        """Create operations for TensorBoard summaries"""
        self.ppl_op = tf.summary.scalar('ppl', self.ppl_summary)
        # self.lr_op = tf.summary.scalar('lr', self.lr_summary)
        self.summary_op = tf.summary.merge_all()


def set_config(args, id_to_word):
    """Set configuration for train/valid/test."""
    # config is dict containing hyperparameters used in training/testing
    config = {
        'init_scale': args.init_scale,
        'lr': args.lr,
        'max_grad_norm': args.max_grad_norm,
        'num_layers': args.num_layers,
        'num_steps': args.num_steps,
        'embed_size': args.embed_size,
        'max_epoch': args.max_epoch,
        'keep_prob': args.keep_prob,
        'batch_size': args.batch_size,
        'vocab_size': len(id_to_word) + 1
    }

    # set test parameters
    test_config = copy.deepcopy(config)
    # iterate through one word at a time during testing
    test_config['batch_size'] = 1
    test_config['num_steps'] = 1

    return config, test_config


def set_new_args(train_args, args):
    """Set new arguments when only testing/generating from saved model."""
    new_args = copy.deepcopy(train_args)
    new_args.data_path = args.data_path
    new_args.load_path = args.load_path
    new_args.speakers = args.speakers
    return new_args


def glove_to_dict(glove_path):
    """Convert GloVe word vector text file to dict."""
    with open(glove_path, 'r') as f:
        glove_dict = {}
        for line in f:
            split_line = line.split()
            # word is first entry in each line
            word = split_line[0]
            # get word embedding parameters
            embedding = [float(val) for val in split_line[1:]]
            glove_dict[word] = embedding

    return glove_dict


def init_embedding(id_to_word, dim, init_scale, embed_path='', load_path=''):
    """Initialize embeddings with either random uniform or GloVe vectors."""
    # load GloVe vectors if embed_path is specified, not restoring model
    if embed_path != '' and load_path == '' and dim != 0:
        # create dict with GloVe vectors
        glove_dict = glove_to_dict(embed_path)
        # get dim of glove vector with random word ('you')
        glove_dim = len(glove_dict['you'])
        assert glove_dim == dim, 'glove dim {0} must equal arg dim {1}'.format(
            glove_dim, dim)
    n_vocab = len(id_to_word)
    # embedding has shape [vocab size, dims of GloVe vector]. initialize
    # using scale defined in arguments
    embedding = np.random.uniform(-init_scale, init_scale, size=(n_vocab, dim))

    if embed_path != '' and load_path == '' and dim != 0:
        for i in range(n_vocab):
            word = id_to_word[i]
            # if vocab word in GloVe dict, update embedding for this word
            if word in glove_dict.keys():
                embedding[i] = glove_dict[word]

    return embedding


def create_feed_dict(model, args, x, y, state):
    """"Create dict to feed input, targets, and rnn into TF session"""
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y

    if args.rnn_type == 'rnn' or args.rnn_type == 'gru':
        # basic rnn and gru contain single state variable
        for i, h in enumerate(model.initial_state):
            feed_dict[h] = state[i]
    elif args.rnn_type == 'lstm':
        # lstm contains two state variables
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

    return feed_dict


def get_random_hparams(hparams_dict, rand_hparams=False):
    """Get random set of hyperparameters for training.

    If hparam_type is 'cat', select a random item from list provided, otherwise
    select random int/float in the range provided.
    """
    hparams = {}
    if rand_hparams:
        # select random hyperparameters
        for key, value in hparams_dict.items():
            hparam_type = value[-1]
            if hparam_type == 'cat':
                # categorical parameter so take random item
                hparams[key] = random.choice(value[1])
            elif hparam_type == 'float':
                # numerical item, take random value in range provided
                hparams[key] = round(np.random.uniform(
                    value[1][0], value[1][1]), 4)
            else:
                hparams[key] = np.random.randint(value[1][0], value[1][1] + 1)
    else:
        # use default hyperparameters
        for key, value in hparams_dict.items():
            hparams[key] = value[0]

    return hparams


def get_seed(i):
    """Get random seed for validation split"""
    if i == 0:
        seed = 42
    else:
        seed = np.random.randint(100000)
    return seed


def create_tf_saver(args, pretrain_args, reuse_vars_dict):
    """Creates Saver for TensorFlow model, which is more complicated when
    pretraining as the pretrained model will have saved variable names that
    differ from current model being trained.
    .
    Multiple character model shares parameters for embedding and RNN (weights
    and biases but uses separate parameters for the projection layer (which
    have different TF variable names). If pre-training with a different
    dataset, must initialize the projection layer variables from the restored
    model projection layer parameters. The for loop below is doing this.
    """
    if args.pretrain_data_path == '':
        return tf.train.Saver()
    else:
        saver_dict = {
            'Model/embedding': reuse_vars_dict['Model/embedding'],
            'Model/RNN/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights':
                reuse_vars_dict[
                    'Model/RNN/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights'],
            'Model/RNN/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases':
                reuse_vars_dict[
                    'Model/RNN/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases']}

        for pretrain_speaker in pretrain_args.speakers:
            load_w = 'Model/softmax_w_' + pretrain_speaker
            load_b = 'Model/softmax_b_' + pretrain_speaker
            for speaker in args.speakers:
                speaker_w = 'Model/softmax_w_' + speaker
                speaker_b = 'Model/softmax_b_' + speaker
                saver_dict[load_w] = reuse_vars_dict[speaker_w]
                saver_dict[load_b] = reuse_vars_dict[speaker_b]

        saver = tf.train.Saver(saver_dict)
        return saver
