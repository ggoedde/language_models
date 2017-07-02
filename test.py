"""
Test a trained model on specified speakers/dataset.

To run test.py directly, must provide all arguments listed below. Otherwise,
use run.py and set run_type = 'test'
In both instances, must have a trained and saved model that can be loaded.
"""

import os
import argparse

import _pickle
import numpy as np
import tensorflow as tf

import model
import utils
import data_reader


def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='Where the test data is stored.')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Saved model/config location.')
    parser.add_argument('--speakers', type=str, default=None,
                        help='Name of speakers/characters.')
    parser.add_argument('--pretrain_data_path', type=str, default=None,
                        help='Name of path + file used for pretraining.')
    args = parser.parse_args()

    args.speakers = args.speakers.split(',')
    # load arguments used during training/pretraining
    with open(os.path.join(args.load_path, 'config.pkl'), 'rb') as f:
        train_args, pretrain_args = _pickle.load(f)

    _test(train_args, pretrain_args, args)


def _run_epoch(sess, model, args, data, train_ind, test_ind):
    """Runs the model on the given data."""
    # total cost and number of words evaluated in this epoch
    costs, total_words = 0.0, 0.0
    state = sess.run(model.initial_state)

    for step, (x, y) in enumerate(data_reader.batch_iterator(
            data[test_ind], model.config['batch_size'])):
        # return parameters after running sess
        fetches = {
            'cost': model.cost[train_ind],
            'final_state': model.final_state,
            'seq_len': model.seq_len
        }

        # create dict to feed input, targets, and rnn into TF session
        feed_dict = utils.create_feed_dict(model, args, x, y, state)
        # run all parameters in fetches dict
        vals = sess.run(fetches, feed_dict)

        costs += vals['cost']
        # number of words evaluated
        total_words += np.sum(vals['seq_len'])

        # use perplexity to evaluate language models
        perplexity = np.exp(costs / total_words)

    return perplexity


def _test(train_args, pretrain_args, args):
    """Test saved model on specified speakers."""
    print('Testing', ', '.join(args.speakers), '...')

    # update args with new test args
    test_args = utils.set_new_args(train_args, args)
    # get test data and id_to_word lookup
    _, _, test_data, id_to_word = data_reader.get_data(test_args)
    # set configurations/hyperparameters for model
    _, test_config = utils.set_config(test_args, id_to_word)

    # model requires init embed but this will be overridden by restored model
    init_embed = utils.init_embedding(id_to_word, dim=test_args.embed_size,
                                      init_scale=test_args.init_scale,
                                      embed_path=test_args.embed_path)

    with tf.Graph().as_default():
        with tf.name_scope('Test'):
            with tf.variable_scope('Model', reuse=None):
                m_test = model.Model(test_args, is_training=False,
                                     config=test_config,
                                     init_embed=init_embed, name='Test')
                m_test.build_graph()

        init = tf.global_variables_initializer()

        # if pretrained, must create dict to initialize TF Saver
        if bool(pretrain_args):
            # get trainable variables and convert to dict for Saver
            reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
            # create saver for TF session (see function for addl details)
            saver = utils.create_tf_saver(args, pretrain_args, reuse_vars_dict)
        else:
            saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            print('Restoring model...')
            saver.restore(sess, test_args.load_path)

            # test model on specified speakers
            for test_ind, test_speaker in enumerate(test_args.speakers):
                for train_ind, train_speaker in enumerate(train_args.speakers):
                    print('Testing {0} with {1} model'.format(
                        test_speaker, train_speaker))
                    test_perplexity = _run_epoch(sess, m_test, test_args,
                                                 test_data, train_ind, test_ind)
                    print('Test Perplexity: {0:.3f}'.format(test_perplexity))


if __name__ == '__main__':
    main()
