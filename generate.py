"""
Utilities used to generate text from trained model, or to display
generated text while training.

To run generate.py directly, must provide arguments listed below. Otherwise,
use run.py and set run_type = 'generate'. In both instances, must have a
trained and saved model that can be loaded.
"""

import os
import argparse

import numpy as np
import tensorflow as tf
import _pickle

import data_reader
import model
import utils


def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='Where the training/test data is stored.')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Saved model/config location.')
    parser.add_argument('--speakers', type=str, default='jerry',
                        help='Name of speakers/characters.')
    parser.add_argument('--pretrain_data_path', type=str, default=None,
                        help='Name of path + file used for pretraining.')
    parser.add_argument('--temp', type=float, default=0.9,
                        help='Temperature used to generate text.')
    args = parser.parse_args()

    args.speakers = args.speakers.split(',')
    # load arguments used during training
    with open(os.path.join(args.load_path, 'config.pkl'), 'rb') as f:
        train_args, pretrain_args = _pickle.load(f)

    _generate(train_args, pretrain_args, args)


def generate_text(sess, m_gen, id_to_word, train_ind, temp=0.9):
    """Generate sentences of text by predicting one word after another."""
    # generate one word at a time
    m_gen.config['num_steps'] = 1

    state = sess.run(m_gen.initial_state)
    # <eos> marker has id = 1
    x = np.ones((m_gen.config['batch_size'], 1))
    count = 0
    num_sentences = 3  # number of sentences to generate
    word_to_id = dict(zip(id_to_word.values(), id_to_word.keys()))
    print('Generating text (temp={})...'.format(temp))
    while count < num_sentences:
        # get logits from RNN model
        logits, state = sess.run([m_gen.logits[train_ind], m_gen.final_state],
                                 {m_gen.input_data: x,
                                 m_gen.initial_state: state})

        # get probabilities for each word in dictionary
        prediction = np.exp(logits[0] / temp)
        prediction /= np.sum(prediction)

        word = 3  # 'UNK' word id = 3
        while word == 3:  # don't select rare words that are mapped to 'UNK'
            # randomly select word given probabilities
            word = np.random.choice(len(prediction), p=prediction)
        x[0][0] = word
        # 1 is eos marker, so generate new sentence if this is selected
        if word == 1:
            break
        print(id_to_word[word], end=' ')
        if word == word_to_id['.'] or word == word_to_id['?']:
            # start new line if end of sentence
            print()
            count += 1
            state = sess.run(m_gen.initial_state)


def _generate(train_args, pretrain_args, args):
    """Restore trained model and use to generate sample text."""
    # update args with new generate args
    gen_args = utils.set_new_args(train_args, args)
    # get id_to_word lookup
    _, _, _, id_to_word = data_reader.get_data(gen_args)
    # # get hyperparameters corresponding to text generation
    gen_config, _ = utils.set_config(gen_args, id_to_word)

    # model requires init embed but this will be overridden by restored model
    init_embed = utils.init_embedding(id_to_word, dim=gen_args.embed_size,
                                      init_scale=gen_args.init_scale,
                                      embed_path=gen_args.embed_path)

    with tf.Graph().as_default():
        # use Train name scope as this contains trained model parameters
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                m_gen = model.Model(gen_args, is_training=False,
                                    config=gen_config, init_embed=init_embed,
                                    name='Generate')
                m_gen.build_graph()

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
            saver.restore(sess, gen_args.load_path)

            # generate text for all specified speakers
            for gen_ind, gen_speaker in enumerate(gen_args.speakers):
                print('Generating text for {0}'.format(gen_speaker))
                for train_ind, train_speaker in enumerate(train_args.speakers):
                    if gen_speaker == train_speaker:
                        generate_text(sess, m_gen, id_to_word, train_ind, args.temp)


if __name__ == '__main__':
    main()
