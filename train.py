"""
Train model on specified data set using various hyperparameters. Additionally
can
-create TensorBoard summaries
-save model
-load pretrained model
-writes results to sql database

To run train.py directly, must provide --data_path and --speakers arguments.
Otherwise, use run.py and set run_type = 'train'
"""

import time
import os
import argparse

import _pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.training.training_util import get_or_create_global_step

import model
import utils
import data_reader
import generate
import results_db


def main():
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None,
                        help='Where the training/test data is stored.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Model output directory.')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Location of pre-trained model.')
    parser.add_argument('--pretrain_data_path', type=str, default=None,
                        help='Name of path + file used for pretraining.')
    parser.add_argument('--monte_carlo_cv_num', type=int, default=0,
                        help='Run monte carlo cross validation by random sub- '
                             'sampling a validation dataset. Argument defines '
                             'number of sub-samples to generate.')
    parser.add_argument('--display_text', type=str, default=True,
                        help='Display top 3 words + generate sample text '
                             'during training.')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='RNN cell type. Possible options: rnn, lstm, gru.')
    parser.add_argument('--speakers', type=str, default=None,
                        help='Name of speakers/characters. Use "not_jerry" or '
                             '"movies" if pretraining.')
    parser.add_argument('--insert_db', type=str, default=False,
                        help='Write results to sql database.')
    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='Initial scale of the weights.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial value of the learning rate..')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum permissible norm of the gradient.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers in RNN.')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of unrolled steps of RNN.')
    parser.add_argument('--embed_size', type=int, default=2,
                        help='Number of units in embedding.')
    parser.add_argument('--max_epoch', type=int, default=1,
                        help='Total number of epochs for training.')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Prob of keeping weights in dropout layers.')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of sequences in each batch.')
    parser.add_argument('--embed_path', type=str, default=None,
                        help='Word embedding file path. If blank -> do not use'
                             'pre-trained word embeddings.')
    args = parser.parse_args()

    # args.speakers is (for example) 'jerry,kramer,elaine'
    args.speakers = args.speakers.split(',')

    # create save_path folder if doesn't already exist
    if args.save_path != '':
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    # get arguments used to pretrain model
    if args.load_path != '':
        with open(os.path.join(args.load_path, 'config.pkl'), 'rb') as f:
            pretrain_args, _ = _pickle.load(f)
    else:
        pretrain_args = {}

    # write arguments to pickle file so can be restored
    if args.save_path != '':
        with open(os.path.join(args.save_path, 'config.pkl'), 'wb') as f:
            _pickle.dump([args, pretrain_args], f)

    _train(args, pretrain_args)


def _run_epoch(sess, model, args, data, index=0, tb_summaries=None,
               id_to_word=None, train_op=None, verbose=False):
    """Runs the model on the given data and displays metrics to monitor
    progress.
    """
    epoch_start_time = time.time()
    # total cost and number of words evaluated in this epoch
    costs, total_words = 0.0, 0.0
    # epoch size is number of batches in each epoch
    epoch_size = (len(data[index]) - 1) // model.config['batch_size']
    state = sess.run(model.initial_state)

    # iterate through batches
    for step, (x, y) in enumerate(data_reader.batch_iterator(
            data[index], model.config['batch_size'])):
        # return these parameters after running TF session
        fetches = {
            'cost': model.cost[index],
            'final_state': model.final_state,
            'seq_len': model.seq_len
            }
        # only train model has optimizer operation
        if train_op is not None:
            fetches['train_op'] = train_op[index]

        # create dict to feed input, targets, and rnn into TF session
        feed_dict = utils.create_feed_dict(model, args, x, y, state)
        # run all parameters in fetches dict
        vals = sess.run(fetches, feed_dict)

        costs += vals['cost']
        # number of words evaluated
        total_words += np.sum(vals['seq_len'])
        # use perplexity to evaluate language models
        perplexity = np.exp(costs / total_words)

        if verbose and step % (epoch_size // 2) == 1:
            # display perplexity and top word predictions for sequence
            _display_epoch_metrics(step, epoch_size, perplexity, total_words,
                                   epoch_start_time, args, model, sess,
                                   index, feed_dict, vals, id_to_word, y)

    # generate sample text while training to monitor progress
    if args.display_text == 'True' and model.name == 'Train':
        generate.generate_text(sess, model, id_to_word, train_ind=index)

    # write TensorBoard summaries for Train/Valid
    if args.save_path != '' and model.name != 'Test':
        summary = sess.run(tb_summaries.summary_op,
                           {tb_summaries.ppl_summary: perplexity})
        model.file_writer.add_summary(summary, get_or_create_global_step().eval())

    return perplexity


def _display_epoch_metrics(step, epoch_size, perplexity, total_words,
                           epoch_start_time, args, model, sess,
                           index, feed_dict, vals, id_to_word, y):
    """Display perplexities during epoch and top word predictions for a
    sequence.
    """
    # display perplexity during epochs
    print('{0:.3f} perplexity: {1:.3f} speed: {2:.0f} wps'.format(
        step * 1.0 / epoch_size, perplexity,
        total_words / (time.time() - epoch_start_time)))

    if args.display_text == 'True' and model.name == 'Train':
        # display top 3 word predictions for sequence while training
        batch_logits = sess.run(model.logits[index], feed_dict)
        pred = np.fliplr(batch_logits.argsort()[:, -3:])
        seq_len0 = int(vals['seq_len'][0])
        for i in range(seq_len0):
            print('top 3: {0:15} {1:15} {2:15} actual: {3}'.format(
                id_to_word[pred[i][0]], id_to_word[pred[i][1]],
                id_to_word[pred[i][2]], id_to_word[int(y[0][i])]))


def _update_ppls(ppls, epoch=0, speaker='', ppl=0, dataset='', initialize=False):
    """Update perplexity dict that stores best perplexities."""
    improved = False
    if initialize:
        # initialize all speaker perplexities with blank entry
        for spk in ['not_jerry', 'george', 'elaine', 'kramer']:
            ppls['best_train_ppl_' + spk] = ''
            ppls['best_train_ppl_epoch_' + spk] = ''
            ppls['best_valid_ppl_' + spk] = ''
            ppls['best_valid_ppl_epoch_' + spk] = ''
            ppls['test_ppl_' + spk] = ''
    else:
        if epoch == 1:  # at first epoch ppl is lowest ppl
            ppls['best_' + dataset + '_ppl_' + speaker] = ppl
            ppls['best_' + dataset + '_ppl_' + 'epoch_' + speaker] = epoch
        else:
            # if current ppl < best ppl, update ppl dict
            if ppl < ppls['best_' + dataset + '_ppl_' + speaker]:
                ppls['best_' + dataset + '_ppl_' + speaker] = ppl
                ppls['best_' + dataset + '_ppl_' + 'epoch_' + speaker] = epoch
                improved = True
    return ppls, improved


def _train(args, pretrain_args):
    """Train the language model.

    Creates train/valid/test models, runs training epochs, saves model and
    writes results to database if specified.
    """
    start_time = time.time()
    print('Training', ', '.join(args.speakers), '...')

    # randomly sample validation set monte_carlo_cv_num times
    for num in range(args.monte_carlo_cv_num):
        # get seed used to sub-sample validation dataset (use 42 for 1st run)
        seed = utils.get_seed(num)

        # get train/valid/test data and convert to sequences
        train_data, valid_data, test_data, id_to_word = data_reader.get_data(
            args, seed=seed)
        # set configurations/hyperparameters for model
        config, test_config = utils.set_config(args, id_to_word)

        # initialize word embeddings
        init_embed = utils.init_embedding(id_to_word, dim=args.embed_size,
                                          init_scale=args.init_scale,
                                          embed_path=args.embed_path)

        with tf.Graph().as_default():
            # initializer used to initialize TensorFlow variables
            initializer = tf.random_uniform_initializer(-config['init_scale'],
                                                        config['init_scale'])
            # create Train model
            with tf.name_scope('Train'):
                with tf.variable_scope('Model', reuse=None,
                                       initializer=initializer):
                    m_train = model.Model(args, is_training=True, config=config,
                                          init_embed=init_embed, name='Train')
                    m_train.build_graph()

            # create Valid model
            with tf.name_scope('Valid'):
                with tf.variable_scope('Model', reuse=True,
                                       initializer=initializer):
                    m_valid = model.Model(args, is_training=False, config=config,
                                          init_embed=init_embed, name='Valid')
                    m_valid.build_graph()

            # create Test model
            with tf.name_scope('Test'):
                with tf.variable_scope('Model', reuse=True,
                                       initializer=initializer):
                    m_test = model.Model(args, is_training=False, config=test_config,
                                         init_embed=init_embed, name='Test')
                    m_test.build_graph()

            # create summaries to be viewed in TensorBoard
            tb_summaries = utils.TensorBoardSummaries()
            tb_summaries.create_ops()

            init = tf.global_variables_initializer()

            # if pretrained, must create dict to initialize TF Saver
            if bool(pretrain_args):
                # get trainable variables and convert to dict for Saver
                reuse_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES)
                reuse_vars_dict = dict(
                    [(var.op.name, var) for var in reuse_vars])
                # create saver for TF session (see function for addl details)
                saver = utils.create_tf_saver(args, pretrain_args,
                                              reuse_vars_dict)
            else:
                saver = tf.train.Saver()

            # ppls dict has perplexities that are stored in results database
            ppls = {}
            ppls, _ = _update_ppls(ppls, initialize=True)

            with tf.Session() as sess:
                sess.run(init)

                if args.load_path != '':
                    print('Restoring model...')
                    saver.restore(sess, args.load_path)

                for epoch in range(config['max_epoch']):
                    print('Epoch: {0} Learning rate: {1:.3f}\n'.format(
                        epoch + 1, sess.run(m_train.lr)))
                    for i, speaker in enumerate(args.speakers):
                        print('Training {0} ...'.format(speaker))

                        # run epoch on training data
                        train_perplexity = _run_epoch(sess, m_train, args, train_data,
                                                      i, tb_summaries, id_to_word,
                                                      train_op=m_train.train_op,
                                                      verbose=True)
                        print('Epoch: {0} Train Perplexity: {1:.3f}'.format(
                            epoch + 1, train_perplexity))
                        ppls, _ = _update_ppls(ppls, epoch=epoch+1,
                                            speaker=speaker,
                                            ppl=train_perplexity,
                                            dataset='train')

                        print('Validating...')
                        # run epoch on validation data
                        valid_perplexity = _run_epoch(sess, m_valid, args,
                                                      valid_data, i, tb_summaries,
                                                      id_to_word, verbose=True)
                        print('Epoch: {0} Valid Perplexity: {1:.3f}'.format(
                            epoch + 1, valid_perplexity))
                        ppls, improved = _update_ppls(ppls, epoch=epoch+1,
                                            speaker=speaker,
                                            ppl=valid_perplexity,
                                            dataset='valid')

                        if improved:
                            # save model if valid ppl is lower than current
                            # best valid ppl
                            if args.save_path != '':
                                print('Saving model to {0}.'.format(
                                    args.save_path))
                                saver.save(sess, args.save_path)

                for i, speaker in enumerate(args.speakers):
                    print('Testing {0} ...'.format(speaker))
                    print('Restoring best model for testing...')
                    saver.restore(sess, args.save_path)
                    # run model on test data
                    test_perplexity = _run_epoch(sess, m_test, args, test_data, i)
                    ppls['test_ppl_' + speaker] = test_perplexity
                    print('Test Perplexity: {0:.3f}'.format(test_perplexity))

            if args.insert_db == 'True':
                # write params/config/results to sql database
                results_db.insert_results(args, config, start_time, ppls)


if __name__ == '__main__':
    main()
