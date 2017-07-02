"""
Utilities to create and update sql database used to store results.
"""

import os
from datetime import datetime
import time
import sqlite3


DB_NAME = 'lm_results.db'
DB_PATH = 'results/'
TABLE_NAME = 'results'


def main():
    # create new table to store results of model training/testing
    create_results_table = """ CREATE TABLE IF NOT EXISTS results (
    date_time TEXT PRIMARY KEY,
    save_path TEXT, 
    load_path TEXT,
    run_time TEXT,
    addl_descr TEXT,
    data_path TEXT, 
    speakers TEXT, 
    pretrain_data_path TEXT, 
    rnn_type TEXT, 
    optimizer TEXT, 
    init_scale REAL, 
    init_lr REAL, 
    max_grad_norm REAL, 
    num_layers INT, 
    num_steps INT, 
    embed_path TEXT,
    embed_size INT, 
    max_epoch INT, 
    keep_prob REAL, 
    batch_size INT, 
    vocab_size INT, 
    best_train_ppl_jerry REAL, 
    best_train_ppl_epoch_jerry INT, 
    best_valid_ppl_jerry REAL, 
    best_valid_ppl_epoch_jerry INT, 
    test_ppl_jerry REAL,
    best_train_ppl_george REAL, 
    best_train_ppl_epoch_george INT, 
    best_valid_ppl_george REAL, 
    best_valid_ppl_epoch_george INT, 
    test_ppl_george REAL,
    best_train_ppl_elaine REAL, 
    best_train_ppl_epoch_elaine INT, 
    best_valid_ppl_elaine REAL, 
    best_valid_ppl_epoch_elaine INT, 
    test_ppl_elaine REAL,
    best_train_ppl_kramer REAL, 
    best_train_ppl_epoch_kramer INT, 
    best_valid_ppl_kramer REAL, 
    best_valid_ppl_epoch_kramer INT, 
    test_ppl_kramer REAL);"""

    conn = _create_connection(DB_PATH, DB_NAME)
    c = conn.cursor()
    c.execute('DROP TABLE results')
    _create_table(conn, create_results_table)


def _create_connection(db_path, db_name):
    """Create database connection to sqlite database."""
    return sqlite3.connect(os.path.join(db_path, db_name))


def _create_table(conn, create_tb_query):
    """Create table using create_tb_query statement."""
    c = conn.cursor()
    c.execute(create_tb_query)


def _insert_row(conn, table, vals):
    """Insert row of values into table."""
    c = conn.cursor()
    c.execute('INSERT INTO {tn} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '
              '?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '
              '?, ?, ?, ?, ?, ?, ?, ?)'.format(tn=table), vals)
    conn.commit()


def insert_results(args, config, start_time, ppls):
    """Insert data from training/testing into results table."""
    db_vals = (str(datetime.now()), args.save_path, args.load_path,
               time.time() - start_time, '', args.data_path,
               ' '.join(args.speakers), args.pretrain_data_path, args.rnn_type,
               'adam', config['init_scale'], config['lr'],
               config['max_grad_norm'], config['num_layers'], config['num_steps'],
               args.embed_path, config['embed_size'], config['max_epoch'],
               config['keep_prob'], config['batch_size'], config['vocab_size'],
               ppls['best_train_ppl_jerry'],
               ppls['best_train_ppl_epoch_jerry'],
               ppls['best_valid_ppl_jerry'],
               ppls['best_valid_ppl_epoch_jerry'],
               ppls['test_ppl_jerry'],
               ppls['best_train_ppl_george'],
               ppls['best_train_ppl_epoch_george'],
               ppls['best_valid_ppl_george'],
               ppls['best_valid_ppl_epoch_george'],
               ppls['test_ppl_george'],
               ppls['best_train_ppl_elaine'],
               ppls['best_train_ppl_epoch_elaine'],
               ppls['best_valid_ppl_elaine'],
               ppls['best_valid_ppl_epoch_elaine'],
               ppls['test_ppl_elaine'],
               ppls['best_train_ppl_kramer'],
               ppls['best_train_ppl_epoch_kramer'],
               ppls['best_valid_ppl_kramer'],
               ppls['best_valid_ppl_epoch_kramer'],
               ppls['test_ppl_kramer']
               )

    conn = _create_connection(DB_PATH, DB_NAME)
    _insert_row(conn, TABLE_NAME, db_vals)


if __name__ == '__main__':
    main()
