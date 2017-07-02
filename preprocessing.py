"""
Utilities to clean raw text and create individual character and movie line
text files.
"""

import re
import os
import argparse

import sqlite3
import pandas as pd


MOVIES_DIR = 'data/movies'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_type',
                        type=str,
                        default='sql',
                        help='Type of file data is stored (sql or csv).')
    parser.add_argument('--file_name', type=str, default=None,
                        help='Name of sql db or csv file.')
    parser.add_argument('--sql_table', type=str, default='utterance',
                        help='Table in sql db with data.')
    parser.add_argument('--dataset_name', type=str, default='seinfeld',
                        help='Name of dataset / tv show.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Location to save text files.')
    args = parser.parse_args()

    # read table from sql database to pandas DataFrame
    df = _sql_to_pd(args.file_name, args.sql_table)
    # create text file with entire seinfeld dataset
    _df_to_clean_txt(df, args.dataset_name, split='',
                     dataset_name=args.dataset_name)
    # generate train/test text files for each speaker
    _create_speaker_files(df, args.dataset_name)
    # generate files that don't include target speaker to use in pre-training
    _create_not_speaker_files(df, args.dataset_name)

    # create clean text file with movie lines dataset
    read_name = 'movie_lines_raw.txt'
    write_name = 'movies.txt'
    # create text file with cleaned movie lines
    _movies_clean_txt(os.path.join(MOVIES_DIR, read_name),
                      os.path.join(MOVIES_DIR, write_name))
    # split movie lines into train/test files
    _movies_train_test(os.path.join(MOVIES_DIR, write_name))


def _sql_to_pd(db, table):
    """Read sql table into pandas DataFrame."""
    conn = sqlite3.connect(db)
    query = 'SELECT * FROM {}'.format(table)
    
    return pd.read_sql_query(query, conn)


def _df_to_clean_txt(df, name, split, dataset_name):
    """Writes pandas speaker DataFrame to text file, with each row
    representing one continuous utterance.
    """
    path = './data/' + dataset_name + '/text'
    file = name + split + '.txt'
    filename = os.path.join(path, file)

    with open(filename, 'w') as f:
        for row in range(df.shape[0]):
            # clean and write to text file
            f.write(_clean_str(df['text'].iloc[row],
                               punctuation=False,
                               dataset_name=dataset_name) + '\n')


def _movies_clean_txt(read_file, write_file):
    """Clean movie lines file and write to new text file."""
    with open(read_file, 'rb') as f_read:
        lines = f_read.read().split(b'\n')

        with open(write_file, 'w') as f_write:
            for line in lines:
                line = line.split(b' +++$+++ ')
                # text is last part of each line
                f_write.write(_clean_str(line[-1].strip().decode(
                    'utf-8', errors='ignore'), punctuation=False) + '\n')


def _movies_train_test(file_path):
    """Split total movie lines text file into train/test text files."""
    # read text files into pandas DataFrames
    df = pd.read_csv(file_path, header=None)
    # split train into train/test pandas data frames
    df_train, df_test = train_test_split(df, train_pct=0.70, seed=42)

    # create train file for movie scripts
    train_file = os.path.join(MOVIES_DIR, 'movies_train.txt')
    with open(train_file, 'w') as f:
            for row in range(df_train.shape[0]):
                f.write(df_train.iloc[row][0] + '\n')
    # create test file for movie scripts
    test_file = os.path.join(MOVIES_DIR, 'movies_test.txt')
    with open(test_file, 'w') as f:
            for row in range(df_test.shape[0]):
                f.write(df_test.iloc[row][0] + '\n')


def _clean_str(string, punctuation=False, dataset_name=None):
    """String cleaning/preprocessing of text data."""
    # lowercase
    string = string.lower()
    
    if dataset_name == 'seinfeld':
        # seinfeld scripts include various peculiarities
        # ( ) or [ ] is used to provide context around scene
        string = re.sub(r'\([^\(]+\)', '', string)
        string = re.sub(r'\[[^\[]+\]', '', string)

    # multiple punctuation
    string = re.sub(r'\.{2,}', ' ', string)
    string = re.sub(r'\?{2,}', '?', string)
    string = re.sub(r'\!{2,}', '!', string)
    
    # i.e. doin' --> doing
    string = re.sub(r"n\' ", 'ng ', string)
    string = re.sub(r"n\'\.", 'ng \.', string)
    
    # quotation marks
    string = re.sub(r" \'", ' ', string)
    string = re.sub(r"\' ", ' ', string)
    string = re.sub(r"\'\.", '.', string)
    string = re.sub(r"\'\?", '?', string)
    string = re.sub(r"^\'", '', string)

    string = re.sub(r'gotta', 'got to', string)
    string = re.sub(r'gonna', 'going to', string)
    string = re.sub(r'wanna', 'want to', string)

    # i.e. doin' --> doing
    # string = re.sub(r"n\' ", 'ng ', string)
    # string = re.sub(r"n\'\.", 'ng \.', string)
    
    # i.e. d'you --> do you
    string = re.sub(r"d\'", 'do ', string)
    
    # i.e. O.K. --> ok
    string = re.sub(r'o\.k\.', 'ok', string)
    
    if punctuation:
        # keep punctuation
        string = re.sub(r'\?', ' ? ', string)
        string = re.sub(r'\!', ' ! ', string)
        string = re.sub(r'\.', ' . ', string)
        string = re.sub(r'\,', ' , ', string)
        string = re.sub(r'\:', ' ', string)
        string = re.sub(r'\;', ' ; ', string)
        string = re.sub(r'\_', ' _ ', string)

    else:
        # remove punctuation
        # keep periods though for end of sentence marker
        string = re.sub(r'\?', ' ? ', string)
        string = re.sub(r'\!', ' . ', string)
        string = re.sub(r'\.', ' . ', string)
        string = re.sub(r'\,', ' ', string)
        string = re.sub(r'\:', ' ', string)
        string = re.sub(r'\;', ' ', string)
        string = re.sub(r'\_', ' ', string)
    
    # i.e. just-in-case --> just in case
    string = re.sub(r'-+', ' ', string)

    # remove misc. symbols
    string = re.sub(r'\*+', '', string)
    string = re.sub(r'\<+', '', string)
    string = re.sub(r'\>+', '', string)
    string = re.sub(r'\/+', '', string)
    string = re.sub(r'\\+', '', string)

    # contracted words, i.e. she's --> she 's
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'m", " \'m", string)

    # remove extra space
    string = re.sub(r'\s{2,}', ' ', string)

    # remove double quotes
    string = re.sub(r'"', '', string)
    
    # replace numbers with NUM
    string = re.sub(r'[0-9]+th', 'NUM', string)
    string = re.sub(r'[0-9]+', 'NUM', string)
    
    return string.strip(' ')


def _top_speakers(df, min_sequences=5000):
    """Returns list of speakers with at least min_sequences utterances."""
    seq_count = df['speaker'].value_counts()
    speakers = seq_count[seq_count > min_sequences]
    
    return speakers.keys().tolist()


def train_test_split(df, train_pct=0.70, seed=42):
    """Generates train, validation, and test splits from text in Pandas df."""
    df_train = df.sample(frac=train_pct, random_state=seed)
    df_test = df.drop(df_train.index)

    return df_train, df_test


def _create_speaker_files(df, dataset_name):
    """Generates train/test text files for each speaker"""
    speakers = _top_speakers(df)
    for speaker in speakers:
        # Select rows corresponding to specific speaker
        df_speaker = df[df['speaker'] == speaker]
        # Split data into train/val/test
        df_speaker_train, df_speaker_test = train_test_split(df_speaker)

        # write data frames to total/train/test text files
        _df_to_clean_txt(df_speaker, speaker.lower(), split='',
                         dataset_name=dataset_name)
        _df_to_clean_txt(df_speaker_train, speaker.lower(), split='_train',
                         dataset_name=dataset_name)
        _df_to_clean_txt(df_speaker_test, speaker.lower(), split='_test',
                         dataset_name=dataset_name)


def _create_not_speaker_files(df, dataset_name):
    """Generates train/test text files for all of scripts excluding speaker,
    which can be used during pre-training for the speaker model."""
    path = './data/' + dataset_name + '/text'

    speakers = _top_speakers(df)
    speakers = [speaker.lower() for speaker in speakers]
    for speaker in speakers:
        for dataname in ['train', 'test']:
            # file to write not speakers to
            append_name = os.path.join(path, 'not_'
                                       + speaker + '_' + dataname + '.txt')
            with open(append_name, 'a') as f_append:
                for not_speaker in speakers:
                    if speaker != not_speaker:
                        read_name = os.path.join(path, not_speaker
                                                 + '_' + dataname + '.txt')
                        # not speaker files to read from
                        with open(read_name, 'r') as f_read:
                            for line in f_read:
                                f_append.write(line)


if __name__ == '__main__':
    main()
