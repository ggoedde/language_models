"""
Run train.py, test.py, and generate.py with various hyperparameters and
functionality (monte carlo cross validation, random hyperparameter search).
"""

import subprocess
import utils


def main():
    data_path = 'data/seinfeld/text/'  # folder with text file
    save_path = ''  # save path for model and TensorBoard summaries
    load_path = ''  # location of saved model
    pretrain_data_path = ''  # use full path including file name
    # speakers: use 'not_jerry' or 'movies' if pre-training with those files
    speakers = 'jerry'  # 'jerry' or 'jerry,elaine,kramer' etc
    display_text = True  # display top words and generated text while training
    insert_db = False  # insert results into sql database
    max_epoch = 10  # maximum epochs to run during training
    rand_hparams = False  # get random hyperparameters for training
    monte_carlo_cv_num = 1  # number of monte carlo cross validations
    use_glove = False  # use pre-trained GLoVe word vectors

    run_type = 'train'  # set to 'train', 'test', or 'generate'

    # Hyperparameters
    # can either 1. define specific hyperparameters (left entry for each
    # hparam in hparams_dict) or 2. specify range to get random hyperparameter
    # from (middle entry of hparams_dict). set rand_params=True if random
    # parameters are desired.
    # each entry in hparams_dict has following format
    # name: [selected, [range_to_search], precision/type], note that the
    # upper range of 'int' variables is exclusive
    hparams_dict = {
        'rnn_type': ['lstm', ['lstm', 'lstm'], 'cat'],
        'init_scale': [0.1, [0.01, 0.25], 'float'],
        'lr': [0.001, [0.0001, 0.01], 'float'],
        'max_grad_norm': [3, [1, 5], 'int'],
        'num_layers': [1, [1, 1], 'int'],
        'num_steps': [20, [5, 40], 'int'],
        'embed_size_pre': [100, [100, 100], 'cat'],
        'embed_size_no_pre': [100, [100, 100], 'int'],
        'keep_prob': [0.5, [0.2, 0.7], 'float'],
        'batch_size': [32, [20, 128], 'int']
    }
    # number of times to run the model. Useful if using random hparams
    model_runs = 1

    for _ in range(model_runs):
        hparams = utils.get_random_hparams(hparams_dict,
                                           rand_hparams=rand_hparams)
        # set path where GloVe vectors are located
        if not use_glove:
            embed_path = ''
            hparams['embed_size'] = hparams['embed_size_no_pre']
        elif hparams['embed_size_pre'] == 50:
            embed_path = 'data/glove/glove.6B.50d.txt'
            hparams['embed_size'] = hparams['embed_size_pre']
        else:
            embed_path = 'data/glove/glove.6B.100d.txt'
            hparams['embed_size'] = hparams['embed_size_pre']

        if run_type == 'train':
            subprocess.call('CUDA_VISIBLE_DEVICES=0 python3.5 train.py '
                            '--data_path={dp} --save_path={sp} --load_path={lp} '
                            '--pretrain_data_path={pd} --rnn_type={rnn} '
                            '--speakers={ss} --display_text={dt} --insert_db={db} '
                            '--init_scale={sc} --lr={lr} --max_grad_norm={mgn} '
                            '--num_layers={nl} --num_steps={ns} --embed_size={es} '
                            '--max_epoch={me} --keep_prob={kp} --batch_size={bs} '
                            '--embed_path={ep} --monte_carlo_cv_num={mc}'.format(
                dp=data_path, sp=save_path, lp=load_path, ep=embed_path, ss=speakers,
                dt=display_text, db=insert_db, me=max_epoch, pd=pretrain_data_path,
                mc=monte_carlo_cv_num, rnn=hparams['rnn_type'],
                sc=hparams['init_scale'], lr=hparams['lr'],
                mgn=hparams['max_grad_norm'], nl=hparams['num_layers'],
                ns=hparams['num_steps'], es=hparams['embed_size'],
                kp=hparams['keep_prob'], bs=hparams['batch_size']), shell=True)

        elif run_type == 'test':
            subprocess.call('CUDA_VISIBLE_DEVICES=0 python3.5 test.py --data_path={dp} '
                            '--load_path={lp} --speakers={ss} '
                            '--pretrain_data_path={pd}'.format(
                dp=data_path, lp=load_path, ss=speakers,
                pd=pretrain_data_path), shell=True)

        else:
            subprocess.call('CUDA_VISIBLE_DEVICES=0 python3.5 generate.py --data_path={dp} '
                            '--load_path={lp} --speakers={ss} '
                            '--pretrain_data_path={pd}'.format(
                dp=data_path, lp=load_path, ss=speakers,
                pd=pretrain_data_path), shell=True)


if __name__ == '__main__':
    main()
