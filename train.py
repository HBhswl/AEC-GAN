from typing import Tuple
import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import torch
from sklearn.model_selection import train_test_split

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import pickle_it, load_pickle
from lib.utils import sample_indices


from pathos.multiprocessing import ProcessingPool as Pool

def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    if key in SIGCWGAN_CONFIGS.keys():
        return SIGCWGAN_CONFIGS[key]
    elif dataset[:3] == 'ett':
        return SIGCWGAN_CONFIGS['ETT']
    else:
        return SIGCWGAN_CONFIGS['Other']


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo

def run_test(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))

    # make dirs
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    base_config.experiment_directory = experiment_directory

    # set seeds
    set_seed(base_config.seed)

    # obtain data
    pipeline, x_real = get_data(dataset, base_config.p, base_config.q, **data_params)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:] #train_test_split(x_real, train_size = 0.8)

    del x_real
    del x_real_test

    # set algo
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)

    # load model 
    G_weight = load_pickle(os.path.join(experiment_directory, 'G_weights.torch'))
    algo.G.load_state_dict(G_weight)

    with torch.no_grad():
        size = int(x_real_train.shape[0] * 5.0) # generate 5 times the data in training set
        batch_size = base_config.batch_size
        results = np.zeros((size, 4000, x_real_train.shape[2]))
        conditions = np.zeros((size, base_config.p, x_real_train.shape[2]))
        start = 0
        while start < size:
            this_size = min(batch_size, size - start)
            indices = sample_indices(x_real_train.shape[0], this_size)
            x_past = x_real_train[indices, :base_config.p].clone().to(base_config.device)
            x_fake_future = algo.G.sample(4000, x_past)

            if isinstance(x_fake_future, Tuple):
                x_fake = x_fake_future[0]
            else:
                x_fake = x_fake_future
            x_fake = pipeline.inverse_transform(x_fake)
            
            results[start:start+this_size:, :, :] = x_fake.cpu().numpy()
            start += this_size
            print(f'{start}/{size} = {round(start/size, 4)}')

        np.save(os.path.join(experiment_directory, 'gen.npy'), results)
        print('complete generation')
        

def run(algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    base_config.experiment_directory = experiment_directory

    set_seed(base_config.seed)
    pipeline, x_real = get_data(dataset, base_config.p, base_config.q, **data_params)
    x_real = x_real # .to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:] 

    # set algorithm
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    
    # Train the algorithm
    algo.fit()
    
    # Pickle generator weights, real path and hqyperparameters.
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    pickle_it(algo.D.to('cpu').state_dict(), pt.join(experiment_directory, 'D_weights.torch'))
    
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


def get_dataset_configuration(dataset):
    if dataset in ['sine', 'square', 'triangle', 'sawtooth', 'etth1', 'etth2', 'ettm1', 'ettm2']:
        generator = [('a', dict())]
    elif dataset in ['ILI', 'us_births'] :
        generator = [('a', dict())]
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator


def main(args):
    if not pt.exists('./data'):
        os.mkdir('./data')

    args.use_cuda = True
    print('Start of training. CUDA: %s' % args.use_cuda)
    for dataset in args.datasets:
        hidden_dims = args.hidden_dims
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig(
                        device='cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu',
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=hidden_dims,
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,
                    mc_samples=1000,
                    eps=args.eps,
                    noise_type=args.noise_type,
                    use_ec=args.use_ec,
                )
                set_seed(seed)
                generator = get_dataset_configuration(dataset)
                for spec, data_params in generator:
                    if args.test:
                        run_test(
                            algo_id=algo_id,
                            base_config=base_config,
                            data_params=data_params,
                            dataset=dataset,
                            base_dir=args.base_dir,
                            spec=spec,
                        )
                    else:
                        run(
                            algo_id=algo_id,
                            base_config=base_config,
                            data_params=data_params,
                            dataset=dataset,
                            base_dir=args.base_dir,
                            spec=spec,
                        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    parser.add_argument('-datasets', default=['etth1'], nargs="+")
    parser.add_argument('-algos', default=['AECGAN'], nargs="+")


    # Algo hyperparameters
    parser.add_argument('-batch_size', default=16, type=int)
    parser.add_argument('-p', default=168, type=int)
    parser.add_argument('-q', default=168, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple)
    parser.add_argument('-total_steps', default=10000, type=int)
    parser.add_argument('-noise_type', default='min_adv', type=str)
    parser.add_argument('-use_ec', default=2, type=int)

    # other 
    parser.add_argument('-eps', default=0.01, type=float)
    parser.add_argument('-test', action='store_true')
    
    args = parser.parse_args()
    main(args)
