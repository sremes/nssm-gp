#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiments using datasets defined in datasets.py

@author: Sami Remes
"""

import argparse
import os
import shutil
from datetime import datetime

from experiments.datasets import Datasets

import tensorflow as tf
import gpflow
from gpflow.saver import Saver
from gpflow import settings
import gpflow.training.monitor as mon

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.utils import shuffle

from nssm_gp.initializers import init_gsm, init_spectral, init_neural
from nssm_gp.spectral_kernels import SMKernel


float_type = settings.dtypes.float_type

# Model / inference hyperparameters
NUM_INDUCING_POINTS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
Q = 3
MAX_FREQ = 20.0
MAX_LEN = 10.0
EPOCHS = 500
LATENT_ELL = 1.0


def simple_model(x, y, likelihood, M=NUM_INDUCING_POINTS, bs=BATCH_SIZE, ARD=True):
    # Gaussian kernel
    ell = np.std(x)
    var = np.std(y)
    kern_rbf = gpflow.kernels.RBF(x.shape[1], variance=var, lengthscales=ell, ARD=ARD)
    # Randomly select inducing point locations among given inputs
    idx = np.random.randint(len(x), size=M)
    Z = x[idx, :].copy()
    # Create SVGP with Gaussian likelihood
    m_rbf = gpflow.models.SVGP(x, y, kern_rbf, likelihood,
                               Z, minibatch_size=bs)
    return m_rbf


model_functions = {
    'gsm': lambda x, y, likelihood, q=Q, ARD=False, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS: init_gsm(x, y, M=M, max_freq=MAX_FREQ,
                                 Q=q, minibatch_size=bs, ell=LATENT_ELL, n_inits=5, noise_var=0.1, ARD=ARD, likelihood=likelihood),
    'sm': lambda x, y, likelihood, q=Q, ARD=False, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS: init_spectral(x, y, kern=SMKernel, n_inits=5,
                                     M=M, Q=q, minibatch_size=bs, noise_var=0.1, ARD=ARD, likelihood=likelihood),
    'rbf': simple_model,
    'neural': lambda x, y, likelihood, q=Q, bs=BATCH_SIZE, M=NUM_INDUCING_POINTS, ARD=None: init_neural(x, y, n_inits=5,
                                                                                  M=M, Q=q, minibatch_size=bs,
                                                                                  likelihood=likelihood,
                                                                                  hidden_sizes=(32, 32)),
}

if __name__ == '__main__':
    print('Tensorflow: %s' % tf.__version__)
    print('GPflow: %s' % gpflow.__version__)

    # Datasets available
    datasets = Datasets('./data/')


    # Check arguments
    parser = argparse.ArgumentParser(description="Run experiment using a dataset and kernel.")
    parser.add_argument('--kernel', help='Kernel to use', choices=list(model_functions.keys()), required=True)
    parser.add_argument('--data', help='Dataset to use', choices=list(datasets.all_datasets.keys()), required=True)
    parser.add_argument('--q', help='Number of mixture components', default=Q)
    parser.add_argument('--ard', help='Use ARD for lengthscales', action='store_true')
    parser.add_argument('--lr', help='Learning rate', default=LEARNING_RATE)
    parser.add_argument('--bs', help='Batch size', default=BATCH_SIZE)
    parser.add_argument('--m', help='Inducing point size', default=NUM_INDUCING_POINTS)
    args = parser.parse_args()
    print(args)

    # Load data
    print('Loading data...')
    dataset = datasets.all_datasets[args.data]
    data = dataset.get_data()

    # Reset seed to get different shuffles (and model initialization below)
    np.random.seed(None)
    data['X'], data['Y'] = shuffle(data['X'], data['Y'])

    # Select model type
    if dataset.type == 'regression':
        likelihood = gpflow.likelihoods.Gaussian(0.1**2)
    elif dataset.type == 'classification':
        likelihood = gpflow.likelihoods.Bernoulli()
    else:
        raise ValueError('Dataset type unknown: {}.'.format(dataset.type))

    # Initialize
    print('Initializing model...')
    if args.kernel == 'rbf':
        model = model_functions[args.kernel](data['X'], data['Y'], ARD=args.ard, likelihood=likelihood,
                                             bs=int(args.bs), M=int(args.m))
    else:
        model = model_functions[args.kernel](data['X'], data['Y'], q=int(args.q), ARD=args.ard, likelihood=likelihood,
                                             bs=int(args.bs), M=int(args.m))

    # Create monitoring
    session = model.enquire_session()
    global_step = mon.create_global_step(session)
    model_name = '{data}_{kernel}_Q-{q}_ARD-{ard}_lr{lr}_bs{bs}_{date}'.format(data=args.data, kernel=args.kernel,
                                                                               q=args.q, ard=args.ard, lr=args.lr, bs=args.bs,
                                                                               date=datetime.now().strftime('%y%m%d-%H%M%S'))
    tensorboard_dir = 'tensorboard/' + model_name
    shutil.rmtree(tensorboard_dir, ignore_errors=True)
    with mon.LogdirWriter(tensorboard_dir) as writer:
        tensorboard_task = mon.ModelToTensorBoardTask(writer, model, only_scalars=False)\
            .with_name('tensorboard')\
            .with_condition(mon.PeriodicIterationCondition(100))\
            .with_exit_condition(True)
        print_task = mon.PrintTimingsTask() \
            .with_name('print') \
            .with_condition(mon.PeriodicIterationCondition(500))

        # Create optimizer
        epoch_steps = len(data['X']) // int(args.bs)
        learning_rate = tf.train.exponential_decay(float(args.lr), global_step, decay_steps=epoch_steps, decay_rate=0.99)
        optimizer = gpflow.train.AdamOptimizer(learning_rate)

        maxiter = epoch_steps * EPOCHS
        print('Optimizing model (running {} iterations)...'.format(maxiter))
        with mon.Monitor([tensorboard_task, print_task], session, global_step, print_summary=True) as monitor:
            optimizer.minimize(model, maxiter=maxiter, step_callback=monitor, global_step=global_step)

    # Save model
    print('Saving model...')
    fname = 'models/{model_name}.gpflow'.format(model_name=model_name)
    if os.path.exists(fname):
        os.remove(fname)
    try:
        Saver().save(fname, model)
    except ValueError as e:
        print('Failed to save model:', e)

    # Evaluate model
    print('Evaluating model...')
    metrics = []
    for X, Y in [('X', 'Y'), ('Xs', 'Ys')]:
        y, _ = model.predict_y(data[X])
        y_bin = y > 0.5
        logp = model.predict_density(data[X], data[Y])

        if dataset.type == 'regression':
            metrics += [{
                'mse': mean_squared_error(data[Y], y),
                'mae': mean_absolute_error(data[Y], y),
                'logp': logp.mean(),
            }]
        elif dataset.type == 'classification':
            metrics += [{
                'logp': logp.mean(),
                'accuracy': accuracy_score(data[Y], y_bin),
                'precision': precision_score(data[Y], y_bin),
                'recall': recall_score(data[Y], y_bin),
                'f1': f1_score(data[Y], y_bin),
                'auc': roc_auc_score(data[Y], y)
            }]
    print(metrics)
    print('Saving results...')
    filename = 'results/{model_name}.csv'.format(model_name=model_name)
    df = pd.DataFrame(metrics, index=['training', 'test'])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False)
    else:
        df.to_csv(filename, mode='w', header=True)
    

