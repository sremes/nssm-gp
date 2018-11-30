#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to initialize spectral kernels

@author: sremes
"""

import gc
import tempfile

from .spectral_svgp import SpectralSVGP
from .neural import NeuralSpectralKernel

import tensorflow as tf
import numpy as np

import gpflow
from gpflow.models import SVGP

from gpflow import settings
float_type = settings.dtypes.float_type

from scipy.spatial.distance import pdist


#
# Some helper functions.
#


def random_Z(x, N, M):
    inducing_idx = np.random.randint(N, size=M)
    Z = x[inducing_idx, :].copy()
    Z += 1e-3 * np.random.randn(*Z.shape)
    Z = np.sort(Z, 0)
    return Z


#
# Initializers for GP-GSM kernels.
#


def _gsm_rand_params(M, Q, input_dim, max_freq=1.0, max_len=1.0, max_var=1.0, ell=1.0, ARD=False):
    """
    Make random init for a GSM kernel with Q components. Use M inducing points.
    """
    def softplus_inverse(x):
        return np.log(np.exp(x) - 1.)

    def logistic_inverse(x):
        return np.log(x / (max_freq - x))

    #variances = [softplus_inverse(np.random.rand()*max_var*np.ones((M, 1), dtype=float_type)) for _ in range(Q)]
    #frequencies = [logistic_inverse(np.random.rand()*max_freq*np.ones((M, input_dim), dtype=float_type))
    #               for _ in range(Q)]
    variances = [np.random.randn(M, 1).astype(float_type) for _ in range(Q)]
    frequencies = [np.random.randn(M, input_dim).astype(float_type) for _ in range(Q)]
    ell_shape = (M, input_dim) if ARD else (M, 1)
    #lengthscales = [softplus_inverse(np.random.rand()*max_len*np.ones(ell_shape, dtype=float_type)) for _ in range(Q)]
    lengthscales = [np.random.randn(*ell_shape).astype(float_type) for _ in range(Q)]

    var_scale = 1/Q  # each kernel component contributes 1/Q on average
    Kvar = gpflow.kernels.RBF(input_dim=input_dim, lengthscales=ell, variance=var_scale, name='kvar_rbf')
    #Kvar += gpflow.kernels.White(input_dim=input_dim, variance=1e-6, name='kvar_white')
    Kvar.trainable = False
    Kfreq = gpflow.kernels.RBF(input_dim=input_dim, lengthscales=ell, name='kfreq_rbf')
    #Kfreq += gpflow.kernels.White(input_dim=input_dim, variance=1e-6, name='kfreq_white')
    Kfreq.trainable = False
    Klen = gpflow.kernels.RBF(input_dim=input_dim, lengthscales=ell, name='klen_rbf')
    #Klen += gpflow.kernels.White(input_dim=input_dim, variance=1e-6, name='klen_white')
    Klen.trainable = False
    return {'Kvar': Kvar, 'Kfreq': Kfreq, 'Klen': Klen,
            'variances': variances, 'frequencies': frequencies,
            'lengthscales': lengthscales}


def init_gsm(x, y, M, Q, max_freq=1.0, max_len=1.0, ell=1.0, n_inits=10,
             minibatch_size=256, noise_var=10.0, ARD=False, likelihood=None):
    print('Initializing GSM...')
    best_loglik = -np.inf
    best_m = None
    N, input_dim = x.shape
    for k in range(n_inits):
        print('init:', k)
        try:
            #gpflow.reset_default_graph_and_session()
            with gpflow.defer_build():
                Z = random_Z(x, N, M)
                p = _gsm_rand_params(M, Q, input_dim, max_freq=max_freq, max_len=max_len, ell=ell, ARD=ARD)
                if likelihood is not None:
                    likhood = likelihood
                else:
                    likhood = gpflow.likelihoods.Gaussian(noise_var)
                    likhood.variance.prior = gpflow.priors.LogNormal(mu=0, var=1)
                spectral = SpectralSVGP(X=x, Y=y, Z=Z, ARD=ARD, likelihood=likhood,
                                        max_freq=max_freq,
                                        minibatch_size=minibatch_size,
                                        variances=p['variances'],
                                        frequencies=p['frequencies'],
                                        lengthscales=p['lengthscales'],
                                        Kvar=p['Kvar'], Kfreq=p['Kfreq'],
                                        Klen=p['Klen'])
                spectral.feature.Z.prior = gpflow.priors.Gaussian(0, 1)
            spectral.compile()
            loglik = spectral.compute_log_likelihood()
            print('loglik:', loglik)
            if loglik > best_loglik:
                best_loglik = loglik
                best_m = spectral
                #best_dir = tempfile.TemporaryDirectory()
                #gpflow.saver.Saver().save(best_dir.name + 'model.gpflow', best_m)
            del spectral
            gc.collect()
        except tf.errors.InvalidArgumentError:  # cholesky may fail sometimes
            pass
    print('Best initialization: %f' % best_loglik)
    print(best_m)
    #gpflow.reset_default_graph_and_session()
    #best_m = gpflow.saver.Saver().load(best_dir.name + 'model.gpflow')
    #best_m.compile()
    #print(best_m)
    return best_m


#
# Initializer for Neural Spectral kernel
#

def init_neural(x, y, M, Q, n_inits=1, minibatch_size=256, noise_var=0.1, likelihood=None, hidden_sizes=None):
    print('Initializing neural spectral kernel...')
    best_loglik = -np.inf
    best_m = None
    N, input_dim = x.shape
    for k in range(n_inits):
        try:
            # gpflow.reset_default_graph_and_session()
            with gpflow.defer_build():
                Z = random_Z(x, N, M)
                k = NeuralSpectralKernel(input_dim=input_dim, Q=Q, hidden_sizes=hidden_sizes)
                if likelihood is not None:
                    likhood = likelihood
                else:
                    likhood = gpflow.likelihoods.Gaussian(noise_var)
                    likhood.variance.prior = gpflow.priors.LogNormal(mu=0, var=1)
                model = SVGP(X=x, Y=y, Z=Z, kern=k, likelihood=likhood,
                             minibatch_size=minibatch_size)
                model.feature.Z.prior = gpflow.priors.Gaussian(0, 1)
            model.compile()
            loglik = model.compute_log_likelihood()
            if loglik > best_loglik:
                best_loglik = loglik
                best_m = model
                # best_dir = tempfile.TemporaryDirectory()
                # gpflow.saver.Saver().save(best_dir.name + 'model.gpflow', best_m)
            del model
            gc.collect()
        except tf.errors.InvalidArgumentError:  # cholesky fails sometimes (with really bad init?)
            pass
    print('Best init: %f' % best_loglik)
    print(best_m)
    # gpflow.reset_default_graph_and_session()
    # best_m = gpflow.saver.Saver().load(best_dir.name + 'model.gpflow')
    # best_m.compile()
    # print(best_m)
    return best_m

#
# Initializers for other spectral kernels.
#

def init_spectral(x, y, M, Q, kern, n_inits=10, minibatch_size=256, noise_var=10.0, ARD=True, likelihood=None):
    print('Initializing a spectral kernel...')
    best_loglik = -np.inf
    best_m = None
    N, input_dim = x.shape
    for k in range(n_inits):
        try:
            #gpflow.reset_default_graph_and_session()
            with gpflow.defer_build():
                Z = random_Z(x, N, M)
                dists = pdist(Z, 'euclidean').ravel()
                max_freq = min(10.0, 1./np.min(dists[dists > 0.0]))
                max_len = min(5.0, np.max(dists) * (2*np.pi))
                k = kern(input_dim=input_dim, max_freq=max_freq, Q=Q, ARD=ARD, max_len=max_len)
                if likelihood is not None:
                    likhood = likelihood
                else:
                    likhood = gpflow.likelihoods.Gaussian(noise_var)
                    likhood.variance.prior = gpflow.priors.LogNormal(mu=0, var=1)
                model = SVGP(X=x, Y=y, Z=Z, kern=k, likelihood=likhood,
                             minibatch_size=minibatch_size)
                model.feature.Z.prior = gpflow.priors.Gaussian(0, 1)
            model.compile()
            loglik = model.compute_log_likelihood()
            if loglik > best_loglik:
                best_loglik = loglik
                best_m = model
                #best_dir = tempfile.TemporaryDirectory()
                #gpflow.saver.Saver().save(best_dir.name + 'model.gpflow', best_m)
            del model
            gc.collect()
        except tf.errors.InvalidArgumentError:  # cholesky fails sometimes (with really bad init?)
            pass
    print('Best init: %f' % best_loglik)
    print(best_m)
    #gpflow.reset_default_graph_and_session()
    #best_m = gpflow.saver.Saver().load(best_dir.name + 'model.gpflow')
    #best_m.compile()
    #print(best_m)
    return best_m
