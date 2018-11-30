# -*- coding: utf-8 -*-
"""
Implements the Generalized Spectral Mixture (GP-GSM) from Remes (2017).

@author: sremes
"""

import tensorflow as tf
import numpy as np

import gpflow
from gpflow import Param, ParamList
from gpflow.conditionals import conditional

from gpflow import settings
float_type = settings.dtypes.float_type


def square_dist(X, X2):
    Xs = tf.reduce_sum(tf.square(X), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return (-2 * tf.matmul(X, X2, transpose_b=True)
            + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1)))


def _check_eigvals(kern, name):
    eigvals = tf.self_adjoint_eigvals(kern)

    def print_op():
        return tf.Print(kern, [eigvals], message='negative eigenvalues %s' % name)

    def identity_op():
        return tf.identity(kern)

    return tf.cond(tf.less(tf.reduce_min(eigvals), 0.0), true_fn=print_op, false_fn=identity_op)


class SpectralSVGP(gpflow.models.SVGP):
    """ Implements SVGP with the non-stationary GSM kernel. """
    def __init__(self, X, Y, likelihood,
                 mean_function=None,
                 q_diag=False, whiten=True,
                 minibatch_size=None, Z=None, ARD=False,
                 variances=None, frequencies=None, lengthscales=None,
                 Kvar=None, Kfreq=None, Klen=None, max_freq=float_type(100.0),
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        super().__init__(X, Y, None, likelihood=likelihood,
                         feat=None, mean_function=mean_function,
                         num_latent=1, q_diag=q_diag,
                         whiten=whiten, num_data=None, Z=Z,
                         minibatch_size=minibatch_size, **kwargs)

        # Check that all kernel variables are same length
        assert(len(variances) == len(frequencies)
               and len(frequencies) == len(lengthscales))
        # Add kernel variables directly, as kernel is implemented here
        def to_param_list(var_list, name):
            param_list = []
            for idx, var in enumerate(var_list):
                name_idx = '{name}_{idx}'.format(name=name, idx=idx)
                param_list.append(Param(var, dtype=float_type, name=name_idx))
            return ParamList(param_list)
        self.variances = to_param_list(variances, 'variances')
        self.frequencies = to_param_list(frequencies, 'frequencies')
        self.lengthscales = to_param_list(lengthscales, 'lengthscales')
        self.ARD = ARD
        self.Kvar, self.Kfreq, self.Klen = Kvar, Kfreq, Klen
        self.num_inducing, self.num_inputs = Z.shape
        self.max_freq = max_freq

    def logistic(self, x):
        x = x - 3.0  # translate x to promote lower frequencies
        return self.max_freq / (1. + tf.exp(-x))

    @gpflow.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        """ Implement kernel here, so we can access inducing features. """
        if X2 is None:
            X2 = X
        kern = 0.0
        for frequencies, variances, lengthscales in zip(self.frequencies,
                                                        self.variances,
                                                        self.lengthscales):
            # Interpolate parameters from inducing points
            def interpolator(f, K, transform):
                def g(x):
                    return self._interpolate(x, f, K, transform=transform)
                return g
            freq, freq2 = map(interpolator(frequencies, self.Kfreq, transform=self.logistic), [X, X2])
            #var, var2 = map(interpolator(variances, self.Kvar, transform=tf.nn.softplus), [X, X2])
            var, var2 = map(interpolator(variances, self.Kvar, transform=tf.nn.softplus), [X, X2])
            lens, lens2 = map(interpolator(lengthscales, self.Klen, transform=tf.nn.softplus), [X, X2])

            # add some jitter to strictly positive variables \ell and \w
            eps = 1e-4
            lens += eps
            lens2 += eps
            var += eps
            var2 += eps
            # compute kernel variance term
            WW = tf.matmul(var, var2, transpose_b=True)  # w*w'^T
            # compute length-scale term
            if not self.ARD:
                ll = tf.matmul(lens, lens2, transpose_b=True)  # l*l'^T
                # l^2*1^T + 1*(l'^2)^T:
                ll2 = tf.square(lens) + tf.transpose(tf.square(lens2))
                D = square_dist(X, X2)
                E = tf.sqrt(2 * ll / ll2) * tf.exp(-D/ll2)
            else:
                # E_ij := sqrt(2*sqrt(|S_i||S_j|)/|S_i+S_j|)*exp(-1/2*(x_i - x_j)^T ((S_i+S_j)/2)^-1 (x_i - x_j))
                # S_i = diag(\ell(x_i)^2)
                Xr = tf.expand_dims(X, 1)  # N1 1 D
                X2r = tf.expand_dims(X2, 0)  # 1 N2 D
                l1 = tf.expand_dims(lens, 1)  # N1 1 D
                l2 = tf.expand_dims(lens2, 0)  # 1 N2 D
                L = tf.square(l1) + tf.square(l2)  # N1 N2 D
                #D = tf.square((Xr - X2r) / L)  # N1 N2 D
                D = tf.square(Xr - X2r) / L  # N1 N2 D
                D = tf.reduce_sum(D, 2)  # N1 N2
                det = tf.sqrt(2 * l1 * l2 / L)  # N1 N2 D
                det = tf.reduce_prod(det, 2)  # N1 N2
                E = det * tf.exp(-D)  # N1 N2
            # compute periodic term
            muX = (tf.reduce_sum(freq*X, 1, keepdims=True)
                   - tf.transpose(tf.reduce_sum(freq2*X2, 1, keepdims=True)))
            COS = tf.cos(2*np.pi*muX)
            # Compute the kernel
            kern += WW*E*COS
        if X == X2:
            eigvals = tf.self_adjoint_eigvals(kern)
            min_eig = tf.reduce_min(eigvals)
            jitter = settings.numerics.jitter_level
            def abs_min_eig():
                return tf.Print(tf.abs(min_eig), [min_eig], 'kernel had negative eigenvalue!')
            def zero():
                return float_type(0.0)
            jitter += tf.cond(tf.less(min_eig, 0.0), abs_min_eig, zero)
            return kern + jitter * tf.eye(tf.shape(X)[0], dtype=settings.dtypes.float_type)
        else:
            return kern

    @gpflow.params_as_tensors
    def Kdiag(self, X, presliced=False):
        kdiag = 1e-4  # some jitter
        for variances in self.variances:
            kdiag += tf.square(self._interpolate(X, variances, self.Kvar, transform=tf.nn.softplus))
        with tf.control_dependencies([tf.assert_positive(kdiag, message='Kdiag negative: ')]):
            # kdiag = tf.Print(kdiag, [kdiag], 'kdiag: ')
            kdiag = tf.identity(kdiag)
        return tf.squeeze(kdiag)

    @gpflow.params_as_tensors
    def _interpolate(self, X, f, kern, transform=tf.identity):
        """ Compute parameter values at X. """
        mu, _ = conditional(X, self.feature, kern, f, white=True)
        return transform(mu)

    @gpflow.autoflow((settings.float_type, [None, None]),
                     (settings.float_type, [None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)

    @gpflow.autoflow((settings.float_type, [None, None]))
    @gpflow.params_as_tensors
    def compute_variances(self, Xnew):
        return [self._interpolate(Xnew, variances, self.Kvar, transform=tf.nn.softplus)
                for variances in self.variances]

    @gpflow.autoflow((settings.float_type, [None, None]))
    @gpflow.params_as_tensors
    def compute_frequencies(self, Xnew):
        return [self._interpolate(Xnew, frequencies, self.Kfreq, transform=self.logistic)
                for frequencies in self.frequencies]

    @gpflow.autoflow((settings.float_type, [None, None]))
    @gpflow.params_as_tensors
    def compute_lengthscales(self, Xnew):
        return [self._interpolate(Xnew, lengthscales, self.Klen, transform=tf.nn.softplus)
                for lengthscales in self.lengthscales]

    @gpflow.params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = self.feature.Kuu(self, jitter=settings.numerics.jitter_level)
        return gpflow.kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        Adds also the prior term for the GP parameters.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False)

        #with tf.control_dependencies([tf.assert_positive(fvar, message='fvar negative: ')]):
        #    # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale var_exp for minibatch size
        scale = (tf.cast(self.num_data, settings.tf_float)
                 / tf.cast(tf.shape(self.X)[0], settings.tf_float))
        var_exp = tf.reduce_sum(var_exp) * scale

        # latent functions have a whitened GP prior
        prior = float_type(0.0)
        for vars in zip(self.frequencies, self.variances, self.lengthscales):
            prior += -0.5 * sum(tf.reduce_sum(tf.square(x)) for x in vars)

        # re-scale prior for inducing point size
        # scale = tf.cast(self.num_data, settings.tf_float) / tf.cast(self.num_inducing, settings.tf_float)
        # prior = prior * scale

        # print tensors
        #var_exp = tf.Print(var_exp, [var_exp], message='var_exp:')
        #KL = tf.Print(KL, [KL], message='KL:')
        #prior = tf.Print(prior, [prior], message='prior:')
        likelihood = var_exp - KL + prior
        likelihood = tf.Print(likelihood, [likelihood], 'likelihood')
        return likelihood

    @gpflow.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        # register Kernel implementations for SpectralSVGP
        from gpflow import name_scope
        from gpflow.dispatch import dispatch

        @conditional.register(object, type(self.feature), type(self), object)
        @name_scope("conditional")
        def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
            # find correct function signature from the dispatcher
            cond = conditional.dispatch(object, type(self.feature), gpflow.kernels.Kernel, object)
            return cond(Xnew, feat, kern, f, full_cov=full_cov, full_output_cov=full_output_cov, q_sqrt=q_sqrt, white=white)

        @dispatch(type(self.feature), type(self))
        def Kuu(feat, kern, *, jitter=0.0):
            with gpflow.decors.params_as_tensors_for(feat):
                Kzz = kern.K(feat.Z)
                Kzz += jitter * tf.eye(len(feat), dtype=settings.dtypes.float_type)
            return Kzz

        @dispatch(type(self.feature), type(self), object)
        def Kuf(feat, kern, Xnew):
            with gpflow.decors.params_as_tensors_for(feat):
                Kzx = kern.K(feat.Z, Xnew)
            return Kzx

        mu, var = conditional(Xnew, self.feature, self, self.q_mu, q_sqrt=self.q_sqrt,
                              full_cov=full_cov, white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

