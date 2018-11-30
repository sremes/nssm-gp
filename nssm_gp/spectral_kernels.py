"""
Implements the following spectral kernels:
 (1) Spectral Mixture (SM) by Wilson (2013) [stationary]
 (2) Bivariate Spectral Mixture (BSM) in Remes (2017) [non-stationary, not well-tested]
"""
import tensorflow as tf
import numpy as np

import gpflow
from gpflow.kernels import Kernel, Stationary, Sum, Product  # used to derive new kernels
from gpflow import Param
from gpflow import transforms

from gpflow import settings
float_type = settings.dtypes.float_type


def square_dist(X, X2):
    Xs = tf.reduce_sum(tf.square(X), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return -2 * tf.matmul(X, X2, transpose_b=True) + tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))    
    

class SMKernelComponent(Stationary):
    """
    Spectral Mixture kernel.
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None, 
                 frequency=1.0, active_dims=None, ARD=False):
        Stationary.__init__(self, input_dim=input_dim, variance=variance, lengthscales=lengthscales,
                            active_dims=active_dims, ARD=ARD)
        self.frequency = Param(frequency, transforms.positive, dtype=float_type)
        self.frequency.prior = gpflow.priors.Exponential(1.0)
        self.variance.prior = gpflow.priors.LogNormal(0, 1)

    @gpflow.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)  # 1 x 1 x D
        r = tf.reduce_sum(2.0 * np.pi * freq * (f - f2), 2)
        return self.variance * tf.exp(-2.0*np.pi**2*self.scaled_square_dist(X, X2)) * tf.cos(r)
    
    @gpflow.params_as_tensors
    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    @gpflow.params_as_tensors
    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist


def SMKernel(Q, input_dim, active_dims=None, variances=None, frequencies=None,
             lengthscales=None, max_freq=1.0, max_len=1.0, ARD=False):
    """
    Initialises a SM kernel with Q components. Optionally uses a given initialisation,
    otherwise uses a random initialisation.

    max_freq: Nyquist frequency of the signal, used to initialize frequencies
    max_len: range of the inputs x, used to initialize length-scales
    """
    if variances is None:
        variances = [1./Q for _ in range(Q)]
    if frequencies is None:
        frequencies = [np.random.rand(input_dim)*max_freq for _ in range(Q)]
    if lengthscales is None:
        lengthscales = [np.abs(max_len*np.random.randn(input_dim if ARD else 1)) for _ in range(Q)]
    kerns = [SMKernelComponent(input_dim, active_dims=active_dims, variance=variances[i], 
                               frequency=frequencies[i], lengthscales=lengthscales[i], ARD=ARD)
             for i in range(Q)]
    return Sum(kerns)


class BSMKernelComponent(Kernel):
    """
    Bi-variate Spectral Mixture Kernel.
    """
    def __init__(self, input_dim, variance=1.0, frequency=np.array([1.0, 1.0]),
                 lengthscale=1.0, correlation=0.0, max_freq=1.0, active_dims=None):
        assert(input_dim == 1)  # the derivations are valid only for one dimensional input
        Kernel.__init__(self, input_dim=input_dim, active_dims=active_dims)
        self.variance = Param(variance, transforms.positive)
        self.frequency = Param(frequency, transforms.Logistic(0.0, max_freq))
        self.lengthscale = Param(lengthscale, transforms.positive)
        correlation = np.clip(correlation, 1e-4, 1-1e-4)  # clip for numerical reasons
        self.correlation = Param(correlation, transforms.Logistic())
    
    @gpflow.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        pi = np.pi
        # exp term; x^T * [1 rho; rho 1] * x, x=[x,-x']^T
        XX, XX2 = tf.meshgrid(X, X2, indexing='ij')
        R = tf.square(XX) + tf.square(XX2) - 2.0*self.correlation*XX*XX2
        exp_term = tf.exp(-2.0 * pi**2 * tf.square(self.lengthscale) * R)
        
        # phi cosine terms
        mu = self.frequency
        phi1 = tf.stack([tf.cos(2*pi*mu[0]*X) + tf.cos(2*pi*mu[1]*X),
                         tf.sin(2*pi*mu[0]*X) + tf.sin(2*pi*mu[1]*X)], axis=1)
        phi2 = tf.stack([tf.cos(2*pi*mu[0]*X2) + tf.cos(2*pi*mu[1]*X2),
                         tf.sin(2*pi*mu[0]*X2) + tf.sin(2*pi*mu[1]*X2)], axis=1)
        phi = tf.matmul(tf.squeeze(phi1), tf.squeeze(phi2), transpose_b=True)
        
        return self.variance * exp_term * phi
    
    @gpflow.params_as_tensors
    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


def BSMKernel(input_dim=1, active_dims=None, Q=1, max_freq=1.0):
    if active_dims is None:
        active_dims = [0]
    kerns = []
    for q in range(Q):
        var = 1.0 / Q
        mu_f = max_freq * np.random.rand(2).astype(float_type)  # two paired frequencies
        ell = np.random.rand()
        kerns.append(BSMKernelComponent(input_dim=input_dim, active_dims=active_dims, max_freq=max_freq,
                                        variance=var, frequency=mu_f, lengthscale=ell))
    return Sum(kerns)


def ProductBSMKernel(input_dim, Q=1, max_freq=1.0):
    kerns = []
    for dim in range(input_dim):
        kerns.append(BSMKernel(input_dim=1, active_dims=[dim], Q=Q, max_freq=max_freq))
    return Product(kerns)

