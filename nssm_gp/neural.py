import tensorflow as tf
import numpy as np

import gpflow
from gpflow import Param, ParamList

from gpflow import settings
float_type = settings.dtypes.float_type


def _create_params(input_dim, output_dim):
    def initializer():
        limit = np.sqrt(6. / (input_dim + output_dim))
        return np.random.uniform(-limit, +limit, (input_dim, output_dim))
    return Param(initializer(), dtype=float_type, prior=gpflow.priors.Gaussian(0, 1)), \
           Param(np.zeros(output_dim), dtype=float_type)


def robust_kernel(kern, shape_X):
    eigvals = tf.self_adjoint_eigvals(kern)
    min_eig = tf.reduce_min(eigvals)
    jitter = settings.numerics.jitter_level

    def abs_min_eig():
        return tf.Print(tf.abs(min_eig), [min_eig], 'kernel had negative eigenvalue')

    def zero():
        return float_type(0.0)

    jitter += tf.cond(tf.less(min_eig, 0.0), abs_min_eig, zero)
    return kern + jitter * tf.eye(shape_X, dtype=settings.dtypes.float_type)


class NeuralSpectralKernel(gpflow.kernels.Kernel):
    def __init__(self, input_dim, active_dims=None, Q=1, hidden_sizes=None):
        super().__init__(input_dim, active_dims=active_dims)
        self.Q = Q
        if hidden_sizes is None:
            hidden_sizes = (32, 32)
        self.num_hidden = len(hidden_sizes)
        for v, final_size in zip(['freq', 'len', 'var'], [input_dim, input_dim, 1]):
            self._create_nn_params(v, hidden_sizes, final_size)

    def _create_nn_params(self, prefix, hidden_sizes, final_size):
        for q in range(self.Q):
            input_dim = self.input_dim
            for level, hidden_size in enumerate(hidden_sizes):
                """name_W = '{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level)
                name_b = '{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level)
                params = _create_params(input_dim, hidden_size)
                setattr(self, name_W, params[0])
                setattr(self, name_b, params[1])"""
                name_W = '{prefix}_W_{level}'.format(prefix=prefix, level=level)
                name_b = '{prefix}_b_{level}'.format(prefix=prefix, level=level)
                if not hasattr(self, name_W):
                    params = _create_params(input_dim, hidden_size)
                    setattr(self, name_W, params[0])
                    setattr(self, name_b, params[1])
                # input dim for next layer
                input_dim = hidden_size
            params = _create_params(input_dim, final_size)
            setattr(self, '{prefix}_{q}_W_final'.format(prefix=prefix, q=q), params[0])
            setattr(self, '{prefix}_{q}_b_final'.format(prefix=prefix, q=q), params[1])

    @gpflow.params_as_tensors
    def _nn_function(self, x, prefix, q, dropout=0.8, final_activation=tf.nn.softplus):
        for level in range(self.num_hidden):
            """W = getattr(self, '{prefix}_{q}_W_{level}'.format(prefix=prefix, q=q, level=level))
            b = getattr(self, '{prefix}_{q}_b_{level}'.format(prefix=prefix, q=q, level=level))"""
            W = getattr(self, '{prefix}_W_{level}'.format(prefix=prefix, level=level))
            b = getattr(self, '{prefix}_b_{level}'.format(prefix=prefix, level=level))
            x = tf.nn.selu(tf.nn.xw_plus_b(x, W, b))  # self-normalizing neural network
            # if dropout < 1.0:
            #     x = tf.contrib.nn.alpha_dropout(x, keep_prob=dropout)
        W = getattr(self, '{prefix}_{q}_W_final'.format(prefix=prefix, q=q))
        b = getattr(self, '{prefix}_{q}_b_final'.format(prefix=prefix, q=q))
        return final_activation(tf.nn.xw_plus_b(x, W, b))

    @gpflow.params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        kern = 0.0
        for q in range(self.Q):
            # compute latent function values by the neural network
            freq, freq2 = self._nn_function(X, 'freq', q), self._nn_function(X2, 'freq', q)
            lens, lens2 = self._nn_function(X, 'len', q), self._nn_function(X2, 'len', q)
            var, var2 = self._nn_function(X, 'var', q), self._nn_function(X2, 'var', q)

            # compute length-scale term
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

            # compute cosine term
            muX = (tf.reduce_sum(freq * X, 1, keepdims=True)
                   - tf.transpose(tf.reduce_sum(freq2 * X2, 1, keepdims=True)))
            COS = tf.cos(2 * np.pi * muX)

            # compute kernel variance term
            WW = tf.matmul(var, var2, transpose_b=True)  # w*w'^T

            # compute the q'th kernel component
            kern += WW * E * COS
        if X == X2:
            return robust_kernel(kern, tf.shape(X)[0])
        else:
            return kern

    @gpflow.params_as_tensors
    def Kdiag(self, X):
        kd = settings.jitter
        for q in range(self.Q):
            kd += tf.square(self._nn_function(X, 'var', q))
        return tf.squeeze(kd)



