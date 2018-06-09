__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
from numpy.linalg import cholesky


def ut(m, c, lmbda, m_a):
    """
    unscented transform

    expects lambda to be pre-calculated
    and weights calculated outside the transformation.

    Implicit return via m_a

    Parameters
    ----------
    m : array_like
        (D) encoding the mean
    c : array_like
        (D x D) array encoding the covariance
    lmbda : lmbda parameter
    m_a : array_like, return
        (2D + 1 x D) augmented state

    Returns
    -------

    """

    n_x = m.shape[-1]

    cov_sqrt = cholesky((n_x + lmbda) * c).T
    m_a[:] = m
    m_a[1:n_x + 1] += cov_sqrt
    m_a[1 + n_x:] -= cov_sqrt


def batch_ut(m, c, lmbda, m_a):
    """
    batch version for the unscented transform

    expects lambda to be pre-calculated
    and weights calculated outside the transformation.

    Implicit return via m_a

    Parameters
    ----------
    m : array_like
        (N x D) encoding the mean
    c : array_like
        (N x D x D) array encoding the covariance
    lmbda : lmbda parameter
    m_a : array_like, return
        (N x 2D + 1 x D) augmented state

    Returns
    -------

    """
    n_x = m.shape[-1]

    cov_sqrt = np.transpose(cholesky((n_x + lmbda) * c), axes=(0, 2, 1))
    m_a[:] = m[:, None, :]
    m_a[:, 1:n_x + 1] += cov_sqrt
    m_a[:, 1 + n_x:] -= cov_sqrt


def batch_ut_2d(m, c, lmbda, m_a):
    """
    batch version for the unscented transform

    expects lambda to be pre-calculated
    and weights calculated outside the transformation.

    Implicit return via m_a

    Parameters
    ----------
    m : array_like
        (N x M x D) encoding the mean
    c : array_like
        (N x M x D x D) array encoding the covariance
    lmbda : lmbda parameter
    m_a : array_like, return
        (N x M x 2D + 1 x D) augmented state

    Returns
    -------

    """
    n_x = m.shape[-1]

    cov_sqrt = np.transpose(cholesky((n_x + lmbda) * c), axes=(0, 1, 3, 2))
    m_a[:] = m[:, :, None, :]
    m_a[:, :, 1:n_x + 1] += cov_sqrt
    m_a[:, :, 1 + n_x:] -= cov_sqrt


