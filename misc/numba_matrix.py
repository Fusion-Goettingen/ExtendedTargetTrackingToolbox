__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
from numba import guvectorize
from numba import float32, float64


@guvectorize([(float64[:], float64[:, :], float64[:]),
              (float32[:], float32[:, :], float32[:])],
             '(n), (n, n) -> ()',
             nopython=True, target='cpu')
def mhd_guv(d, m, out):
    out[:] = np.dot(d, np.dot(m, d))


@guvectorize([(float64[:, :], float64[:, :], float64[:, :]),
              (float32[:, :], float32[:, :], float32[:, :])],
             '(n, m), (m, k) -> (n, k)',
             nopython=True, target='cpu')
def mm_dot_guv(a, b, out):
    out[:] = np.dot(a, b)


@guvectorize([(float64[:, :], float64[:], float64[:]),
              (float32[:, :], float32[:], float32[:])],
             '(n, m), (m) -> (n)',
             nopython=True, target='cpu')
def vv_dot_guv(a, b, out):
    out[:] = np.dot(a, b)


@guvectorize([(float64[:, :], float64[:, :], float64[:]),
              (float32[:, :], float32[:, :], float32[:])],
             '(n, n), (n, n) -> ()',
             nopython=True, target='cpu')
def mm_dot_trace_guv(a, b, out):
    out[:] = np.trace(np.dot(a, b))


@guvectorize([(float64[:], float64[:, :]),
              (float32[:], float32[:, :])],
             '(n) -> (n, n)',
             nopython=True, target='cpu')
def v_outer_guv(a, out):
    out[:] = np.outer(a, a)


@guvectorize([(float64[:], float64[:], float64[:, :]),
              (float32[:], float32[:], float32[:, :])],
             '(n), (m) -> (n, m)',
             nopython=True, target='cpu')
def vv_outer_guv(a, b, out):
    # out[:] = np.outer(a, b)
    for i in range(len(a)):
        for j in range(len(b)):
            out[i, j] = a[i] * b[j]


@guvectorize([(float64[:], float64[:], float64[:]),
              (float32[:], float32[:], float32[:])],
             '(n), (n) -> ()',
             nopython=True, target='cpu')
def cross_2d_guv(a, b, out):
    out[0] = a[0] * b[1] - a[1] * b[0]


@guvectorize([(float64[:], float64[:, :]),
              (float32[:], float32[:, :])],
             '(n) -> (n, n)',
             nopython=True, target='cpu')
def diag_guv(a, out):
    out[:] = np.zeros((len(a), len(a)), dtype=a.dtype)
    for i in range(len(a)):
        out[i, i] = a[i]


@guvectorize([(float64[:, :], float64[:, :], float64[:, :]),
              (float32[:, :], float32[:, :] , float32[:, :])],
             '(n, n), (m, m) -> (n, n)',
             nopython=True, target='cpu')
def add_block_diag_guv(a, b, out):
    out[:] = np.zeros(a.shape, dtype=a.dtype)
    out[:] = a[:]

    dim_b = len(b)
    mult_factor = len(a) // dim_b

    for i in range(mult_factor):
        offset = dim_b * i
        for j in range(dim_b):
            for k in range(dim_b):
                out[offset + j, offset + k] += b[j, k]
