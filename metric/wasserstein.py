__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from sklearn.utils.linear_assignment_ import linear_assignment


def point_set_wasserstein_distance(x, y, c=100.0, p=1.0):
    """
    Wasserstein Distance for point sets

    Parameters
    ----------
    x : Set 1
    y : Set 2
    c : c parameter, i.e. penalty factor for cardinality mismatch
    p : Minkovski metric parameter

    Returns
    -------

    """

    if len(x) < 1 or len(y) < 1:
        if len(x) == len(y):
            return .0
        else:
            raise AssertionError('Point sets need to be the same')

    dt = np.minimum(cdist(x, y), c)
    indices = np.transpose(linear_assignment(dt))
    reg_d = dt[indices[0], indices[1]]

    if np.isclose(p, 1.0):
        log_wsd = logsumexp(p * np.log(reg_d))
        log_wsd -= np.log(max(len(x), len(y)))
        log_wsd /= p
        wsd = np.exp(log_wsd)
    else:
        wsd = sum(reg_d)
        wsd /= max(len(x), len(y))

    return wsd
