__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
from datatypes import dt_bbox


class SingleTargetTracker(object):
    """
    Tracker base class.

    Features the basic steps : step, prediction, correction, reduction, extraction.

    Attributes
    ----------
    _uc             update count
    _epsilon        numerical constant
    _min_log_prob   numerical constant to limit logarithmic probability
    _dt             step time difference
    _steps          number of steps to be processed
    _z_flat         flat measurement list
    _log_lik        log likelihood matrix
    _dt_bbox        dtype for the bounding box extraction

    """

    def __init__(self, **kwargs):

        self._uc = 0

        self._epsilon = kwargs.get('epsilon', 1e-9)
        self._min_log_prob = kwargs.get('min_log_prob', -700.0)

        self._dt = kwargs.get('dt')

        self._steps = kwargs.get('steps')
        self._z_flat = np.zeros(0, dtype=np.dtype([]))

        self._log_lik = self._min_log_prob * np.ones(self._steps, dtype='f8')

        self._dt_bbox = dt_bbox

    def step(self, z):
        self._uc += 1

        self.predict()
        self.correct(z)
        self.reduce()

    def predict(self):
        pass

    def correct(self, z):
        raise NotImplementedError

    def reduce(self):
        pass

    def extract(self):
        raise NotImplementedError
