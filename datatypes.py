__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, JH private tracking toolbox"
__email__ = "-"
__license__ = "-"
__version__ = "0.1"
__status__ = "Prototype"

import numpy as np

dt_bbox = np.dtype([
    ('ts', 'i4'),
    ('center_xy', 'f4', (2,)),
    ('orientation', 'f4'),
    ('dimension', 'f4', (2,)),
])

dt_z = np.dtype([('ts', 'i4'),
                 ('xy', 'f8', (2,))])
