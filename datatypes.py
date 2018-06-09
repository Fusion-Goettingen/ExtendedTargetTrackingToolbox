__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
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
