__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, JH private tracking toolbox"
__email__ = "-"
__license__ = "-"
__version__ = "0.1"
__status__ = "Prototype"

from .ggiw import GgiwTracker, plot_ggiw
from .scgp import ScGpTracker, DecorrelatedScGpTracker, plot_gp
from .spline import EkfSplineTracker, RigidEkfSplineTracker, UkfSplineTracker, plot_spline
