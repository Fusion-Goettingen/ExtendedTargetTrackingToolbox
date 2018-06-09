__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

from .convert import convert_rectangle_to_eight_point
from .numba_matrix import mhd_guv, mm_dot_guv, mm_dot_trace_guv, vv_dot_guv, v_outer_guv, vv_outer_guv, cross_2d_guv, \
    diag_guv, add_block_diag_guv
from .unscented import ut, batch_ut, batch_ut_2d
