__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np


_bbox_sign_factors = np.asarray(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
        ], dtype='f4')


def convert_rectangle_to_eight_point(bboxes):
    pt_set = np.zeros((len(bboxes), 8, 2))

    pt_set[:] = bboxes['center_xy'][:, None, :]
    for i, bbox in enumerate(bboxes):
        s_phi_offset, c_phi_offset = np.sin(bbox['orientation']), np.cos(bbox['orientation'])
        rot = np.array([[c_phi_offset, - s_phi_offset], [s_phi_offset, c_phi_offset]])
        offset_xy = np.dot(_bbox_sign_factors * 0.5 * bbox['dimension'], rot.T)
        pt_set[i, :, :] += offset_xy

    return pt_set
