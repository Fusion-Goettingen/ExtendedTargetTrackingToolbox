__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, JH private tracking toolbox"
__email__ = "-"
__license__ = "-"
__version__ = "0.1"
__status__ = "Prototype"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from models.spline import map_idx
from datatypes import dt_bbox, dt_z


class SplineDataSimulator(object):
    """
    Simple data generation class

    """

    def __init__(self, **kwargs):

        self._uc = 0

        self._dt_bbox = dt_bbox

        self._dt = kwargs.get('dt')
        self._steps = kwargs.get('steps')

        self._d = kwargs.get('d')
        self._sd = kwargs.get('sd')

        self._m_basis = np.array([[0.5, -1.0, 0.5],
                                  [-1.0, 1.0, 0.0],
                                  [0.5, 0.5, 0.0]], dtype='f4')
        self._p_basis_unscaled = kwargs.get('p_basis')
        self._p_basis_dim = len(self._p_basis_unscaled)

        self._sel_idx_full = np.arange(0, self._p_basis_dim + 3, dtype='i4') % self._p_basis_dim

        self._dt_gt = np.dtype([('m', 'f8', self._sd)])
        self._dt_z = dt_z

        self._gt_state = np.zeros(self._steps, dtype=self._dt_gt)
        self._measurements = np.zeros(0, dtype=self._dt_z)

        self._gt_state['m'][0] = kwargs.get('init_m')

        self._r = kwargs.get('r')

        self._poisson_scat_mean = kwargs.get('mean_scat_number')

    def _transition_f(self, u):
        """
        transition function for state vector [x, y, phi, v, omega]

        time difference is implicit

        Parameters
        ----------
        u       array_like
            sigma points

        """

        u[0] += self._dt * u[3] * np.cos(u[2])
        u[1] += self._dt * u[3] * np.sin(u[2])

        u[2] += self._dt * u[4]

    def predict(self):
        self._gt_state['m'][self._uc] = self._gt_state['m'][self._uc - 1]
        self._transition_f(self._gt_state['m'][self._uc])

    def emit(self):
        """
        Calculates random points on the spline surface and stores them into self._data

        """

        # calculate rotation matrices
        s_phi_u, c_phi_u = np.sin(self._gt_state['m'][self._uc, 2]), np.cos(self._gt_state['m'][self._uc, 2])
        rot_u = np.asarray([[c_phi_u, -s_phi_u],
                            [s_phi_u, c_phi_u]])

        # Random number of measurements
        cd_z = np.random.poisson(lam=self._poisson_scat_mean)

        # Randomly select indices and tau
        ref_idx = np.random.randint(0, self._p_basis_dim, cd_z)
        sel_idx = np.zeros(ref_idx.shape + (3,), dtype='i8')
        map_idx(ref_idx, self._sel_idx_full, sel_idx)
        tau = np.random.rand(cd_z)

        mp_base_sel_bare = np.dot(self._m_basis, self._p_basis_unscaled[sel_idx])
        measurements_t = np.zeros(cd_z, dtype=self._dt_z)
        measurements_t['xy'] = \
            mp_base_sel_bare[2] + mp_base_sel_bare[1] * tau[:, None] + mp_base_sel_bare[0] * (tau ** 2)[:, None]
        measurements_t['xy'] = np.dot(measurements_t['xy'], rot_u.T)
        measurements_t['xy'] += self._gt_state['m'][self._uc, :2]
        measurements_t['ts'] = self._uc - 1

        self._measurements = np.concatenate((self._measurements, measurements_t))

    def step(self):
        self._uc += 1

        self.predict()
        self.emit()

    def extract(self):
        """

        Returns
        -------

        gt: array_like,
            structured array of the ground truth trajectory
        measurements: array_like,
            structured array of the measurements in cartesian coordinates, dtype [('ts', 'i4'), ('xy', 'f4', (2,))]

        """
        self._measurements['xy'] += np.random.multivariate_normal(np.zeros(2), self._r, len(self._measurements))
        gt_extr = np.zeros(self._gt_state.shape[0] - 1, dtype=np.dtype([('ts', 'i4')] + self._gt_state.dtype.descr))
        gt_extr['m'] = self._gt_state['m'][1:]
        gt_extr['ts'] = np.arange(len(gt_extr['ts']))
        return gt_extr, self._measurements

    def extrackt_bbox(self):
        """
        Bounding box extraction function.
        The algorithm creates the minimal bounding box defined by the spline knots.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of bounding boxes

        """

        bbox_extr = np.zeros(self._steps - 1, dtype=self._dt_bbox)
        bbox_extr['ts'] = np.arange(self._steps - 1, dtype='i4')
        bbox_extr['center_xy'] = self._gt_state['m'][1:, :2]
        bbox_extr['orientation'] = self._gt_state['m'][1:, 2]
        basis_dim = np.max(self._p_basis_unscaled, axis=0)
        bbox_extr['dimension'] = 2 * basis_dim

        return bbox_extr


if __name__ == "__main__":

    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    np.seterr('warn')

    steps = 100
    dt = 0.04

    config = {
        'steps': steps + 1,
        'd': 2,
        'sd': 5,
        'r': 0.05 ** 2 * np.identity(2),
        'init_m': np.asarray([6.5, 2.5, 0.00, 12, 0.1]),
        'p_basis': np.array([
            [2.5, 0.0],
            [2.5, 1.0],
            [0.0, 1.0],
            [-2.5, 1.0],
            [-2.5, 0.0],
            [-2.5, -1.0],
            [0.0, -1.0],
            [2.5, -1.0],
        ]),
        'mean_scat_number': 125,
    }

    data_source = SplineDataSimulator(dt=dt, **config)

    for i in range(steps):
        print('step: {:3d}'.format(i))

        data_source.step()

    gt, measurements = data_source.extract()
    bboxes = data_source.extrackt_bbox()

    stride = 5

    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 1, figsize=(20.0, 10.0))
    fig.suptitle('Single Target Framework Data Generation', fontsize=12, x=0.02, horizontalalignment='left')

    for a in ax:
        a.set_xlabel(r'$x$')
        a.set_ylabel(r'$y$')
        a.set_aspect('equal')

    ax[1].get_shared_x_axes().join(ax[0], ax[1])
    ax[1].get_shared_y_axes().join(ax[0], ax[1])


    def plot_rectangle(bboxes, ax, step, c='#ff7f0e'):

        for bbox in bboxes[bboxes['ts'] % step == 0]:
            s_phi_offset, c_phi_offset = np.sin(bbox['orientation']), np.cos(bbox['orientation'])
            rot = np.array([[c_phi_offset, - s_phi_offset], [s_phi_offset, c_phi_offset]])
            offset_xy = np.dot(rot, 0.5 * bbox['dimension'])

            r = Rectangle(xy=bbox['center_xy'] - offset_xy, width=bbox['dimension'][0], height=bbox['dimension'][1],
                          angle=np.rad2deg(bbox['orientation']))

            ax.add_artist(r)
            r.set_clip_box(ax.bbox)
            r.set_alpha(0.8)
            r.set_facecolor('none')
            r.set_edgecolor(c)


    plot_rectangle(bboxes, ax[1], stride, c='#ff7f0e')
    ax[0].plot(gt['m'][:, 0], gt['m'][:, 1], label='track', c=color_sequence[0])

    sel = measurements['ts'] % stride == 0
    ax[1].plot(measurements['xy'][sel, 0], measurements['xy'][sel, 1],
               c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.5, label='measurements')

    ax[0].plot(measurements['xy'][:, 0], measurements['xy'][:, 1],
               c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.5, label='measurements')

    for a in ax:
        a.legend()

    plt.show()

    if False:  # change to True to write data
        import os
        path = os.path.join(os.getcwd(), 'data')
        if not os.path.exists(path):
            os.makedirs(path)

        filename = r'simulated_data'
        np.save(os.path.join(path, filename + '.npy'), measurements)
        filename = r'gt_path'
        np.save(os.path.join(path, filename + '.npy'), gt)
        filename = r'gt_bboxes'
        np.save(os.path.join(path, filename + '.npy'), bboxes)
