__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np

from misc import vv_outer_guv, vv_dot_guv, cross_2d_guv, diag_guv, mm_dot_guv, add_block_diag_guv
from numba import guvectorize, float64, float32, int64, int32

from tracker import SingleTargetTracker
from misc import batch_ut, ut


@guvectorize([(float64[:], float64[:], float64[:]),
              (float32[:], float32[:], float32[:])],
             '(n), (n) -> ()',
             nopython=True, target='cpu')
def left_selected_dot_2d(a, b, out):
    out[0] = (a[0] * b[0] + a[1] * b[1]) - 1e9 * (a[0] * b[1] - a[1] * b[0] < 0)


@guvectorize([(int64[:], int64[:], int64[:]),
              (int32[:], int32[:], int32[:])],
             '(), (n), (m)',
             nopython=True, target='cpu')
def map_idx(ref_idx, idx, out):
    out[0] = idx[ref_idx[0]]
    out[1] = idx[ref_idx[0] + 1]
    out[2] = idx[ref_idx[0] + 2]


@guvectorize([(float64[:], float64[:, :], float64[:], float64[:]),
              (float32[:], float32[:, :], float32[:], float32[:])],
             '(n), (n, m), (), (m)',
             nopython=True, target='cpu')
def nuisance_calculation(u_scaled, mp_base_sel_bare, out_tau, out_pf):
    if np.abs(u_scaled[0]) < 1e-9:
        out_tau[0] = - u_scaled[2] / u_scaled[1]
        out_pf[:] = u_scaled[2] * mp_base_sel_bare[1] - u_scaled[1] * mp_base_sel_bare[2]
        out_pf[:] /= u_scaled[1] ** 2
    else:
        theta = np.sqrt(u_scaled[1] ** 2 - 4.0 * u_scaled[0] * u_scaled[2])  # use other solution
        out_tau[0] = - theta - u_scaled[1]
        out_tau[0] /= 2 * u_scaled[0]

        out_pf[:] = mp_base_sel_bare[0] * (theta + u_scaled[1]) / (u_scaled[0] ** 2)
        out_pf[:] += - mp_base_sel_bare[1] / u_scaled[0]
        out_pf[:] += \
            - (u_scaled[1] * mp_base_sel_bare[1]
               - 2 * u_scaled[2] * mp_base_sel_bare[0]
               - 2 * u_scaled[0] * mp_base_sel_bare[2]
               ) / (u_scaled[0] * theta)

        out_pf[:] *= 0.5


@guvectorize([(float64[:], float64[:]),
              (float32[:], float32[:])],
             '(n), ()',
             nopython=True, target='cpu')
def nuisance_calculation_bare(u_scaled, out_tau):
    if np.abs(u_scaled[0]) < 1e-9:
        out_tau[0] = - u_scaled[2] / u_scaled[1]
    else:
        theta = np.sqrt(u_scaled[1] ** 2 - 4.0 * u_scaled[0] * u_scaled[2])  # use other solution
        out_tau[0] = - theta - u_scaled[1]
        out_tau[0] /= 2 * u_scaled[0]


def plot_spline(estimates, ax, stride=10, c='#1f77b4'):
    tau = np.linspace(0, 1, 100)
    tau_vec = np.vstack((tau ** 2, tau, np.ones(len(tau)))).T
    m_base = np.array([[0.5, -1.0, 0.5],
                       [-1.0, 1.0, 0.0],
                       [0.5, 0.5, 0.0]], dtype='f4')

    for est in estimates[estimates['ts'] % stride == 0]:
        s_phi, c_phi = np.sin(est['m'][2]), np.cos(est['m'][2])
        rot = np.asarray([[c_phi, -s_phi],
                          [s_phi, c_phi]])

        for i in range(len(est['p_basis'])):
            sel = np.arange(i, i + 3, dtype='i4') % len(est['p_basis'])
            surface = np.dot(tau_vec, np.dot(m_base, est['p_basis'][sel]))
            surface = np.dot(surface, rot.T) + est['m'][:2]
            ax.plot(surface[:, 0], surface[:, 1], color=c, alpha=0.5, zorder=5, linewidth=1.0)


class SplineTrackerBare(SingleTargetTracker):
    """
    Tracker base class.

    Features the basic steps : step, prediction, correction, reduction, extraction.

    Attributes
    ----------
    _d                  dimension of the measurement space
    _sd                 dimension of the state space
    _sd_red             dimension of the reduced state space, i.e. state space without extent
    _m_basis            spline m matrix
    _p_basis_unscaled   spline knots
    _p_basis_dim        number of spline knots
    _p_transition       spline transition points
    _dt_extr            dtype of the extraction array
    _dt_spline          dtype of the spline track
    _spline_prior       time series of the spline track prior
    _spline_post        time series of the spline track posterior
    _r                  measurement uncertainty
    _q                  process noise

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._id2 = np.identity(2)

        self._d = kwargs.get('d')
        self._sd = kwargs.get('sd')
        self._sd_red = self._sd - 2

        self._m_basis = np.array([[0.5, -1.0, 0.5],
                                  [-1.0, 1.0, 0.0],
                                  [0.5, 0.5, 0.0]], dtype='f4')

        self._p_basis_unscaled = kwargs.get('p_basis')
        self._p_basis_dim = len(self._p_basis_unscaled)

        self._sel_idx_full = np.arange(0, self._p_basis_dim + 3, dtype='i4') % self._p_basis_dim
        self._p_transition = np.zeros(self._p_basis_unscaled.shape, dtype='f4')
        for i in range(self._p_basis_dim):
            sel = np.arange(i, i + 3, dtype='i4') % self._p_basis_dim
            self._p_transition[i] = np.dot(self._m_basis[-1], self._p_basis_unscaled[sel])

        self._dt_extr = np.dtype([
            ('ts', 'i4'),
            ('log_w', 'f8'),
            ('m', 'f8', self._sd),
            ('c', 'f8', (self._sd, self._sd)),
            ('p_basis', 'f4', self._p_basis_unscaled.shape),
        ])
        self._dt_spline = np.dtype([
            ('m', 'f8', self._sd),
            ('u', 'f8', (2 * self._sd_red + 1, self._sd_red)),
            ('c', 'f8', (self._sd, self._sd)),
            ('log_w', 'f8'),
        ])

        self._spline_prior = np.zeros(self._steps, dtype=self._dt_spline)
        self._spline_post = np.zeros(self._steps, dtype=self._dt_spline)

        self._spline_post['m'][0, :self._sd] = kwargs.get('init_m')
        self._spline_post['c'][0, :self._sd, :self._sd] = kwargs.get('init_c')

        self._q = kwargs.get('q')
        self._r = kwargs.get('r')

    def _transition_f(self, u):
        """
        transition function for state vector [x, y, phi, v, omega]

        time difference is implicit

        Parameters
        ----------
        u       array_like
            sigma points

        """

        u[:, 0] += self._dt * u[:, 3] * np.cos(u[:, 2])
        u[:, 1] += self._dt * u[:, 3] * np.sin(u[:, 2])

        u[:, 2] += self._dt * u[:, 4]

    def correct(self, z):
        raise NotImplementedError

    def extract(self):
        """
        Spline extraction function.
        Note: timestamp -1 is the initial state without any measurements processed.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of spline targets

        """
        scgp_extr = np.zeros(self._steps, dtype=self._dt_extr)
        scgp_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        scgp_extr['log_w'] = self._log_lik
        scgp_extr['m'] = self._spline_post['m'][:, :self._sd]
        scgp_extr['c'] = self._spline_post['c'][:, :self._sd, :self._sd]
        scgp_extr['p_basis'] = self._p_basis_unscaled * self._spline_post['m'][:, None, -2:]

        return scgp_extr, self._log_lik

    def extrackt_bbox(self):
        """
        Bounding box extraction function.
        The algorithm creates the minimal bounding box defined by the spline knots.
        Note: timestamp -1 is the initial state without any measurements processed.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of bounding boxes

        """

        bbox_extr = np.zeros(self._steps, dtype=self._dt_bbox)
        bbox_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        bbox_extr['center_xy'] = self._spline_post['m'][:, :2]
        bbox_extr['orientation'] = self._spline_post['m'][:, 2]
        basis_dim = np.max(self._p_basis_unscaled, axis=0)
        bbox_extr['dimension'] = 2 * self._spline_post['m'][:, -2:] * basis_dim

        return bbox_extr


class EkfSplineTracker(SplineTrackerBare):
    """
    Spline tracker class with an ekf implementation of the correction step.

    Additional functions: prediction, correction.

    Attributes
    ----------
    _p_transition_normed        nomalized spline transition points
    _ukf_a                      ukf alpha parameter
    _ukf_b                      ukf beta parameter
    _ukf_k                      ukf kappa parameter
    _ukf_lambda_red             ukf lambda parameter based on the kinematic state alone
    _w_m_red                    ukf weight array for the mean based on the kinematic state alone
    _w_c_red                    ukf weight array for the covariance based on the kinematic state alone
    _sigma_a_sq                 acceleration noise squared
    _scale_correction           trigger flag for the scale correction
    _orientation_correction     trigger flag for the orientation correction

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ukf parameters
        self._ukf_a = kwargs.get('alpha', 1.0)
        self._ukf_b = kwargs.get('beta', 2.0)
        self._ukf_k = kwargs.get('kappa', 2.0)

        # ukf weights for prediction
        self._ukf_lambda_red = self._ukf_a ** 2 * (self._sd_red + self._ukf_k) - self._sd_red

        self._w_m_red = np.zeros(2 * self._sd_red + 1, dtype='f8')
        self._w_c_red = np.zeros(2 * self._sd_red + 1, dtype='f8')

        self._w_m_red[:] = 1.0 / (self._sd_red + self._ukf_lambda_red)
        self._w_m_red[0] *= self._ukf_lambda_red
        self._w_m_red[1:] *= 0.5

        self._w_c_red[:] = 1.0 / (self._sd_red + self._ukf_lambda_red)
        self._w_c_red[0] *= self._ukf_lambda_red
        self._w_c_red[0] += 1 - self._ukf_a ** 2 + self._ukf_b
        self._w_c_red[1:] *= 0.5

        self._sigma_a_sq = kwargs.get('sa_sq')

        self._scale_correction = kwargs.get('scale_correction')
        self._orientation_correction = kwargs.get('orientation_correction')

    def predict(self):
        # transfer
        self._spline_prior[self._uc] = self._spline_post[self._uc - 1]

        # unscented transformation
        ut(self._spline_post['m'][self._uc - 1, :self._sd_red],
           self._spline_post['c'][self._uc - 1, :self._sd_red, :self._sd_red],
           self._ukf_lambda_red, m_a=self._spline_prior['u'][self._uc])

        # evolution
        self._transition_f(self._spline_prior['u'][self._uc])

        # condensation
        self._spline_prior['m'][self._uc, :self._sd_red] = \
            np.average(self._spline_prior['u'][self._uc], weights=self._w_m_red, axis=0)
        m_diff = self._spline_prior['u'][self._uc] - self._spline_prior['m'][self._uc, None, :self._sd_red]

        self._spline_prior['c'][self._uc, :self._sd_red, :self._sd_red] = \
            np.sum(self._w_c_red[:, None, None] * vv_outer_guv(m_diff, m_diff), axis=0)
        self._spline_prior['c'][self._uc] += self._q

        # state dependent part
        w = 0.5 * self._dt ** 2 * np.array([np.cos(self._spline_post['m'][self._uc - 1, 2]),
                                            np.sin(self._spline_post['m'][self._uc - 1, 2])])
        self._spline_prior['c'][self._uc, :2, :2] += self._sigma_a_sq * np.outer(w, w)
        v = self._sigma_a_sq * w * self._dt
        self._spline_prior['c'][self._uc, 3, :2] += v
        self._spline_prior['c'][self._uc, :2, 3] += v

    def correct(self, z):
        """
        correction function for the spline tracker

        Parameters
        ----------
        z:  array_like, dtype: [('ts', 'i4'), ('xy', 'f4', (2))]

        Returns
        -------

        """

        # calculate rotations
        phi = self._spline_prior['m'][self._uc, 2]
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        rot = np.asarray([[c_phi, -s_phi],
                          [s_phi, c_phi]])
        drot = np.asarray([[-s_phi, -c_phi],
                           [c_phi, -s_phi]])
        inv_rot = rot.T
        dinv_rot = drot.T

        # local measurement
        z_local = np.dot(z['xy'] - self._spline_prior['m'][self._uc, :2], inv_rot.T)
        cd_z = len(z)

        # scaling
        scaling_c_array = self._spline_prior['m'][self._uc, -2:]
        scaling_c = np.diag(self._spline_prior['m'][self._uc, -2:])
        scaling_hat_c = np.diag(self._spline_prior['m'][self._uc, -2:][::-1])

        # Transition Points
        p_transition_normed = self._p_transition * scaling_c_array[None, :]
        p_transition_normed = p_transition_normed / np.linalg.norm(p_transition_normed, axis=-1)[:, None]

        # calculate reference points
        # noinspection PyTypeChecker
        ref_idx = np.argmax(left_selected_dot_2d(p_transition_normed, z_local[:, None, :]), axis=-1)
        sel_idx = np.zeros((cd_z, 3), dtype='i8')
        map_idx(ref_idx, self._sel_idx_full, sel_idx)
        p_basis_sel = self._p_basis_unscaled[sel_idx]

        # From here the calculation of tau, z_hat and the jacobian begins
        mp_base_sel_bare = np.dot(self._m_basis, p_basis_sel)
        mp_base_sel_scaled = np.dot(self._m_basis, p_basis_sel * scaling_c_array[None, :])
        mp_base_sel_bare_t = np.transpose(mp_base_sel_bare, axes=(1, 0, 2))

        u_scaled = np.cross(mp_base_sel_scaled, z_local)
        u_scaled_t = u_scaled.T

        # derivative factors
        dys_dm = - np.dot(inv_rot, scaling_hat_c).T  # requires transpose!
        dys_dphi = vv_dot_guv(scaling_hat_c, np.dot(z['xy'] - self._spline_prior['m'][self._uc, :2], dinv_rot.T))
        dys_dscale = np.zeros((cd_z, 2, 2), dtype='f4')
        dys_dscale[:, 0, 1] = z_local[:, 1]  # implicit transposition transpose!
        dys_dscale[:, 1, 0] = z_local[:, 0]  # implicit transposition transpose!
        # dys_dscale_cmp = np.transpose(np.array([[np.zeros(cd_z), z_local[:, 1]],
        #                                        [z_local[:, 0], np.zeros(cd_z)]]), axes=(2, 0, 1))
        # if not np.isclose(dys_dscale_cmp, dys_dscale).all():
        #     raise AssertionError

        # linear vs quadratic selection
        tau_i = np.zeros(cd_z, dtype='f8')
        dtau_dq_prefactor = np.zeros((cd_z, 2), dtype='f8')
        nuisance_calculation(u_scaled_t, mp_base_sel_bare_t, tau_i, dtau_dq_prefactor)

        # stacked tau solution vectors
        tau_i_matrix = np.ones((3, cd_z), dtype='f8')
        tau_i_matrix[0] = tau_i ** 2
        tau_i_matrix[1] = tau_i
        dtau_i_matrix = np.ones((2, cd_z), dtype='f8')
        dtau_i_matrix[0] = 2 * tau_i

        # contour derivative
        c_tau = np.einsum('ij, ijn-> jn', tau_i_matrix, mp_base_sel_bare)  # (??)
        dc_dtau = np.einsum('ij, ijn-> jn', dtau_i_matrix, mp_base_sel_bare[:2])  # (35)

        # tau derivatives
        dtau_dm = cross_2d_guv(dtau_dq_prefactor[:, None, :], dys_dm)
        dtau_dphi = cross_2d_guv(dtau_dq_prefactor, dys_dphi)
        dtau_dscale = cross_2d_guv(dtau_dq_prefactor[:, None, :], dys_dscale)

        # h matrix calculations
        dh_dm = self._id2 + mm_dot_guv(rot, vv_outer_guv(np.einsum('ij, zj -> zi', scaling_c, dc_dtau), dtau_dm))
        # dh_dpsi = \
        #     np.dot(np.dot(dc_dtau * dtau_dphi[:, None], scaling_c), rot.T) + \
        #     np.dot(np.dot(c_tau, scaling_c), drot.T)
        dh_dpsi = \
            np.einsum('ij, jk, zk -> zi', rot, scaling_c, dc_dtau * dtau_dphi[:, None]) + \
            np.einsum('ij, jk, zk -> zi', drot, scaling_c, c_tau)
        dh_dscale = \
            mm_dot_guv(rot, vv_outer_guv(np.dot(dc_dtau, scaling_c), dtau_dscale)) + \
            mm_dot_guv(rot, diag_guv(c_tau))

        if not self._scale_correction:
            dh_dscale[:] = 0
        if not self._orientation_correction:
            dh_dpsi[:] = 0

        h_ekf = np.zeros((cd_z, 2, self._sd), dtype='f8')
        h_ekf[:, :, :2] = dh_dm
        h_ekf[:, :, 2] = dh_dpsi
        h_ekf[:, :, -2:] = dh_dscale
        h_ekf.shape = (-1, h_ekf.shape[-1])

        # calculation of expected measurement
        z_hat = np.dot(np.einsum('ij, ijn-> jn', tau_i_matrix, mp_base_sel_scaled), rot.T) + \
            self._spline_prior['m'][self._uc, :2]
        z_hat.shape = (-1)
        z_flat = z['xy'].reshape(-1)

        # ekf
        s = add_block_diag_guv(np.dot(np.dot(h_ekf, self._spline_prior['c'][self._uc]), h_ekf.T), self._r)
        inv_s = np.linalg.inv(s)
        gain = np.dot(np.dot(self._spline_prior['c'][self._uc], h_ekf.T), inv_s)
        residuals = z_flat - z_hat

        self._spline_post[self._uc] = self._spline_prior[self._uc]
        self._spline_post['m'][self._uc] = self._spline_prior['m'][self._uc] + np.dot(gain, z_flat - z_hat)
        self._spline_post['c'][self._uc] = self._spline_prior['c'][self._uc] - np.dot(np.dot(gain, s), gain.T)
        self._spline_post['c'][self._uc] += self._spline_post['c'][self._uc].T
        self._spline_post['c'][self._uc] *= 0.5

        self._log_lik[self._uc] = np.einsum('z, zc, c', residuals, inv_s, residuals)


class RigidEkfSplineTracker(SplineTrackerBare):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ukf parameters
        self._ukf_a = kwargs.get('alpha', 1.0)
        self._ukf_b = kwargs.get('beta', 2.0)
        self._ukf_k = kwargs.get('kappa', 2.0)

        # ukf weights for prediction
        self._ukf_lambda_red = self._ukf_a ** 2 * (self._sd + self._ukf_k) - self._sd

        self._w_m_red = np.zeros(2 * self._sd + 1, dtype='f8')
        self._w_c_red = np.zeros(2 * self._sd + 1, dtype='f8')

        self._w_m_red[:] = 1.0 / (self._sd + self._ukf_lambda_red)
        self._w_m_red[0] *= self._ukf_lambda_red
        self._w_m_red[1:] *= 0.5

        self._w_c_red[:] = 1.0 / (self._sd + self._ukf_lambda_red)
        self._w_c_red[0] *= self._ukf_lambda_red
        self._w_c_red[0] += 1 - self._ukf_a ** 2 + self._ukf_b
        self._w_c_red[1:] *= 0.5

        self._sigma_a_sq = kwargs.get('sa_sq')

        self._orientation_correction = kwargs.get('orientation_correction')

        self._p_transition_normed = self._p_transition / np.linalg.norm(self._p_transition, axis=-1)[:, None]

        self._dt_spline = np.dtype([
            ('m', 'f8', self._sd),
            ('u', 'f8', (2 * self._sd + 1, self._sd)),
            ('c', 'f8', (self._sd, self._sd)),
            ('log_w', 'f8'),
        ])

        self._spline_prior = np.zeros(self._steps, dtype=self._dt_spline)
        self._spline_post = np.zeros(self._steps, dtype=self._dt_spline)

        self._spline_post['m'][0, :self._sd] = kwargs.get('init_m')
        self._spline_post['c'][0, :self._sd, :self._sd] = kwargs.get('init_c')

    def predict(self):
        # transfer
        self._spline_prior[self._uc] = self._spline_post[self._uc - 1]

        # unscented transformation
        ut(self._spline_post['m'][self._uc - 1], self._spline_post['c'][self._uc - 1],
           self._ukf_lambda_red, m_a=self._spline_prior['u'][self._uc])

        # evolution
        self._transition_f(self._spline_prior['u'][self._uc])

        # condensation
        self._spline_prior['m'][self._uc] = \
            np.average(self._spline_prior['u'][self._uc], weights=self._w_m_red, axis=0)
        m_diff = self._spline_prior['u'][self._uc] - self._spline_prior['m'][self._uc, None, :]

        self._spline_prior['c'][self._uc] = \
            np.sum(self._w_c_red[:, None, None] * vv_outer_guv(m_diff, m_diff), axis=0)
        self._spline_prior['c'][self._uc] += self._q

        # state dependent part
        w = 0.5 * self._dt ** 2 * np.array([np.cos(self._spline_post['m'][self._uc - 1, 2]),
                                            np.sin(self._spline_post['m'][self._uc - 1, 2])])
        self._spline_prior['c'][self._uc, :2, :2] += self._sigma_a_sq * np.outer(w, w)
        v = self._sigma_a_sq * w * self._dt
        self._spline_prior['c'][self._uc, 3, :2] += v
        self._spline_prior['c'][self._uc, :2, 3] += v

    def correct(self, z):
        """
        correction function for the spline tracker

        Parameters
        ----------
        z:  array_like, dtype: [('ts', 'i4'), ('xy', 'f4', (2))]

        Returns
        -------

        """

        # calculate rotations
        phi = self._spline_prior['m'][self._uc, 2]
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        rot = np.asarray([[c_phi, -s_phi],
                          [s_phi, c_phi]])
        drot = np.asarray([[-s_phi, -c_phi],
                           [c_phi, -s_phi]])
        inv_rot = rot.T
        dinv_rot = drot.T

        # local measurement
        z_local = np.dot(z['xy'] - self._spline_prior['m'][self._uc, :2], inv_rot.T)
        cd_z = len(z)

        # calculate reference points
        # noinspection PyTypeChecker
        ref_idx = np.argmax(left_selected_dot_2d(self._p_transition_normed, z_local[:, None, :]), axis=-1)
        sel_idx = np.zeros((cd_z, 3), dtype='i8')
        map_idx(ref_idx, self._sel_idx_full, sel_idx)
        p_basis_sel = self._p_basis_unscaled[sel_idx]

        # From here the calculation of tau, z_hat and the jacobian begins
        mp_base_sel_bare = np.dot(self._m_basis, p_basis_sel)
        mp_base_sel_bare_t = np.transpose(mp_base_sel_bare, axes=(1, 0, 2))

        u_bare_t = np.cross(mp_base_sel_bare, z_local).T

        # derivative factors
        dys_dm = - inv_rot.T  # requires transpose!
        dys_dphi = np.dot(z['xy'] - self._spline_prior['m'][self._uc, :2], dinv_rot.T)

        # linear vs quadratic selection
        tau_i = np.zeros(cd_z, dtype='f8')
        dtau_dq_prefactor = np.zeros((cd_z, 2), dtype='f8')
        nuisance_calculation(u_bare_t, mp_base_sel_bare_t, tau_i, dtau_dq_prefactor)

        # stacked tau solution vectors
        tau_i_matrix = np.vstack([tau_i ** 2, tau_i, np.ones(len(tau_i))])
        dtau_i_matrix = np.vstack([2 * tau_i, np.ones(len(tau_i))])

        c_tau = np.einsum('ij, ijn-> jn', tau_i_matrix, mp_base_sel_bare)
        dc_dtau = np.einsum('ij, ijn-> jn', dtau_i_matrix, mp_base_sel_bare[:2])

        # tau derivatives
        dtau_dm = cross_2d_guv(dtau_dq_prefactor[:, None, :], dys_dm)
        dtau_dphi = cross_2d_guv(dtau_dq_prefactor, dys_dphi)

        # h matrix calculations
        dh_dm = self._id2 + mm_dot_guv(rot, vv_outer_guv(dc_dtau, dtau_dm))
        dh_dpsi = \
            np.einsum('ij, zj -> zi', rot, dc_dtau * dtau_dphi[:, None]) + \
            np.einsum('ij, zj -> zi', drot, c_tau)

        if not self._orientation_correction:
            dh_dpsi[:] = 0

        h_ekf = np.zeros((cd_z, 2, self._sd), dtype='f8')
        h_ekf[:, :, :2] = dh_dm
        h_ekf[:, :, 2] = dh_dpsi
        h_ekf.shape = (-1, h_ekf.shape[-1])

        # calculation of expected measurement
        z_hat = np.dot(np.einsum('ij, ijn-> jn', tau_i_matrix, mp_base_sel_bare), rot.T) + \
                self._spline_prior['m'][self._uc, :2]
        z_hat.shape = (-1)
        z_flat = z['xy'].reshape(-1)

        # ekf
        s = add_block_diag_guv(np.dot(np.dot(h_ekf, self._spline_prior['c'][self._uc]), h_ekf.T), self._r)
        inv_s = np.linalg.inv(s)
        gain = np.dot(np.dot(self._spline_prior['c'][self._uc], h_ekf.T), inv_s)
        residuals = z_flat - z_hat

        self._spline_post[self._uc] = self._spline_prior[self._uc]
        self._spline_post['m'][self._uc] = self._spline_prior['m'][self._uc] + np.dot(gain, z_flat - z_hat)
        self._spline_post['c'][self._uc] = self._spline_prior['c'][self._uc] - np.dot(np.dot(gain, s), gain.T)
        self._spline_post['c'][self._uc] += self._spline_post['c'][self._uc].T
        self._spline_post['c'][self._uc] *= 0.5

        self._log_lik[self._uc] = np.einsum('z, zc, c', residuals, inv_s, residuals)

    def extract(self):
        """
        Spline extraction function.
        Note: timestamp -1 is the initial state without any measurements processed.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of spline targets

        """

        scgp_extr = np.zeros(self._steps, dtype=self._dt_extr)
        scgp_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        scgp_extr['log_w'] = self._log_lik
        scgp_extr['m'] = self._spline_post['m'][:, :self._sd]
        scgp_extr['c'] = self._spline_post['c'][:, :self._sd, :self._sd]
        scgp_extr['p_basis'] = self._p_basis_unscaled

        return scgp_extr, self._log_lik

    def extrackt_bbox(self):
        """
        Bounding box extraction function.
        The algorithm creates the minimal bounding box defined by the spline knots.
        Note: timestamp -1 is the initial state without any measurements processed.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of bounding boxes

        """

        bbox_extr = np.zeros(self._steps, dtype=self._dt_bbox)
        bbox_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        bbox_extr['center_xy'] = self._spline_post['m'][:, :2]
        bbox_extr['orientation'] = self._spline_post['m'][:, 2]
        basis_dim = np.max(self._p_basis_unscaled, axis=0)
        bbox_extr['dimension'] = 2 * basis_dim

        return bbox_extr


class UkfSplineTracker(SplineTrackerBare):
    """
    Spline tracker class with an ukf implementation of the correction step.

    Additional functions: prediction, correction.

    Attributes
    ----------
    _p_transition_normed        nomalized spline transition points
    _ukf_a                      ukf alpha parameter
    _ukf_b                      ukf beta parameter
    _ukf_k                      ukf kappa parameter
    _ukf_lambda_red             ukf lambda parameter based on the kinematic state alone
    _w_m_red                    ukf weight array for the mean based on the kinematic state alone
    _w_c_red                    ukf weight array for the covariance based on the kinematic state alone
    _ukf_lambda                 ukf lambda parameter based on the full state
    _w_m                        ukf weight array for the mean based on the full state
    _w_c                        ukf weight array for the covariance based on the full state
    _sigma_a_sq                 acceleration noise squared
    _scale_correction           trigger flag for the scale correction
    _orientation_correction     trigger flagg for the orientation correction

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._p_transition_normed = self._p_transition / np.linalg.norm(self._p_transition, axis=-1)[:, None]

        # ukf parameters
        self._ukf_a = kwargs.get('alpha', 1.0)
        self._ukf_b = kwargs.get('beta', 2.0)
        self._ukf_k = kwargs.get('kappa', 2.0)

        # ukf weights for prediction
        self._ukf_lambda_red = self._ukf_a ** 2 * (self._sd_red + self._ukf_k) - self._sd_red

        self._w_m_red = np.zeros(2 * self._sd_red + 1, dtype='f8')
        self._w_c_red = np.zeros(2 * self._sd_red + 1, dtype='f8')

        self._w_m_red[:] = 1.0 / (self._sd_red + self._ukf_lambda_red)
        self._w_m_red[0] *= self._ukf_lambda_red
        self._w_m_red[1:] *= 0.5

        self._w_c_red[:] = 1.0 / (self._sd_red + self._ukf_lambda_red)
        self._w_c_red[0] *= self._ukf_lambda_red
        self._w_c_red[0] += 1 - self._ukf_a ** 2 + self._ukf_b
        self._w_c_red[1:] *= 0.5

        # ukf weights for ciorrection
        self._ukf_lambda = self._ukf_a ** 2 * (self._sd + self._ukf_k) - self._sd

        self._w_m = np.zeros(2 * self._sd + 1, dtype='f8')
        self._w_c = np.zeros(2 * self._sd + 1, dtype='f8')

        self._w_m[:] = 1.0 / (self._sd + self._ukf_lambda)
        self._w_m[0] *= self._ukf_lambda
        self._w_m[1:] *= 0.5

        self._w_c[:] = 1.0 / (self._sd + self._ukf_lambda)
        self._w_c[0] *= self._ukf_lambda
        self._w_c[0] += 1 - self._ukf_a ** 2 + self._ukf_b
        self._w_c[1:] *= 0.5

        self._sigma_a_sq = kwargs.get('sa_sq')

        self._scale_correction = kwargs.get('scale_correction')
        self._orientation_correction = kwargs.get('orientation_correction')

    def predict(self):
        # transfer
        self._spline_prior[self._uc] = self._spline_post[self._uc - 1]

        # unscented transformation
        ut(self._spline_post['m'][self._uc - 1, :self._sd_red],
           self._spline_post['c'][self._uc - 1, :self._sd_red, :self._sd_red],
           self._ukf_lambda_red, m_a=self._spline_prior['u'][self._uc])

        # evolution
        self._transition_f(self._spline_prior['u'][self._uc])

        # condensation
        self._spline_prior['m'][self._uc, :self._sd_red] = \
            np.average(self._spline_prior['u'][self._uc], weights=self._w_m_red, axis=0)
        m_diff = self._spline_prior['u'][self._uc] - self._spline_prior['m'][self._uc, None, :self._sd_red]

        self._spline_prior['c'][self._uc, :self._sd_red, :self._sd_red] = \
            np.sum(self._w_c_red[:, None, None] * vv_outer_guv(m_diff, m_diff), axis=0)
        self._spline_prior['c'][self._uc] += self._q

        # state dependent part
        w = 0.5 * self._dt ** 2 * np.array([np.cos(self._spline_post['m'][self._uc - 1, 2]),
                                            np.sin(self._spline_post['m'][self._uc - 1, 2])])
        self._spline_prior['c'][self._uc, :2, :2] += self._sigma_a_sq * np.outer(w, w)
        v = self._sigma_a_sq * w * self._dt
        self._spline_prior['c'][self._uc, 3, :2] += v
        self._spline_prior['c'][self._uc, :2, 3] += v

    def correct(self, z):
        """
        correction function for the spline tracker

        Parameters
        ----------
        z:  array_like, dtype: [('ts', 'i4'), ('xy', 'f4', (2))]

        Returns
        -------

        """
        # calculate sigma points
        m_a = np.zeros(
            (1, 2 * self._spline_prior['m'][self._uc].shape[-1] + 1, self._spline_prior['m'][self._uc].shape[-1]),
            dtype='f8')
        batch_ut(
            self._spline_prior['m'][self._uc].reshape((1,) + self._spline_prior['m'][self._uc].shape),
            self._spline_prior['c'][self._uc].reshape((1,) + self._spline_prior['c'][self._uc].shape),
            lmbda=self._ukf_lambda_red, m_a=m_a)
        m_a = m_a[0]

        # calculate rotation matrices
        s_phi_u, c_phi_u = np.sin(m_a[:, 2]), np.cos(m_a[:, 2])
        rot_u = np.asarray([[c_phi_u, -s_phi_u],
                            [s_phi_u, c_phi_u]])
        inv_rot_u = np.asarray([[c_phi_u, s_phi_u],
                                [-s_phi_u, c_phi_u]])

        # local measurement
        z_local = np.einsum('ecu, zuc -> uze', inv_rot_u, z['xy'][:, None, :] - m_a[:, :2])
        z_local /= m_a[:, None, -2:]  # apply scaling on local coordinates

        # Transition Points
        # noinspection PyTypeChecker
        ref_idx = np.argmax(left_selected_dot_2d(self._p_transition_normed, z_local[:, :, None, :]), axis=-1)  # [u, z]
        sel_idx = np.zeros(ref_idx.shape + (3,), dtype='i8')
        map_idx(ref_idx, self._sel_idx_full, sel_idx)
        p_basis_sel = self._p_basis_unscaled[sel_idx]  # [u, z, M, c]

        # calculate mp_base
        mp_base_sel_bare = np.dot(self._m_basis, p_basis_sel)
        mp_base_sel_scaled = mp_base_sel_bare * m_a[:, None, -2:]
        # noinspection PyTypeChecker
        u_bare_t = np.transpose(cross_2d_guv(mp_base_sel_bare, z_local), axes=(1, 2, 0))

        tau_i = np.zeros(u_bare_t.shape[:2], dtype='f8')
        nuisance_calculation_bare(u_bare_t, tau_i)

        if (tau_i < 0).any() or (tau_i > 1).any():
            raise AssertionError('tau ill-defined!')

        c_tau = mp_base_sel_scaled[2] + \
                mp_base_sel_scaled[1] * tau_i[:, :, None] + \
                mp_base_sel_scaled[0] * (tau_i ** 2)[:, :, None]
        c_tau_u = np.einsum('ecu, uzc -> uze', rot_u, c_tau)
        c_tau_u += m_a[:, None, :2]  # seems like direct addition yields different memory layout

        ukf_z_hat_mean = np.average(c_tau_u, weights=self._w_m, axis=0)
        ukf_m_mean = np.average(m_a, weights=self._w_m, axis=0)

        h_m_diff = c_tau_u - ukf_z_hat_mean
        h_m_diff.shape = (h_m_diff.shape[0], -1)
        m_diff = m_a - ukf_m_mean

        # noinspection PyTypeChecker
        c_xz_bare = np.average(vv_outer_guv(m_diff, h_m_diff), weights=self._w_c, axis=0)
        # noinspection PyTypeChecker
        c_zz_bare = np.average(vv_outer_guv(h_m_diff, h_m_diff), weights=self._w_c, axis=0)

        c_zz_inv = np.linalg.inv(add_block_diag_guv(c_zz_bare, self._r))
        gain = np.dot(c_xz_bare, c_zz_inv)

        residual = z['xy'] - ukf_z_hat_mean
        residual.shape = -1

        self._spline_post[self._uc] = self._spline_prior[self._uc]
        self._spline_post['m'][self._uc] = self._spline_prior['m'][self._uc] + np.dot(gain, residual)
        self._spline_post['c'][self._uc] = self._spline_prior['c'][self._uc] - np.dot(np.dot(gain, c_zz_bare), gain.T)
        self._spline_post['c'][self._uc] += self._spline_post['c'][self._uc].T
        self._spline_post['c'][self._uc] *= 0.5

        self._log_lik[self._uc] = np.einsum('z, zc, c', residual, c_zz_inv, residual)
