__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
import numpy.linalg as la

from misc import vv_outer_guv
from misc import ut

from tracker import SingleTargetTracker
from matplotlib.patches import Polygon


def k_bare(d, f, l):
    """

    Parameters
    ----------
    d: array_like
        kernel distance on S1
    f: array_like
        kernel amplitude
    l: array_like
        length scale of the kernel

    Returns
    -------

    array_like, periodic gaussian kernel.

    """
    return f * np.exp(-(2 * np.sin(0.5 * np.absolute(d)) ** 2) / l)


def plot_gp(estimates, ax, stride=10, c='#1f77b4'):
    for est in estimates:
        if est['ts'] % stride == 0:
            ax.add_patch(Polygon(
                np.vstack([np.cos(est['uf'] + est['m'][2]) * est['xf'] + est['m'][0],
                           np.sin(est['uf'] + est['m'][2]) * est['xf'] + est['m'][1]]).T,
                facecolor='none', edgecolor=c, alpha=0.5, zorder=5, linewidth=1.0))


class ScGpTracker(SingleTargetTracker):
    """
    Spline tracker class with an ekf implementation of the correction step.

    Additional functions: prediction, correction.

    Attributes
    ----------
    _p_transition_normed        normalized spline transition points
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

        self._id2 = np.identity(2)

        self._d = kwargs.get('d')
        self._sd = kwargs.get('sd')

        self._dt_z = np.dtype([('xy', 'f4', self._d),
                               ])

        self._uf = kwargs.get('uf')
        self._xf_dim = len(self._uf)
        self._sl_sq = kwargs.get('sl') ** 2
        self._sf_sq = kwargs.get('sf') ** 2
        self._sr_sq = kwargs.get('sr') ** 2
        self._r = kwargs.get('r')
        self._r_bare = 0.5 * np.trace(kwargs.get('r'))

        self._k_uu = k_bare(self._uf[None, :] - self._uf[:, None], self._sf_sq, self._sl_sq)
        self._k_uu_inv = np.linalg.inv(self._k_uu)

        self._dt_extr = np.dtype([
            ('ts', 'i4'),
            ('log_w', 'f8'),
            ('m', 'f8', self._sd),
            ('c', 'f8', (self._sd, self._sd)),
            ('xf', 'f8', (self._xf_dim,)),
            ('uf', 'f8', (self._xf_dim,)),
        ])
        self._dt_scgp = np.dtype([
            ('m', 'f8', self._sd + self._xf_dim),
            ('u', 'f8', (2 * self._sd + 1, self._sd)),
            ('c', 'f8', (self._sd + self._xf_dim, self._sd + self._xf_dim)),
            ('xf', 'f8', (self._xf_dim,)),
            ('uf', 'f8', (self._xf_dim,)),
            ('log_w', 'f8'),
        ])

        self._scgp_prior = np.zeros(self._steps, dtype=self._dt_scgp)
        self._scgp_post = np.zeros(self._steps, dtype=self._dt_scgp)

        self._scgp_post['m'][0, :self._sd] = kwargs.get('init_m')
        self._scgp_post['m'][0, self._sd:] = kwargs.get('xf')

        self._scgp_post['c'][0, :self._sd, :self._sd] = kwargs.get('init_c')
        self._scgp_post['c'][0, self._sd:, self._sd:] = self._k_uu

        self._scgp_post['xf'][0] = kwargs.get('xf')
        self._scgp_post['uf'] = kwargs.get('uf')

        self._xf0 = kwargs.get('xf')
        self._alpha_f = kwargs.get('alpha_f')

        self._ff = np.exp(- self._alpha_f * self._dt)
        self._ff_sq = self._ff ** 2
        self._qf = (1 - self._ff_sq) * self._k_uu

        self._q = kwargs.get('q')

        from scipy.linalg import block_diag
        self._block_diag = block_diag

        # ukf parameters
        self._ukf_a = kwargs.get('alpha', 1.0)
        self._ukf_b = kwargs.get('beta', 2.0)
        self._ukf_k = kwargs.get('kappa', 2.0)

        self._ukf_lambda = self._ukf_a ** 2 * (self._sd + self._ukf_k) - self._sd

        # ukf weights
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

    def predict(self):
        # transfer
        self._scgp_prior[self._uc] = self._scgp_post[self._uc - 1]

        # unscented transformation
        ut(self._scgp_post['m'][self._uc - 1, :self._sd], self._scgp_post['c'][self._uc - 1, :self._sd, :self._sd],
           self._ukf_lambda, m_a=self._scgp_prior['u'][self._uc])

        # evolution
        self._transition_f(self._scgp_prior['u'][self._uc])

        # condensation
        self._scgp_prior['m'][self._uc, :self._sd] = \
            np.average(self._scgp_prior['u'][self._uc], weights=self._w_m, axis=0)
        m_diff = self._scgp_prior['u'][self._uc] - self._scgp_prior['m'][self._uc, None, :self._sd]

        self._scgp_prior['c'][self._uc, :self._sd, :self._sd] = \
            np.sum(self._w_c[:, None, None] * vv_outer_guv(m_diff, m_diff), axis=0)
        self._scgp_prior['c'][self._uc, :self._sd, :self._sd] += self._q

        # state dependent part
        w = 0.5 * self._dt ** 2 * np.array([np.cos(self._scgp_post['m'][self._uc - 1, 2]),
                                            np.sin(self._scgp_post['m'][self._uc - 1, 2])])
        self._scgp_prior['c'][self._uc, :2, :2] += self._sigma_a_sq * np.outer(w, w)
        v = self._sigma_a_sq * w * self._dt
        self._scgp_prior['c'][self._uc, 3, :2] += v
        self._scgp_prior['c'][self._uc, :2, 3] += v

        self._scgp_prior['m'][self._uc, self._sd:] -= self._xf0
        self._scgp_prior['m'][self._uc, self._sd:] *= self._ff
        self._scgp_prior['m'][self._uc, self._sd:] += self._xf0

        self._scgp_prior['c'][self._uc, self._sd:, self._sd:] *= self._ff_sq
        self._scgp_prior['c'][self._uc, self._sd:, self._sd:] += self._qf

    def correct(self, z):

        p = z['xy'] - self._scgp_prior['m'][self._uc, :2]
        p_norm = np.sqrt(np.einsum('ij, ij -> i', p, p))
        p /= p_norm[:, None]
        p_dyadic_p = vv_outer_guv(p, p)
        dp = p_dyadic_p - self._id2[None, :, :]
        dp /= p_norm[:, None, None]  # 2x2 matrix

        theta = np.arctan2(p[:, 1], p[:, 0]) - self._scgp_prior['m'][self._uc, 2]  # psi
        dtheta = np.transpose([p[:, 1], -p[:, 0]]) / p_norm[:, None]

        d = theta[:, None] - self._scgp_prior['uf'][self._uc, None, :]

        k_theta = k_bare(d, self._sf_sq, self._sl_sq)

        h_f = np.dot(k_theta, self._k_uu_inv)
        dh_f = np.dot(-np.sin(d) / self._sl_sq * k_theta, self._k_uu_inv)

        h_f_xf = np.dot(h_f, self._scgp_prior['m'][self._uc, self._sd:])
        dh_f_xf = np.dot(dh_f, self._scgp_prior['m'][self._uc, self._sd:])

        dh_x_c = self._id2[None, :, :] + dp * h_f_xf[:, None, None] + vv_outer_guv(p, dtheta) * dh_f_xf[:, None, None]

        dh_x_c.shape = (-1, 2)
        dh_x_f = vv_outer_guv(p, h_f)

        dh_x_f.shape = (-1, self._xf_dim)
        dh_psi = - (p * dh_f_xf[:, None])[:, :, None]
        dh_psi.shape = (-1, 1)

        h = np.concatenate([dh_x_c, dh_psi, np.zeros((len(z) * 2, self._sd - 3), dtype='f4'), dh_x_f], axis=-1)

        z_hat = p * h_f_xf[:, None] + self._scgp_prior['m'][self._uc, :2]
        residuals = z['xy'] - z_hat
        residuals.shape = (-1)

        r_f_scalar = self._sf_sq + self._sr_sq + self._r_bare - \
                     np.einsum('zi, ij, zj -> z', k_theta, self._k_uu_inv, k_theta)
        r_f = self._block_diag(*(p_dyadic_p * r_f_scalar[:, None, None] + self._r))

        # Bayes correction
        cht = np.dot(self._scgp_prior['c'][self._uc], h.T)
        s = np.dot(h, cht) + r_f
        inv_s = la.inv(s)
        gain = np.dot(cht, inv_s)

        self._scgp_post[self._uc] = self._scgp_prior[self._uc]
        self._scgp_post['m'][self._uc] = self._scgp_prior['m'][self._uc] + np.dot(gain, residuals)
        self._scgp_post['c'][self._uc] = self._scgp_prior['c'][self._uc] - np.dot(cht, gain.T)
        self._log_lik[self._uc] = np.dot(residuals, np.dot(inv_s, residuals))

        self._scgp_post['c'][self._uc] += self._scgp_post['c'][self._uc].T
        self._scgp_post['c'][self._uc] *= 0.5

    def extract(self):
        """
        Gaussian Process extraction function

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of GP targets

        """

        scgp_extr = np.zeros(self._steps, dtype=self._dt_extr)
        scgp_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        scgp_extr['log_w'] = self._log_lik
        scgp_extr['m'] = self._scgp_post['m'][:, :self._sd]
        scgp_extr['c'] = self._scgp_post['c'][:, :self._sd, :self._sd]
        scgp_extr['xf'] = self._scgp_post['m'][:, self._sd:]
        scgp_extr['uf'] = self._scgp_post['uf']

        return scgp_extr, self._log_lik

    def extrackt_bbox(self):
        """
        Bounding box extraction function.
        The algorithm creates the minimal bounding box defined by the support of the GP.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of bounding boxes

        """

        bbox_extr = np.zeros(self._steps, dtype=self._dt_bbox)
        bbox_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        bbox_extr['orientation'] = self._scgp_post['m'][:, 2]

        poly_x = np.cos(self._uf) * self._scgp_post['m'][:, self._sd:] + self._scgp_post['m'][:, 0, None]
        min_x, max_x = np.min(poly_x, axis=-1), np.max(poly_x, axis=-1)
        bbox_extr['center_xy'][:, 0] = 0.5 * (max_x + min_x)
        bbox_extr['dimension'][:, 0] = max_x - min_x

        poly_y = np.sin(self._uf) * self._scgp_post['m'][:, self._sd:] + self._scgp_post['m'][:, 1, None]
        min_y, max_y = np.min(poly_y, axis=-1), np.max(poly_y, axis=-1)
        bbox_extr['center_xy'][:, 1] = 0.5 * (max_y + min_y)
        bbox_extr['dimension'][:, 1] = max_y - min_y

        return bbox_extr


class DecorrelatedScGpTracker(ScGpTracker):
    """
    Decorrelated version of the ScGp tracker. In particular the correlation between kinematic shape and
    extent is zeroed out.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def correct(self, z):
        p = z['xy'] - self._scgp_prior['m'][self._uc, :2]
        p_norm = np.sqrt(np.einsum('ij,ij->i', p, p))
        p /= p_norm[:, None]
        p_dyadic_p = vv_outer_guv(p, p)
        dp = p_dyadic_p - self._id2[None, :, :]
        dp /= p_norm[:, None, None]  # 2x2 matrix

        theta = np.arctan2(p[:, 1], p[:, 0]) - self._scgp_prior['m'][self._uc, 2]  # psi
        dtheta = np.transpose([p[:, 1], -p[:, 0]]) / p_norm[:, None]

        d = theta[:, None] - self._scgp_prior['uf'][self._uc, None, :]

        k_theta = k_bare(d, self._sf_sq, self._sl_sq)

        h_f = np.dot(k_theta, self._k_uu_inv)
        dh_f = np.dot(-np.sin(d) / self._sl_sq * k_theta, self._k_uu_inv)

        h_f_xf = np.dot(h_f, self._scgp_prior['m'][self._uc, self._sd:])
        dh_f_xf = np.dot(dh_f, self._scgp_prior['m'][self._uc, self._sd:])

        dh_x_c = self._id2[None, :, :] + dp * h_f_xf[:, None, None] + vv_outer_guv(p, dtheta) * dh_f_xf[:, None, None]

        dh_x_c.shape = (-1, 2)
        dh_x_f = vv_outer_guv(p, h_f)

        dh_x_f.shape = (-1, self._xf_dim)
        dh_psi = - (p * dh_f_xf[:, None])[:, :, None]
        dh_psi.shape = (-1, 1)

        h_kin = np.concatenate([dh_x_c, dh_psi, np.zeros((len(z) * 2, self._sd - 3), dtype='f4')], axis=-1)
        h_shape = dh_x_f

        z_hat = p * h_f_xf[:, None] + self._scgp_prior['m'][self._uc, :2]
        residuals = (z['xy'] - z_hat)
        residuals.shape = (-1)

        r_f_scalar = self._sf_sq + self._sr_sq + self._r_bare - \
            np.einsum('zi, ij, zj -> z', k_theta, self._k_uu_inv, k_theta)
        r_f = self._block_diag(*(p_dyadic_p * r_f_scalar[:, None, None] + self._r))

        # Bayes correction
        cht_kin = np.dot(self._scgp_prior['c'][self._uc, :self._sd, :self._sd], h_kin.T)
        cht_shape = np.dot(self._scgp_prior['c'][self._uc, self._sd:, self._sd:], h_shape.T)
        s = np.dot(h_kin, cht_kin) + np.dot(h_shape, cht_shape) + r_f
        inv_s = la.inv(s)

        gain_kin = np.dot(cht_kin, inv_s)
        gain_shape = np.dot(cht_shape, inv_s)

        self._scgp_post[self._uc] = self._scgp_prior[self._uc]
        self._scgp_post['m'][self._uc, :self._sd] = \
            self._scgp_prior['m'][self._uc, :self._sd] + np.dot(gain_kin, residuals)
        self._scgp_post['m'][self._uc, self._sd:] = \
            self._scgp_prior['m'][self._uc, self._sd:] + np.dot(gain_shape, residuals)

        self._scgp_post['c'][self._uc, :self._sd, :self._sd] = \
            self._scgp_prior['c'][self._uc, :self._sd, :self._sd] - np.dot(cht_kin, gain_kin.T)
        self._scgp_post['c'][self._uc, self._sd:, self._sd:] = \
            self._scgp_prior['c'][self._uc, self._sd:, self._sd:] - np.dot(cht_shape, gain_shape.T)
        self._scgp_post['c'][self._uc, self._sd:, :self._sd] = 0
        self._scgp_post['c'][self._uc, :self._sd, self._sd:] = 0

        self._log_lik[self._uc] = np.einsum('z, zc, c', residuals, inv_s, residuals)

