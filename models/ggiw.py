__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, JH private tracking toolbox"
__email__ = "-"
__license__ = "-"
__version__ = "0.1"
__status__ = "Prototype"

import numpy as np
import numpy.linalg as la

from scipy.special import gammaln, multigammaln
from misc import v_outer_guv

from tracker import SingleTargetTracker

from matplotlib.patches import Ellipse


def plot_ggiw(estimates, ax, stride=10, c='#1f77b4'):
    # d = estimates['V_hat'].shape[-1]
    ell_s_factor = 3.0
    rad_to_deg = 180.0 / np.pi

    for i in np.arange(0, len(estimates), stride):
        for est in estimates:
            if est['ts'] % stride == 0:
                w, v = la.eig(est['V_hat'])
                angle_deg = np.arctan2(v[1, 0], v[0, 0])
                angle_deg *= rad_to_deg
                e = Ellipse(xy=est['m'], width=w[0], height=w[1], angle=angle_deg, alpha=0.5, color=c)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.5)
                ax.add_artist(e)

                w *= ell_s_factor
                e = Ellipse(xy=est['m'], width=w[0], height=w[1], angle=angle_deg, alpha=0.2, color=c, zorder=-10)
                e.set_clip_box(ax.bbox)
                e.set_alpha(0.2)
                ax.add_artist(e)

                # var = 2.0 + 1.0 / np.maximum(1, est['nu'] - 2 * d - 2)
                # var /= np.maximum(1, est['nu'] - 2 * d - 1)
                # var /= np.maximum(1, est['nu'] - 2 * d - 4)
                # var = 3 * np.sqrt(var)
                # var += 1.0
                # e = Ellipse(xy=est['m'], width=var * w[0], height=var * w[1], angle=angle_deg)
                #
                # ax.add_artist(e)
                # e.set_clip_box(ax.bbox)
                # e.set_alpha(0.8)
                # e.set_facecolor(c)
                # e.set_zorder(-10)


class GgiwTracker(SingleTargetTracker):
    """
    Gamma Gaussian Inverse Wishart tracker class.

    Additional functions: prediction, correction, extract, extract_bbox.

    Attributes
    ----------
    _d              dimension of the measurement space
    _s              number of time derivatives
    _sd             dimension of the full state space
    _dt_ggiw        dtype of the ggiw track
    _ggiw_prior     time series of the ggiw track prior
    _ggiw_post      time series of the ggiw track posterior
    _q              bare process noise (dimension s x s)
    _q_sd           full state process noise (dimension sd x sd)
    _f              bare transition matrix (dimension s x s)
    _f_sd           full transition matrix (dimension sd x sd)
    _h              bare measurement matrix (dimension s x s)
    _h_sd           full measurement matrix (dimension sd x sd)
    _lambda         information decay parameter for the Gaussian Inverse Wishart distribution
    _eta            information decay parameter for the gamma distribution
    _n_factor       control parameter for the scattering within the extent correction
    _nu_off         lower limit for the scalar design parameter of the Gaussian Inverse Wishart distribution
    _1_d_1          d + 1, support variable
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._d = kwargs.get('d')
        self._s = kwargs.get('s')
        self._sd = self._d * self._s

        self._dt_extr = np.dtype([('ts', 'i4'),
                                  ('log_w', 'f4'),
                                  ('m', 'f4', self._sd),
                                  ('c', 'f4', (self._sd, self._sd)),
                                  ('v_hat', 'f4', (self._d, self._d)),
                                  ('v', 'f4', (self._d, self._d)),
                                  ('nu', 'f4')])

        self._dt_ggiw = np.dtype([('m', 'f8', self._sd),
                                  ('c', 'f8', (self._s, self._s)),
                                  ('c_v', 'f8', (self._sd, self._sd)),
                                  ('v', 'f8', (self._d, self._d)),
                                  ('nu', 'f8'),
                                  ('alpha', 'f8'),
                                  ('beta', 'f8'),
                                  ])

        self._ggiw_prior = np.zeros(self._steps, dtype=self._dt_ggiw)
        self._ggiw_post = np.zeros(self._steps, dtype=self._dt_ggiw)

        self._ggiw_post['m'][0] = kwargs.get('init_m')
        self._ggiw_post['c'][0] = kwargs.get('init_c')
        self._ggiw_post['v'][0] = kwargs.get('init_v')
        self._ggiw_post['nu'][0] = kwargs.get('init_nu')
        self._ggiw_post['alpha'][0] = kwargs.get('init_alpha')
        self._ggiw_post['beta'][0] = kwargs.get('init_beta')

        self._q = kwargs.get('q', np.identity(self._s))
        self._q_sd = np.kron(self._q, np.identity(self._d))
        self._f = kwargs.get('f', np.identity(self._s))
        self._f_sd = np.kron(self._f, np.identity(self._d))

        self._h = kwargs.get('h', np.identity(self._s))
        self._h_sd = np.kron(self._h, np.identity(self._d))

        self._lambda = kwargs.get('lambda_d')
        self._eta = kwargs.get('eta')

        self._n_factor = kwargs.get('n_factor')

        self._1_d_1 = 1.0 * self._d + 1.0
        self._nu_off = 1.0

    def predict(self):
        self._ggiw_prior['m'][self._uc] = np.dot(self._f_sd, self._ggiw_post['m'][self._uc - 1])

        self._ggiw_prior['c'][self._uc] = np.dot(np.dot(self._f, self._ggiw_post['c'][self._uc - 1]), self._f.T)
        self._ggiw_prior['c'][self._uc] += self._q
        self._ggiw_prior['c'][self._uc] += self._ggiw_prior['c'][self._uc].T
        self._ggiw_prior['c'][self._uc] *= 0.5

        # order is important here!
        self._ggiw_prior['v'][self._uc] = self._ggiw_post['v'][self._uc - 1]
        self._ggiw_prior['v'][self._uc] /= self._ggiw_post['nu'][self._uc - 1] - self._1_d_1
        self._ggiw_prior['nu'][self._uc] = self._1_d_1 + \
            self._lambda * (self._ggiw_post['nu'][self._uc - 1] - self._1_d_1)
        self._ggiw_prior['v'][self._uc] *= self._ggiw_prior['nu'][self._uc] - self._1_d_1
        self._ggiw_prior['v'][self._uc] += self._ggiw_prior['v'][self._uc].T
        self._ggiw_prior['v'][self._uc] *= 0.5

        self._ggiw_prior['alpha'][self._uc] = self._ggiw_post['alpha'][self._uc - 1] / self._eta
        self._ggiw_prior['beta'][self._uc] = self._ggiw_post['beta'][self._uc - 1] / self._eta

    def correct(self, z):
        z_mean = np.mean(z['xy'], axis=0)
        z_cd = len(z)
        res = z['xy'] - z_mean
        z_scat = np.einsum('ij,ik->jk', res, res)

        s = 1.0 / z_cd + self._ggiw_prior['c'][self._uc, 0, 0]  # [j]
        inv_s = 1.0 / s
        gain = self._ggiw_prior['c'][self._uc, :, 0] * inv_s
        eps = z_mean - self._ggiw_prior['m'][self._uc, :self._d]  # [j, d]
        eps.shape = -1  # reset shape to vector

        # kinematic components
        self._ggiw_post['m'][self._uc] = self._ggiw_prior['m'][self._uc]
        self._ggiw_post['m'][self._uc, ::2] += gain * eps[0]
        self._ggiw_post['m'][self._uc, 1::2] += gain * eps[1]

        self._ggiw_post['c'][self._uc] = self._ggiw_prior['c'][self._uc] - v_outer_guv(gain) * s
        self._ggiw_post['c'][self._uc] += self._ggiw_post['c'][self._uc].T
        self._ggiw_post['c'][self._uc] *= 0.5

        self._ggiw_post['nu'][self._uc] = self._ggiw_prior['nu'][self._uc] + z_cd
        self._ggiw_post['v'][self._uc] = self._ggiw_prior['v'][self._uc] + z_scat
        self._ggiw_post['v'][self._uc] += self._n_factor * v_outer_guv(eps) * inv_s
        self._ggiw_post['v'][self._uc] += self._ggiw_post['v'][self._uc].T
        self._ggiw_post['v'][self._uc] *= 0.5

        self._ggiw_post['alpha'][self._uc] = self._ggiw_prior['alpha'][self._uc] + z_cd
        self._ggiw_post['beta'][self._uc] = self._ggiw_prior['beta'][self._uc] + 1.0

        # likelihood
        self._log_lik[self._uc] = \
            gammaln(self._ggiw_post['alpha'][self._uc]) - np.log(self._ggiw_post['beta'][self._uc]) * \
            self._ggiw_post['alpha'][self._uc]
        self._log_lik[self._uc] -= \
            (gammaln(self._ggiw_prior['alpha'][self._uc]) - np.log(self._ggiw_prior['beta'][self._uc]) *
             self._ggiw_prior['alpha'][self._uc])

        self._log_lik[self._uc] += - 0.5 * self._d * (z_cd * np.log(np.pi) + np.log(z_cd) + np.log(s))
        self._log_lik[self._uc] += \
            0.5 * self._ggiw_prior['nu'][self._uc] * la.slogdet(self._ggiw_prior['v'][self._uc])[-1]
        self._log_lik[self._uc] -= \
            0.5 * self._ggiw_post['nu'][self._uc] * la.slogdet(self._ggiw_post['v'][self._uc])[-1]
        self._log_lik[self._uc] += multigammaln(0.5 * self._ggiw_post['nu'][self._uc], self._d)
        self._log_lik[self._uc] -= multigammaln(0.5 * self._ggiw_prior['nu'][self._uc], self._d)

    def extract(self):
        """
        Gamma Gaussian Inverse Wishart extraction function

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of GGIWs

        """

        ggiw_extr = np.zeros(self._steps, dtype=self._dt_extr)
        ggiw_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        ggiw_extr['log_w'] = self._log_lik
        ggiw_extr['m'] = self._ggiw_post['m']
        ggiw_extr['c'] = \
            np.einsum('jkl, jmn -> jkmln',
                      self._ggiw_post['c'], self._ggiw_post['v']).reshape((-1, self._sd, self._sd)) \
            / (self._ggiw_post['nu'][:, None, None] - 2 + self._s * (1.0 - self._d))
        ggiw_extr['v_hat'] = self._ggiw_post['v']
        ggiw_extr['v_hat'] /= np.maximum(1.0, self._ggiw_post['nu'][:, None, None] - 2 * self._1_d_1)
        ggiw_extr['v'] = self._ggiw_post['v']
        ggiw_extr['nu'] = self._ggiw_post['nu']

        return ggiw_extr, self._log_lik

    def extrackt_bbox(self):
        """
        Bounding box extraction function.
        The algorithms creates the minimal bounding box around the ellipse defined by v_hat.

        Returns
        -------
        bbox_extr: array_like
            struct containing the time series of bounding boxes

        """

        v_hat = self._ggiw_post['v']
        v_hat /= np.maximum(1.0, self._ggiw_post['nu'][:, None, None] - 2 * self._1_d_1)

        bbox_extr = np.zeros(self._steps, dtype=self._dt_bbox)
        bbox_extr['ts'] = np.arange(self._steps, dtype='i4') - 1
        bbox_extr['orientation'] = 0.5 * np.arctan2(2 * v_hat[:, 0, 1], v_hat[:, 0, 0] - v_hat[:, 1, 1])
        bbox_extr['dimension'] = 3.0 * np.sqrt(la.eigvals(v_hat))
        bbox_extr['center_xy'] = self._ggiw_post['m'][:, :self._d]

        return bbox_extr
