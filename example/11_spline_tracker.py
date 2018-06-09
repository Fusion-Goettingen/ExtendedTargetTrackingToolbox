__author__ = "Jens Honer"
__copyright__ = "Copyright 2018, Jens Honer Tracking Toolbox"
__email__ = "-"
__license__ = "mit"
__version__ = "1.0"
__status__ = "Prototype"

import numpy as np
import os
import cProfile, pstats, io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from metric import point_set_wasserstein_distance as pt_wsd
from misc import convert_rectangle_to_eight_point


def create_spline_tracker(steps):
    dt = 0.04
    sa_sq = 1.5 ** 2
    somega_sq = 0.05 ** 2
    q = np.zeros((5 + 2, 5 + 2), dtype='f4')
    q[2, 2] = somega_sq * dt ** 3 / 3.0
    q[2, 4] = somega_sq * dt ** 2 / 2.0
    q[4, 2] = q[2, 4]
    q[4, 4] = somega_sq * dt
    q[3, 3] = sa_sq * dt
    q[0, 0] = 1e-3 * dt
    q[1, 1] = 1e-3 * dt
    q[5, 5] = 1e-5 * dt
    q[6, 6] = q[5, 5]

    config = {
        'steps': steps + 1,
        'd': 2,
        'sd': 7,
        'alpha': 1.0,
        'beta': 2.0,
        'kappa': 2.0,
        'sa_sq': sa_sq,
        'somega_sq': somega_sq,
        'q': q,
        'r': 0.05 ** 2 * np.identity(2),
        #                       x    y   phi  vx  o sx sy
        'init_m': np.asarray([6.5, 2.5, 0.00, 12, 0, 1, 1]),
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
        'init_c': 2 ** 2 * q + np.diag([1.25, 1.25, 0.05, 0.0, 0.0, 0.0, 0.0]),
        'scale_correction': True,
        'orientation_correction': True,
        'fixed_size': False,
    }

    from models.spline import UkfSplineTracker, EkfSplineTracker
    from models.spline import plot_spline
    tracker = EkfSplineTracker(dt=dt, **config)

    return tracker, plot_spline, config


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


if __name__ == "__main__":

    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    np.seterr('warn')

    path = os.path.join(os.getcwd(), 'data')
    measurements = np.load(os.path.join(path, 'simulated_data' + '.npy'))
    gt_bboxes = np.load(os.path.join(path, 'gt_bboxes' + '.npy'))
    gt = np.load(os.path.join(path, 'gt_path' + '.npy'))
    steps = max(measurements['ts']) + 1

    # tracker definition
    pr = cProfile.Profile()

    tracker, plot_estimates, config = create_spline_tracker(steps)

    for i in range(steps):
        print('step: {:3d}'.format(i))
        scan = measurements[measurements['ts'] == i]

        pr.enable()
        tracker.step(scan)
        pr.disable()

    estimates, log_lik = tracker.extract()
    bboxes = tracker.extrackt_bbox()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(25)

    print('total time: {:f}, total steps: {:d}, average step time: {:f}'.format(
        ps.total_tt, max(measurements['ts']) - 2, ps.total_tt / (max(measurements['ts']) - 2)))
    print(s.getvalue())

    stride = 5

    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    fig, ax = plt.subplots(2, 3, figsize=(20.0, 10.0))
    fig.suptitle('Single Target Framework', fontsize=12, x=0.02, horizontalalignment='left')

    for a in ax[:, 0]:
        a.set_xlabel(r'$x$')
        a.set_ylabel(r'$y$')
        a.set_aspect('equal')

    ax[1, 0].get_shared_x_axes().join(ax[0, 0], ax[1, 0])
    ax[1, 0].get_shared_y_axes().join(ax[0, 0], ax[1, 0])

    t_range_m1 = np.arange(steps + 1) - 1
    t_range = np.arange(steps)

    ax[0, 0].plot(estimates['m'][:, 0], estimates['m'][:, 1], label='track', c=color_sequence[0])
    ax[0, 0].plot(measurements['xy'][:, 0], measurements['xy'][:, 1],
                  c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.3, label='measurements')
    ax[0, 0].legend()

    plot_estimates(estimates, ax[1, 0], stride)
    plot_rectangle(bboxes, ax[1, 0], stride, c='#ff7f0e')
    plot_rectangle(gt_bboxes, ax[1, 0], stride, c='#2ca02c')
    sel = measurements['ts'] % stride == 0
    ax[1, 0].plot(measurements['xy'][sel, 0], measurements['xy'][sel, 1],
                  c='k', marker='.', linewidth=0, markersize=0.5, alpha=0.3, label='measurements')

    ax[0, 1].plot(t_range_m1, log_lik, label='log lik')
    eight_pts = convert_rectangle_to_eight_point(bboxes[1:])  # drop prior bounding box
    eight_pts_gt = convert_rectangle_to_eight_point(gt_bboxes)
    wsd = np.zeros(len(gt_bboxes), dtype='f8')
    for i, (pts, gt_pts) in enumerate(zip(eight_pts, eight_pts_gt)):
        wsd[i] = pt_wsd(pts, gt_pts, p=2.0)
        print('wsd', i, wsd[i])
    ax[1, 1].plot(t_range, wsd, label='Wasserstein Distance')

    ax[0, 2].plot(t_range_m1, estimates['m'][:, 0], label='x position')
    ax[0, 2].plot(t_range_m1, estimates['m'][:, 1], label='y position')
    ax[0, 2].plot(t_range, gt['m'][:, 0], label='ground truth x position')
    ax[0, 2].plot(t_range, gt['m'][:, 1], label='ground truth y position')

    p_basis_factors = np.max(config.get('p_basis'), axis=0)
    ax[1, 2].plot(t_range_m1, 2 * p_basis_factors[0] * estimates['m'][:, 5], label='x dimension')
    ax[1, 2].plot(t_range_m1, 2 * p_basis_factors[1] * estimates['m'][:, 6], label='y dimension')
    ax[1, 2].plot(t_range_m1, estimates['m'][:, 2], label='orientation')
    ax[1, 2].plot(t_range, gt_bboxes['dimension'][:, 0], label='ground truth x dimension')
    ax[1, 2].plot(t_range, gt_bboxes['dimension'][:, 1], label='ground truth y dimension')
    ax[1, 2].plot(t_range, gt_bboxes['orientation'], label='ground truth orientation')

    for a_row in ax:
        for a in a_row:
            a.legend()

    plt.show()
