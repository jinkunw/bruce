#!/usr/bin/env python
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
import rosparam

plt.style.use("seaborn-talk")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

"""
This script is to load logs from ros LOG_DIR
that are produced by em_exploration_server and then plot
occupancy map with virtual landmarks.

"""

LOG_DIR = os.path.expanduser("~/.ros/")
params = rosparam.load_file(LOG_DIR + "em_server.yaml")
# params = params[0][0]["bruce"]
pose_nstd = 1
# sigma0 = params["exploration"]["server"]["virtual_map"]["sigma0"]
sigma0 = 3.0
# resolution = params["exploration"]["server"]["virtual_map"]["resolution"]
resolution = 2.0
# Ellipse will fill the entire cell.
vm_nstd = resolution / 2.0 / sigma0
# icp noise model (sigma x when p(z) = 1.0)
# sx0 = params["slam"]["icp_odom_sigmas"][0]
sx0 = 0.1


def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def load_states(step):
    bs = []
    vm = []
    graph = []
    for i in range(100000):
        if not os.path.exists(LOG_DIR + "em-step-{}-vm{}.csv".format(step, i)):
            break
        bs_i = np.loadtxt(LOG_DIR + "em-step-{}-bs{}.csv".format(step, i))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph_i = np.loadtxt(
                LOG_DIR + "em-step-{}-graph{}.csv".format(step, i), ndmin=2
            )
        f_vm = open(LOG_DIR + "em-step-{}-vm{}.csv".format(step, i))
        header = np.loadtxt(f_vm, max_rows=1)
        vm_i = {"extent": header[:4]}
        n_rows = int(header[4])
        while True:
            if f_vm.tell() == os.fstat(f_vm.fileno()).st_size:
                break
            layer = np.loadtxt(f_vm, max_rows=1, dtype=str).tostring()
            vm_i[layer] = np.loadtxt(f_vm, max_rows=n_rows)

        bs.append(bs_i)
        vm.append(vm_i)
        graph.append(graph_i)

    return bs, vm, graph


def rot(mat):
    return np.fliplr(np.rot90(mat))


if __name__ == "__main__":
    import sys

    step = int(sys.argv[1])
    print("Read logs at step {}".format(step))
    bs, vm, graph = load_states(step)

    bs0, vm0, graph0 = bs[0], vm[0], graph[0]
    occ0 = vm0["occupancy"]
    occ0[np.isnan(occ0)] = 50
    occ0 = rot(occ0)
    extent = np.array(vm0["extent"])
    extent[2], extent[3] = extent[3], extent[2]

    res = (extent[1] - extent[0]) / occ0.shape[1]
    cov011 = rot(vm0["cov11"])
    cov012 = rot(vm0["cov12"])
    cov022 = rot(vm0["cov22"])

    for i in range(len(bs)):
        fig, ax = plt.subplots(1, 1, True, True, figsize=(10, 10))

        bs0, vm0, graph0 = bs[0], vm[0], graph[0]
        bsi, vmi, graphi = bs[i], vm[i], graph[i]

        # plot occupancy grid map
        occi = vmi["occupancy"]
        occi[np.isnan(occi)] = 50
        occi = rot(occi)
        extent = np.array(vmi["extent"])
        extent[2], extent[3] = extent[3], extent[2]
        ax.imshow(
            occi, extent=extent, origin="upper", cmap="gray_r", vmax=100, alpha=0.5
        )

        ax.plot(bs0[:, 0], bs0[:, 1], "k", lw=1)
        if i > 0:
            ax.plot(bsi[len(bs0) - 1 :, 0], bsi[len(bs0) - 1 :, 1], "r", lw=1)

        for j in range(len(bs0)):
            c, s = np.cos(bs0[j, 2]), np.sin(bs0[j, 2])
            cov = bs0[j, -9:].reshape(3, 3)
            plot_cov_ellipse(
                bs0[j, :2], cov[:2, :2], nstd=pose_nstd, ax=ax, fill=False, color="k"
            )
        if i > 0:
            # Propagate cov using odom model
            c, s = np.cos(bs0[-1, 2]), np.sin(bs0[-1, 2])
            R1 = np.array([[c, -s], [s, c]])
            t1 = bs0[-1, :2]
            # make a copy!
            cov1 = np.array(bs0[-1, -9:].reshape(3, 3))
            # global -> local cov
            cov1[:2, :] = R1.T.dot(cov1[:2, :])
            cov1[:, :2] = cov1[:, :2].dot(R1)
            for j in range(len(bs0), len(bsi)):
                c, s = np.cos(bsi[j, 2]), np.sin(bsi[j, 2])
                R2 = np.array([[c, -s], [s, c]])
                t2 = bsi[j, :2]
                # jacobian in local coordinates
                H = np.identity(3, np.float32)
                H[:2, :2] = R2.T.dot(R1)
                H[:2, 2] = np.array([[0, 1], [-1, 0]]).dot(R2.T).dot(t1 - t2)
                cov2 = H.dot(cov1).dot(H.T) + np.diag([0.2, 0.2, 0.02]) ** 2
                R1, t1, cov1 = R2, t2, cov2
                # local -> global cov
                gcov2 = R2.dot(cov2[:2, :2]).dot(R2.T)
                plot_cov_ellipse(
                    t2, gcov2, nstd=pose_nstd, ax=ax, fill=False, color="k"
                )
            for j in range(len(bsi)):
                c, s = np.cos(bsi[j, 2]), np.sin(bsi[j, 2])
                cov = bsi[j, -9:].reshape(3, 3)
                plot_cov_ellipse(
                    bsi[j, :2],
                    cov[:2, :2],
                    nstd=pose_nstd,
                    ax=ax,
                    fill=False,
                    color="r",
                )

        # plot non-sequential constraints
        if graph0.size:
            for x1, y1, _, x2, y2, _, sx, _, _ in graph0:
                ax.plot((x1, x2), (y1, y2), "k--", lw=0.5, alpha=(sx0 / sx) ** 2)
        if i > 0:
            if graphi.size:
                for x1, y1, _, x2, y2, _, sx, _, _ in graphi[len(graph0) :]:
                    ax.plot((x1, x2), (y1, y2), "r--", lw=0.5, alpha=(sx0 / sx) ** 2)

        covi11 = rot(vmi["cov11"])
        covi12 = rot(vmi["cov12"])
        covi22 = rot(vmi["cov22"])
        for r, c in np.ndindex(occi.shape):
            if occi[r, c] == 0:
                continue

            pos = extent[0] + res * (c + 0.5), extent[3] + res * (r + 0.5)
            cov = np.array([[covi11[r, c], covi12[r, c]], [covi12[r, c], covi22[r, c]]])
            plot_cov_ellipse(
                pos, cov, nstd=vm_nstd, ax=ax, color="k", alpha=0.5, fill=False
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.tight_layout()
        plt.savefig(
            LOG_DIR + "step-{}-path-{}.png".format(step, i),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close("all")
        print("Finished {}/{}".format(i, len(bs) - 1))
