#!/usr/bin/env python
import os
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse
from bruce_slam.utils.conversions import *
from multiprocessing import Pool
import pandas as pd
import seaborn as sns
import rosbag

plt.style.use("seaborn-talk")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

def calculate_coverage(occ, bbox):
    xmin = bbox.points[0].x
    ymin = bbox.points[0].y
    xmax = bbox.points[2].x
    ymax = bbox.points[2].y

    x0 = occ.info.origin.position.x
    y0 = occ.info.origin.position.y
    width = occ.info.width
    height = occ.info.height
    resolution = occ.info.resolution
    occ = np.array(occ.data).reshape(height, width)
    rmin = max(0, np.int32(np.round((ymin - y0) / resolution)))
    cmin = max(0, np.int32(np.round((xmin - x0) / resolution)))
    rmax = min(height - 1, np.int32(np.round((ymax - y0) / resolution)))
    cmax = min(width - 1, np.int32(np.round((xmax - x0) / resolution)))
    occ = occ[rmin : rmax + 1, cmin : cmax + 1]
    known = np.sum((occ != -1) & ((occ > 55) | (occ < 45)))
    coverage = known * (resolution ** 2) / ((xmax - xmin) * (ymax - ymin))
    return coverage


def extract_from_bag(bagname):
    print("Read bag " + bagname)
    alg = os.path.basename(bagname).split("-", 1)[0]
    bag = rosbag.Bag(bagname)

    sync_topics = [
        "/bruce/slam/mapping/occupancy",
        "/bruce/slam/slam/pose",
        "/bruce/slam/slam/traj",
    ]
    sync = defaultdict(dict)

    bbox = None
    for topic, msg, timestamp in bag.read_messages():
        if topic == "/bruce/exploration/server/bbox":
            bbox = msg
        if topic in sync_topics:
            sync[msg.header.stamp][topic] = msg
    items = sync.items()
    items.sort()

    data = []
    prev_pose = None
    dist = 0.0
    for timestamp, msgs in items:
        if len(msgs) != len(sync_topics):
            continue

        occ = msgs["/bruce/slam/mapping/occupancy"]
        pose_msg = msgs["/bruce/slam/slam/pose"]
        traj_msg = msgs["/bruce/slam/slam/traj"]
        pose = pose322(r2g(pose_msg.pose.pose))
        cov = np.array(pose_msg.pose.covariance).reshape(6, 6)
        cov = cov[np.ix_((0, 1, 5), (0, 1, 5))]
        det = np.linalg.det(cov) ** (1.0 / 3)
        coverage = calculate_coverage(occ, bbox)

        traj = r2n(traj_msg)[:, (0, 1, 5)]

        if prev_pose is not None:
            dd = np.linalg.norm(g2n(prev_pose.between(pose)))
            dist += dd

        prev_pose = pose

        data.append((dist, det, coverage))
    bag.close()

    if len(data) == 0:
        print("Empty bag " + bagname)

    return alg, np.array(data)


def extract_and_save(log_dir):
    bags = [
        os.path.join(log_dir, bagname)
        for bagname in os.listdir(log_dir)
        if bagname.endswith(".bag")
    ]
    colors = {"em": "b", "nf": "r", "nbv": "g", "heuristic": "c"}

    pool = Pool(processes=12)
    logs = list(pool.imap_unordered(extract_from_bag, bags))

    with open(os.path.join(log_dir, "logs.pkl"), "w") as f:
        pickle.dump(logs, f)


def load_and_plot(log_dir):
    with open(os.path.join(log_dir, "logs.pkl"), "r") as f:
        logs = pickle.load(f)

    colors = {"em": "b", "nf": "g", "nbv": "r", "heuristic": "c"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    for alg, data in logs:
        ax1.plot(data[:, 0], data[:, 1], colors[alg], label=alg.upper(), ms=10)
        ax2.plot(data[:, 0], data[:, 2], colors[alg], label=alg.upper(), ms=10)

    ax1.set_ylabel('Uncertainty')
    ax2.set_ylabel('Coverage')
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'fig.pdf'), bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1]
    cmd = sys.argv[2]
    if cmd == "save":
        extract_and_save(log_dir)
    elif cmd == 'plot':
        load_and_plot(log_dir)

    # bags = [
    #     os.path.join(log_dir, bagname)
    #     for bagname in os.listdir(log_dir)
    #     if bagname.endswith(".bag")
    # ]
    # colors = {"em": "b", "nf": "r", "nbv": "g", "heuristic": "c"}

    # pool = Pool()
    # logs = pool.imap_unordered(extract_from_bag, bags)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # for alg, dist_vs_det, dist_vs_coverage in logs:
    #     if len(data) != 0:
    #         ax1.plot(data[:, 0], data[:, 1], color=colors[alg])
    #     if len(data2) != 0:
    #         ax2.plot(data2[:, 0], data2[:, 1], color=colors[alg])
    # plt.show()
