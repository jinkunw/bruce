import matplotlib

matplotlib.use("Agg")

from itertools import combinations
from collections import defaultdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.ckdtree import cKDTree as KDTree
from scipy.optimize import shgo

from sklearn.covariance import MinCovDet
import gtsam

from .sonar import OculusProperty
from .utils.conversions import *
from .utils.visualization import *
from .utils.io import *
from . import pcl


class STATUS(Enum):
    NOT_ENOUGH_POINTS = "Not enough points"
    LARGE_TRANSFORMATION = "Large transformation"
    NOT_ENOUGH_OVERLAP = "Not enough overlap"
    NOT_CONVERGED = "Not converged"
    INITIALIZATION_FAILURE = "Initialization failure"
    SUCCESS = "Success"

    def __init__(self, *args, **kwargs):
        Enum.__init__(*args, **kwargs)
        self.description = None

    def __bool__(self):
        return self == STATUS.SUCCESS

    def __nonzero__(self):
        return self == STATUS.SUCCESS

    def __str__(self):
        if self.description:
            return self.value + ": " + self.description
        else:
            return self.value


class Keyframe(object):
    def __init__(
        self, status, time, dr_pose3, points=np.zeros((0, 2), np.float32), cov=None
    ):
        self.status = status  # used to mark keyframe
        self.time = time  # time

        self.dr_pose3 = dr_pose3  # dead reckoning 3d pose
        self.dr_pose = pose322(dr_pose3)  # dead reckoning 2d pose

        self.pose3 = dr_pose3  # estimated 3d pose (will be updated later)
        self.pose = pose322(dr_pose3)  # estimated 2d pose

        self.cov = cov  # cov in local frame (always 2d)
        self.transf_cov = None  # cov in global frame

        self.points = points.astype(np.float32)  # points in local frame (always 2d)
        self.transf_points = None  # transformed points in global frame based on pose

        self.constraints = []  # Non-sequential constraints (key, odom)

        self.twist = None  # twist message for publishing odom

    def update(self, new_pose, new_cov=None):
        self.pose = new_pose
        self.pose3 = n2g(
            (
                new_pose.x(),
                new_pose.y(),
                self.dr_pose3.translation().z(),
                self.dr_pose3.rotation().roll(),
                self.dr_pose3.rotation().pitch(),
                new_pose.theta(),
            ),
            "Pose3",
        )
        self.transf_points = Keyframe.transform_points(self.points, self.pose)

        if new_cov is not None:
            self.cov = new_cov

        if self.cov is not None:
            c, s = np.cos(self.pose.theta()), np.sin(self.pose.theta())
            R = np.array([[c, -s], [s, c]])
            self.transf_cov = np.array(self.cov)
            self.transf_cov[:2, :2] = R.dot(self.transf_cov[:2, :2]).dot(R.T)
            self.transf_cov[:2, 2] = R.dot(self.transf_cov[:2, 2])
            self.transf_cov[2, :2] = self.transf_cov[2, :2].dot(R.T)

    @staticmethod
    def transform_points(points, pose):
        if len(points) == 0:
            return np.empty_like(points, np.float32)

        T = pose.matrix().astype(np.float32)
        return points.dot(T[:2, :2].T) + T[:2, 2]


class InitializationResult(object):
    def __init__(self):
        # all points are in local frame
        self.source_points = np.zeros((0, 2))
        self.target_points = np.zeros((0, 2))
        self.source_key = None
        self.target_key = None
        self.source_pose = None
        self.target_pose = None
        # Cov for sampling
        self.cov = None
        self.occ = None

        self.status = None
        self.estimated_source_pose = None
        self.source_pose_samples = None

    def plot(self, title):
        # fmt: off
        plt.figure()
        # Plot in global frame
        points = Keyframe.transform_points(self.target_points, self.target_pose)
        plt.plot(points[:, 0], points[:, 1], "k.", ms=1, label="target points")
        plt.plot(self.source_pose.x(), self.source_pose.y(), "r+", ms=10)
        points = Keyframe.transform_points(self.source_points, self.source_pose)
        plt.plot(points[:, 0], points[:, 1], "r.", ms=1, label="source points (guess)")
        if self.cov is not None:
            c, s = np.cos(self.source_pose.theta()), np.sin(self.source_pose.theta())
            R = np.array([[c, -s], [s, c]])
            cov = R.dot(self.cov[:2, :2]).dot(R.T)
            plot_cov_ellipse((self.source_pose.x(), self.source_pose.y()), cov, nstd=3, fill=False, color="r")
        if self.estimated_source_pose is not None:
            plt.plot(self.estimated_source_pose.x(), self.estimated_source_pose.y(), "g+", ms=10)
            points = Keyframe.transform_points(self.source_points, self.estimated_source_pose)
            plt.plot(points[:, 0], points[:, 1], "g.", ms=1, label="source points (initialized)")
        if self.source_pose_samples is not None:
            poses = np.array(self.source_pose_samples)
            plt.scatter(poses[:, 0], poses[:, 1], c=poses[:, 3], s=1, label="pose samples")
            plt.colorbar()
        if self.occ:
            x0, y0, resolution, occ_arr = self.occ
            x1 = x0 + (occ_arr.shape[1] - 0.5) * resolution
            y1 = y0 + (occ_arr.shape[0] - 0.5) * resolution
            plt.imshow(occ_arr, origin='upper', extent=(x0, x1, y1, y0), cmap='Greys', vmin=0, vmax=1, alpha=0.5)
            plt.colorbar()
        plt.legend()
        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title(str(self.status))
        plt.savefig(title, dpi=100)
        plt.close("all")
        # fmt: on

    def save(self, filename):
        np.savez(
            filename,
            source_points=self.source_points,
            target_points=self.target_points,
            source_pose=g2n(self.source_pose),
            target_pose=g2n(self.target_pose),
            estimated_source_pose=g2n(self.estimated_source_pose),
        )


class ICPResult(object):
    def __init__(self, init_ret, use_samples=False, sample_eps=0.01):
        # all points are in local frame
        self.source_points = init_ret.source_points
        self.target_points = init_ret.target_points
        self.source_key = init_ret.source_key
        self.target_key = init_ret.target_key
        self.source_pose = init_ret.source_pose
        self.target_pose = init_ret.target_pose

        self.status = init_ret.status
        if init_ret.estimated_source_pose is not None:
            self.initial_transform = self.target_pose.between(
                init_ret.estimated_source_pose
            )
        else:
            self.initial_transform = self.target_pose.between(self.source_pose)
        self.estimated_transform = None
        # Cov derived from ICP
        self.cov = None

        self.initial_transforms = None
        if use_samples and init_ret.source_pose_samples is not None:
            idx = np.argsort(init_ret.source_pose_samples[:, -1])
            transforms = [
                self.target_pose.between(n2g(g, "Pose2"))
                for g in init_ret.source_pose_samples[idx, :3]
            ]
            filtered = [transforms[0]]
            for b in transforms[1:]:
                d = np.linalg.norm(g2n(filtered[-1].between(b)))
                if d < sample_eps:
                    continue
                else:
                    filtered.append(b)
            self.initial_transforms = filtered
        self.sample_transforms = None

        # Whether the result is inserted to factor graph
        self.inserted = False

    def plot(self, title):
        # fmt: off
        plt.figure()
        # Plot in target frame
        plt.plot(self.target_points[:, 0], self.target_points[:, 1], "k.", ms=1, label="target points")
        plt.plot(self.initial_transform.x(), self.initial_transform.y(), "r+", ms=10)
        points = Keyframe.transform_points(self.source_points, self.initial_transform)
        plt.plot(points[:, 0], points[:, 1], "r.", ms=1, label="source points (guess)")
        if self.estimated_transform is not None:
            plt.plot(self.estimated_transform.x(), self.estimated_transform.y(), "g+", ms=10)
            points = Keyframe.transform_points(self.source_points, self.estimated_transform)
            plt.plot(points[:, 0], points[:, 1], "g.", ms=1, label="source points (estimated)")
            if self.cov is not None:
                cov = self.cov[:2, :2]
                c, s = np.cos(self.estimated_transform.theta()), np.sin(self.estimated_transform.theta())
                R = np.array([[c, -s], [s, c]])
                cov = R.dot(cov).dot(R.T)
                plot_cov_ellipse((self.estimated_transform.x(), self.estimated_transform.y()), cov, nstd=3, color="g", fill=False)
        if self.sample_transforms is not None:
            plt.scatter(self.sample_transforms[:, 0], self.sample_transforms[:, 1], color='c', s=1, label="sample estimate")

        plt.legend()
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.title(str(self.status))
        plt.savefig(title, dpi=100)
        plt.close("all")
        # fmt: on

    def save(self, filename):
        np.savez(
            filename,
            source_points=self.source_points,
            target_points=self.target_points,
            source_pose=g2n(self.source_pose),
            target_pose=g2n(self.target_pose),
            initial_transform=g2n(self.initial_transform),
            estimated_transform=g2n(self.estimated_transform),
            cov=self.cov,
        )


class SMParams(object):
    def __init__(self):
        # Use occupancy probability map matching to initialize ICP
        self.initialization = None
        # Global search params
        self.initialization_params = None
        # Minimum number of points
        self.min_points = None
        # Max deviation from initial guess
        self.max_translation = None
        self.max_rotation = None

        # Min separation between source key and the last target frame
        self.min_st_sep = None
        # Number of source frames to build source points
        # Not used in SSM
        self.source_frames = None
        # Number of target frames to build target points
        # Not used in NSSM
        self.target_frames = None

        # Number of ICP instances to run to calculate cov
        self.cov_samples = None


class SLAM(object):
    def __init__(self):
        self.oculus = OculusProperty()

        # Create a new factor when
        # - |ti - tj| > min_duration and
        # - |xi - xj| > max_translation or
        # - |ri - rj| > max_rotation
        self.keyframe_duration = None
        self.keyframe_translation = None
        self.keyframe_rotation = None

        # List of keyframes
        self.keyframes = []
        # Current (non-key)frame with real-time pose update
        # FIXME propagate cov from previous keyframe
        self.current_frame = None

        self.isam_params = gtsam.ISAM2Params()
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        # [x, y, theta]
        self.prior_sigmas = None
        # Noise model without ICP
        # [x, y, theta]
        self.odom_sigmas = None

        # Downsample point cloud for ICP and publishing
        self.point_resolution = 0.5

        # Noise radius in overlap
        self.point_noise = 0.5

        self.ssm_params = SMParams()
        self.ssm_params.initialization = True
        self.ssm_params.initialization_params = 50, 1, 0.01
        self.ssm_params.min_st_sep = 1
        self.ssm_params.min_points = 50
        self.ssm_params.max_translation = 2.0
        self.ssm_params.max_rotation = np.pi / 6
        self.ssm_params.target_frames = 3
        # Don't use ICP covariance
        self.ssm_params.cov_samples = 0

        self.nssm_params = SMParams()
        self.nssm_params.initialization = True
        self.nssm_params.initialization_params = 100, 5, 0.01
        self.nssm_params.min_st_sep = 10
        self.nssm_params.min_points = 100
        self.nssm_params.max_translation = 6.0
        self.nssm_params.max_rotation = np.pi / 2
        self.nssm_params.source_frames = 5
        self.nssm_params.cov_samples = 30

        self.icp = pcl.ICP()

        # Pairwise consistent measurement
        self.nssm_queue = []
        self.pcm_queue_size = 5
        self.min_pcm = 3

        # Use fixed noise model in two cases
        # - Sequential scan matching
        # - ICP cov is too small in non-sequential scan matching
        # [x, y, theta]
        self.icp_odom_sigmas = None

        # FIXME Can't save fig in online mode
        self.save_fig = False
        self.save_data = False

    @property
    def current_keyframe(self):
        return self.keyframes[-1]

    @property
    def current_key(self):
        return len(self.keyframes)

    def configure(self):
        assert (
            self.nssm_params.cov_samples == 0
            or self.nssm_params.cov_samples
            < self.nssm_params.initialization_params[0]
            * self.nssm_params.initialization_params[1]
        )
        assert (
            self.ssm_params.cov_samples == 0
            or self.ssm_params.cov_samples
            < self.ssm_params.initialization_params[0]
            * self.ssm_params.initialization_params[1]
        )
        assert self.nssm_params.source_frames < self.nssm_params.min_st_sep

        self.prior_model = self.create_noise_model(self.prior_sigmas)
        self.odom_model = self.create_noise_model(self.odom_sigmas)
        self.icp_odom_model = self.create_noise_model(self.icp_odom_sigmas)

        self.isam = gtsam.ISAM2(self.isam_params)

    def get_states(self):
        """
        Retrieve all states as array which are represented as
        [time, pose2, dr_pose3, cov]
          - pose2: [x, y, yaw]
          - dr_pose3: [x, y, z, roll, pitch, yaw]
          - cov: 3 x 3

        """
        states = np.zeros(
            self.current_key,
            dtype=[
                ("time", np.float64),
                ("pose", np.float32, 3),
                ("dr_pose3", np.float32, 6),
                ("cov", np.float32, 9),
            ],
        )

        # Update all
        values = self.isam.calculateEstimate()
        for key in range(self.current_key):
            pose = values.atPose2(X(key))
            cov = self.isam.marginalCovariance(X(key))
            self.keyframes[key].update(pose, cov)

        t0 = self.keyframes[0].time
        for key in range(self.current_key):
            keyframe = self.keyframes[key]
            states[key]["time"] = (keyframe.time - t0).to_sec()
            states[key]["pose"] = g2n(keyframe.pose)
            states[key]["dr_pose3"] = g2n(keyframe.dr_pose3)
            states[key]["cov"] = keyframe.transf_cov.ravel()
        return states

    @staticmethod
    def sample_pose(pose, covariance):
        delta = np.random.multivariate_normal(np.zeros(3), covariance)
        return pose.compose(n2g(delta, "Pose2"))

    def sample_current_pose(self):
        return self.sample_pose(self.current_keyframe.pose, self.current_keyframe.cov)

    def get_points(self, frames=None, ref_frame=None, return_keys=False):
        """
        - Accumulate points in frames
        - Transform them to reference frame
        - Downsample points
        - Return the corresponding keys for every point

        """
        if frames is None:
            frames = range(self.current_key)
        if ref_frame is not None:
            if isinstance(ref_frame, gtsam.Pose2):
                ref_pose = ref_frame
            else:
                ref_pose = self.keyframes[ref_frame].pose

        # Add empty point in case all points are empty
        if return_keys:
            all_points = [np.zeros((0, 3), np.float32)]
        else:
            all_points = [np.zeros((0, 2), np.float32)]
        for key in frames:
            if ref_frame is not None:
                points = self.keyframes[key].points
                pose = self.keyframes[key].pose
                transf = ref_pose.between(pose)
                transf_points = Keyframe.transform_points(points, transf)
            else:
                transf_points = self.keyframes[key].transf_points

            if return_keys:
                transf_points = np.c_[
                    transf_points, key * np.ones((len(transf_points), 1))
                ]
            all_points.append(transf_points)

        all_points = np.concatenate(all_points)
        if return_keys:
            return pcl.downsample(
                all_points[:, :2], all_points[:, (2,)], self.point_resolution
            )
        else:
            return pcl.downsample(all_points, self.point_resolution)

    def compute_icp(self, source_points, target_points, guess=gtsam.Pose2()):
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        guess = guess.matrix()
        message, T = self.icp.compute(source_points, target_points, guess)
        # ICP covariance is too small
        # cov = self.icp.getCovariance()
        x, y = T[:2, 2]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return message, gtsam.Pose2(x, y, theta)

    def compute_icp_with_cov(self, source_points, target_points, guesses):
        """
        guesses: list of initial samples
        """
        source_points = np.array(source_points, np.float32)
        target_points = np.array(target_points, np.float32)

        sample_transforms = []
        for g in guesses:
            g = g.matrix()
            message, T = self.icp.compute(source_points, target_points, g)
            if message == "success":
                x, y = T[:2, 2]
                theta = np.arctan2(T[1, 0], T[0, 0])
                sample_transforms.append((x, y, theta))

        sample_transforms = np.array(sample_transforms)
        if len(sample_transforms) < 5:
            return "Too few samples for covariance computation", None, None, None

        # Can't use np.cov(). Too many outliers
        try:
            fcov = MinCovDet(False, support_fraction=0.8).fit(sample_transforms)
        except ValueError as e:
            return "Failed to calculate covariance", None, None, None

        m = n2g(fcov.location_, "Pose2")
        cov = fcov.covariance_

        # unrotate to local frame
        R = m.rotation().matrix()
        cov[:2, :] = R.T.dot(cov[:2, :])
        cov[:, :2] = cov[:, :2].dot(R)

        default_cov = np.diag(self.icp_odom_sigmas) ** 2
        if np.linalg.det(cov) < np.linalg.det(default_cov):
            cov = default_cov

        return "success", m, cov, sample_transforms

    def get_overlap(
        self,
        source_points,
        target_points,
        source_pose=None,
        target_pose=None,
        return_indices=False,
    ):
        if source_pose:
            source_points = Keyframe.transform_points(source_points, source_pose)
        if target_pose:
            target_points = Keyframe.transform_points(target_points, target_pose)

        indices, dists = pcl.match(target_points, source_points, 1, self.point_noise)
        if return_indices:
            return np.sum(indices != -1), indices
        else:
            return np.sum(indices != -1)

    def add_prior(self, keyframe):
        # pose = gtsam.Pose2()
        pose = keyframe.pose
        factor = gtsam.PriorFactorPose2(X(0), pose, self.prior_model)
        self.graph.add(factor)
        self.values.insert(X(0), pose)

    def add_odometry(self, keyframe):
        dt = (keyframe.time - self.keyframes[-1].time).to_sec()
        dr_odom = self.keyframes[-1].pose.between(keyframe.pose)
        factor = gtsam.BetweenFactorPose2(
            X(self.current_key - 1), X(self.current_key), dr_odom, self.odom_model
        )
        self.graph.add(factor)
        self.values.insert(X(self.current_key), keyframe.pose)

    def get_map(self, frames, resolution=None):
        # Implemented in slam_node
        raise NotImplementedError

    def get_matching_cost_subroutine1(
        self,
        source_points,
        source_pose,
        target_points,
        target_pose,
        source_pose_cov=None,
    ):
        """
        Cost = - sum_i log p_i(Tx s_i \in S | t_i \in T),
                given transform Tx, source points S, target points T

                        / - prob_tp  if there exists ||Tx s_i - t_i|| < sigma,
        p_i(z_i | T) =  |
                        \ - prob_fp  otherwise
        """

        # pose_samples = []
        # target_tree = KDTree(target_points)

        # def subroutine(x):
        #     # x = [x, y, theta]
        #     delta = n2g(x, "Pose2")
        #     sample_source_pose = source_pose.compose(delta)
        #     sample_transform = target_pose.between(sample_source_pose)

        #     points = Keyframe.transform_points(source_points, sample_transform)
        #     dists, indices = target_tree.query(
        #         points, distance_upper_bound=self.point_noise
        #     )

        #     cost = -np.sum(indices != len(target_tree.data))

        #     pose_samples.append(np.r_[g2n(sample_source_pose), cost])
        #     return cost

        # return subroutine, pose_samples

        pose_samples = []
        xmin, ymin = np.min(target_points, axis=0) - 2 * self.point_noise
        xmax, ymax = np.max(target_points, axis=0) + 2 * self.point_noise
        resolution = self.point_noise / 10.0
        xs = np.arange(xmin, xmax, resolution)
        ys = np.arange(ymin, ymax, resolution)
        target_grids = np.zeros((len(ys), len(xs)), np.uint8)

        r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
        c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
        r = np.clip(r, 0, target_grids.shape[0] - 1)
        c = np.clip(c, 0, target_grids.shape[1] - 1)
        target_grids[r, c] = 255

        dilate_hs = int(np.ceil(self.point_noise / resolution))
        dilate_size = 2 * dilate_hs + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
        )
        target_grids = cv2.dilate(target_grids, kernel)

        # # Calculate distance to the nearest points
        # target_grids = cv2.bitwise_not(target_grids)
        # target_grids = cv2.distanceTransform(target_grids, cv2.DIST_L2, 3)
        # target_grids = 1.0 - 0.2 * target_grids / self.point_noise
        # target_grids = np.clip(target_grids, 0.2, 1.0)

        source_pose_info = np.linalg.inv(source_pose_cov)

        def subroutine(x):
            # x = [x, y, theta]
            delta = n2g(x, "Pose2")
            sample_source_pose = source_pose.compose(delta)
            sample_transform = target_pose.between(sample_source_pose)

            points = Keyframe.transform_points(source_points, sample_transform)
            r = np.int32(np.round((points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((points[:, 0] - xmin) / resolution))
            inside = (
                (0 <= r)
                & (r < target_grids.shape[0])
                & (0 <= c)
                & (c < target_grids.shape[1])
            )

            cost = -np.sum(target_grids[r[inside], c[inside]] > 0)

            pose_samples.append(np.r_[g2n(sample_source_pose), cost])
            return cost

        return subroutine, pose_samples

    def get_matching_cost_subroutine2(self, source_points, source_pose, occ):
        """
        Ceres scan matching

        Cost = - sum_i  ||1 - M_nearest(Tx s_i)||^2,
                given transform Tx, source points S, occupancy map M
        """
        pose_samples = []
        x0, y0, resolution, occ_arr = occ

        def subroutine(x):
            # x = [x, y, theta]
            delta = n2g(x, "Pose2")
            sample_pose = source_pose.compose(delta)

            xy = Keyframe.transform_points(source_points, sample_pose)
            r = np.int32(np.round((xy[:, 1] - y0) / resolution))
            c = np.int32(np.round((xy[:, 0] - x0) / resolution))

            sel = (r >= 0) & (c >= 0) & (r < occ_arr.shape[0]) & (c < occ_arr.shape[1])
            hit_probs_inside_map = occ_arr[r[sel], c[sel]]
            num_hits_outside_map = len(xy) - np.sum(sel)

            cost = (
                np.sum((1.0 - hit_probs_inside_map) ** 2)
                + num_hits_outside_map * (1.0 - 0.5) ** 2
            )
            cost = np.sqrt(cost / len(source_points))

            pose_samples.append(np.r_[g2n(sample_pose), cost])
            return cost

        return subroutine, pose_samples

    #######################################################
    # TODO Merge SSM and NSSM together
    #######################################################

    def initialize_sequential_scan_matching(self, keyframe):
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None
        # Match current keyframe to previous k frames
        ret.source_key = self.current_key
        ret.target_key = self.current_key - 1
        ret.source_pose = keyframe.pose
        ret.target_pose = self.current_keyframe.pose
        # Accumulate reference points from previous k frames
        ret.source_points = keyframe.points
        target_frames = range(self.current_key)[-self.ssm_params.target_frames :]
        ret.target_points = self.get_points(target_frames, ret.target_key)
        ret.cov = np.diag(self.odom_sigmas)

        if len(ret.source_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        if len(ret.target_points) < self.ssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(ret.target_points))
            return ret

        if not self.ssm_params.initialization:
            return ret

        with CodeTimer("SLAM - sequential scan matching - sampling"):
            pose_stds = np.array([self.odom_sigmas]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )

            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.ssm_params.initialization_params[0],
                iters=self.ssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.ssm_params.initialization_params[2]}
                },
            )

        if result.success:
            ret.source_pose_samples = np.array(pose_samples)
            ret.estimated_source_pose = ret.source_pose.compose(n2g(result.x, "Pose2"))
            ret.status.description = "matching cost {:.2f}".format(result.fun)

            if self.save_data:
                ret.save("step-{}-ssm-sampling.npz".format(self.current_key))
        else:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message
        return ret

    def add_sequential_scan_matching(self, keyframe):
        """
        Add sequential scan matching factor.

        kf[t - k] -- ... -- kf[t - 2] -- kf[t - 1] -- kf[t]
            |_________|________|_____________|          |
                target points in kf[t - 1]         source points

        """

        ret = self.initialize_sequential_scan_matching(keyframe)
        if self.save_fig:
            ret.plot("step-{}-ssm-sampling.png".format(self.current_key))
        if not ret.status:
            self.add_odometry(keyframe)
            return

        ret2 = ICPResult(ret, self.ssm_params.cov_samples > 0)
        # Compute ICP here with a timer
        with CodeTimer("SLAM - sequential scan matching - ICP"):
            if self.ssm_params.initialization and self.ssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.ssm_params.cov_samples],
                )
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.status.description = ""

        # Add some failure detections
        # The transformation compared to dead reckoning can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = delta.translation().norm()
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.ssm_params.max_translation
                or delta_rotation > self.ssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points, ret2.estimated_transform
            )
            if overlap < self.ssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = "overlap {}".format(overlap)

        if ret2.status:
            if ret2.cov is not None:
                icp_odom_model = self.create_full_noise_model(ret2.cov)
            else:
                icp_odom_model = self.icp_odom_model
            factor = gtsam.BetweenFactorPose2(
                X(ret2.target_key),
                X(ret2.source_key),
                ret2.estimated_transform,
                icp_odom_model,
            )
            self.graph.add(factor)
            self.values.insert(
                X(ret2.source_key), ret2.target_pose.compose(ret2.estimated_transform)
            )
            ret2.inserted = True
            if self.save_data:
                ret2.save("step-{}-ssm-icp.npz".format(self.current_key))
        else:
            self.add_odometry(keyframe)

        if self.save_fig:
            ret2.plot("step-{}-ssm-icp.png".format(self.current_key))

    def initialize_nonsequential_scan_matching(self):
        ret = InitializationResult()
        ret.status = STATUS.SUCCESS
        ret.status.description = None

        ret.source_key = self.current_key - 1
        ret.source_pose = self.current_frame.pose
        ret.estimated_source_pose = ret.source_pose
        source_frames = range(
            ret.source_key, ret.source_key - self.nssm_params.source_frames, -1
        )
        ret.source_points = self.get_points(source_frames, ret.source_key)
        if len(ret.source_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "source points {}".format(len(ret.source_points))
            return ret

        # Find target points for matching
        # Limit searching keyframes
        target_frames = range(self.current_key - self.nssm_params.min_st_sep)
        # Target points in global frame
        target_points, target_keys = self.get_points(target_frames, None, True)

        # Further limit points based on source pose uncertainty
        sel = np.zeros(len(target_points), np.bool)
        for source_frame in source_frames:
            pose = self.keyframes[source_frame].pose
            cov = self.keyframes[source_frame].cov
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            range_bound = translation_std * 5.0 + self.oculus.max_range
            bearing_bound = rotation_std * 5.0 + self.oculus.horizontal_aperture * 0.5

            local_points = Keyframe.transform_points(target_points, pose.inverse())
            ranges = np.linalg.norm(local_points, axis=1)
            bearings = np.arctan2(local_points[:, 1], local_points[:, 0])

            sel_i = (ranges < range_bound) & (abs(bearings) < bearing_bound)
            sel |= sel_i

        target_points = target_points[sel]
        target_keys = target_keys[sel]
        target_frames, counts = np.unique(np.int32(target_keys), return_counts=True)
        target_frames = target_frames[counts > 10]
        counts = counts[counts > 10]
        if len(target_frames) == 0 or len(target_points) < self.nssm_params.min_points:
            ret.status = STATUS.NOT_ENOUGH_POINTS
            ret.status.description = "target points {}".format(len(target_points))
            return ret

        ret.target_key = target_frames[np.argmax(counts)]
        ret.target_pose = self.keyframes[ret.target_key].pose
        ret.target_points = Keyframe.transform_points(
            target_points, ret.target_pose.inverse()
        )
        ret.cov = self.keyframes[ret.source_key].cov

        if not self.nssm_params.initialization:
            return ret

        with CodeTimer("SLAM - nonsequential scan matching - sampling"):
            translation_std = np.sqrt(np.max(np.linalg.eigvals(cov[:2, :2])))
            rotation_std = np.sqrt(cov[2, 2])
            pose_stds = np.array([[translation_std, translation_std, rotation_std]]).T
            pose_bounds = 5.0 * np.c_[-pose_stds, pose_stds]

            # ret.occ = self.get_map(target_frames)
            # subroutine, pose_samples = self.get_matching_cost_subroutine2(
            #     ret.source_points,
            #     ret.source_pose,
            #     ret.occ,
            # )
            subroutine, pose_samples = self.get_matching_cost_subroutine1(
                ret.source_points,
                ret.source_pose,
                ret.target_points,
                ret.target_pose,
                ret.cov,
            )

            result = shgo(
                func=subroutine,
                bounds=pose_bounds,
                n=self.nssm_params.initialization_params[0],
                iters=self.nssm_params.initialization_params[1],
                sampling_method="sobol",
                minimizer_kwargs={
                    "options": {"ftol": self.nssm_params.initialization_params[2]}
                },
            )

        if not result.success:
            ret.status = STATUS.INITIALIZATION_FAILURE
            ret.status.description = result.message
            return ret

        delta = n2g(result.x, "Pose2")
        ret.estimated_source_pose = ret.source_pose.compose(delta)
        ret.source_pose_samples = np.array(pose_samples)
        ret.status.description = "matching cost {:.2f}".format(result.fun)

        # Refine target key by searching for the pose with maximum overlap
        # with current source points
        estimated_source_points = Keyframe.transform_points(
            ret.source_points, ret.estimated_source_pose
        )
        overlap, indices = self.get_overlap(
            estimated_source_points, target_points, return_indices=True
        )
        target_frames1, counts1 = np.unique(
            np.int32(target_keys[indices[indices != -1]]), return_counts=True
        )
        if len(counts1) == 0:
            ret.status = STATUS.NOT_ENOUGH_OVERLAP
            ret.status.description = "0"
            return ret

        if self.save_data:
            ret.save("step-{}-nssm-sampling.npz".format(self.current_key - 1))
        ret.target_key = target_frames1[np.argmax(counts1)]
        ret.target_pose = self.keyframes[ret.target_key].pose
        # Recalculate target points with new target key in target frame
        ret.target_points = self.get_points(target_frames, ret.target_key)
        return ret

    def add_nonsequential_scan_matching(self):
        """
        Add non-sequential scan matching factor.

        kf[m - k1] -- ... -- kf[m] -- ... -- kf[m + k2] -- ... -- kf[t - p] -- ... -- kf[t]
            |____ _____|________|________________|                     |_______________|
                target points around kf[m]                               source points

        """
        if self.current_key < self.nssm_params.min_st_sep:
            return

        ret = self.initialize_nonsequential_scan_matching()
        if self.save_fig:
            ret.plot("step-{}-nssm-sampling.png".format(self.current_key - 1))
        if not ret.status:
            return

        ret2 = ICPResult(ret, self.nssm_params.cov_samples > 0)
        # sample_deltas = np.random.uniform(-1, 1, (self.nssm_params.cov_samples, 3))
        # sample_deltas[:, 0] *= self.icp_odom_sigmas[0] * 10
        # sample_deltas[:, 1] *= self.icp_odom_sigmas[1] * 10
        # sample_deltas[:, 2] *= self.icp_odom_sigmas[2] * 10
        # ret2.initial_transforms = [
        #     ret2.initial_transform.compose(n2g(sample_delta, "Pose2"))
        #     for sample_delta in sample_deltas
        # ]

        # Compute ICP here with a timer
        with CodeTimer("SLAM - nonsequential scan matching - ICP"):
            if self.nssm_params.initialization and self.nssm_params.cov_samples > 0:
                message, odom, cov, sample_transforms = self.compute_icp_with_cov(
                    ret2.source_points,
                    ret2.target_points,
                    ret2.initial_transforms[: self.nssm_params.cov_samples],
                )
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret2.cov = cov
                    ret2.sample_transforms = sample_transforms
                    ret2.status.description = "{} samples".format(
                        len(ret2.sample_transforms)
                    )
            else:
                message, odom = self.compute_icp(
                    ret2.source_points, ret2.target_points, ret2.initial_transform
                )
                if message != "success":
                    ret2.status = STATUS.NOT_CONVERGED
                    ret2.status.description = message
                else:
                    ret2.estimated_transform = odom
                    ret.status.description = ""

        # Add some failure detections
        # The transformation compared to initial guess can't be too large
        if ret2.status:
            delta = ret2.initial_transform.between(ret2.estimated_transform)
            delta_translation = delta.translation().norm()
            delta_rotation = abs(delta.theta())
            if (
                delta_translation > self.nssm_params.max_translation
                or delta_rotation > self.nssm_params.max_rotation
            ):
                ret2.status = STATUS.LARGE_TRANSFORMATION
                ret2.status.description = "trans {:.2f} rot {:.2f}".format(
                    delta_translation, delta_rotation
                )

        # There must be enough overlap between two point clouds.
        if ret2.status:
            overlap = self.get_overlap(
                ret2.source_points, ret2.target_points[:, :2], ret2.estimated_transform
            )
            if overlap < self.nssm_params.min_points:
                ret2.status = STATUS.NOT_ENOUGH_OVERLAP
            ret2.status.description = str(overlap)

        if ret2.status:
            if self.save_data:
                ret2.save("step-{}-nssm-icp.npz".format(self.current_key - 1))

            # # DCS
            # if ret2.cov is not None:
            #     icp_odom_model = self.create_robust_full_noise_model(ret2.cov)
            # else:
            #     icp_odom_model = self.create_robust_noise_model(self.icp_odom_sigmas)
            # factor = gtsam.BetweenFactorPose2(
            #     X(ret2.target_key),
            #     X(ret2.source_key),
            #     ret2.estimated_transform,
            #     icp_odom_model,
            # )
            # self.graph.add(factor)
            # self.keyframes[ret2.source_key].constraints.append(
            #     (ret2.target_key, ret2.estimated_transform)
            # )
            # ret2.inserted = True

            while (
                self.nssm_queue
                and ret2.source_key - self.nssm_queue[0].source_key
                > self.pcm_queue_size
            ):
                self.nssm_queue.pop(0)
            self.nssm_queue.append(ret2)
            pcm = self.verify_pcm(self.nssm_queue)

            for m in pcm:
                ret2 = self.nssm_queue[m]
                if not ret2.inserted:
                    if ret2.cov is not None:
                        icp_odom_model = self.create_full_noise_model(ret2.cov)
                    else:
                        icp_odom_model = self.icp_odom_model
                    factor = gtsam.BetweenFactorPose2(
                        X(ret2.target_key),
                        X(ret2.source_key),
                        ret2.estimated_transform,
                        icp_odom_model,
                    )
                    self.graph.add(factor)
                    self.keyframes[ret2.source_key].constraints.append(
                        (ret2.target_key, ret2.estimated_transform)
                    )
                    ret2.inserted = True

        if self.save_fig:
            ret2.plot("step-{}-nssm-icp.png".format(self.current_key - 1))
        return ret2

    def is_keyframe(self, frame):
        if not self.keyframes:
            return True

        duration = frame.time - self.current_keyframe.time
        if duration < self.keyframe_duration:
            return False

        dr_odom = self.keyframes[-1].dr_pose.between(frame.dr_pose)
        translation = dr_odom.translation().norm()
        rotation = abs(dr_odom.theta())

        return (
            translation > self.keyframe_translation or rotation > self.keyframe_rotation
        )

    def create_full_noise_model(self, cov):
        return gtsam.noiseModel_Gaussian.Covariance(cov)

    def create_robust_full_noise_model(self, cov):
        model = gtsam.noiseModel_Gaussian.Covariance(cov)
        robust = gtsam.noiseModel_mEstimator_DCS.Create(1.0)
        return gtsam.noiseModel_Robust.Create(robust, model)

    def create_noise_model(self, *sigmas):
        return gtsam.noiseModel_Diagonal.Sigmas(np.r_[sigmas])

    def create_robust_noise_model(self, *sigmas):
        model = gtsam.noiseModel_Diagonal.Sigmas(np.r_[sigmas])
        robust = gtsam.noiseModel_mEstimator_DCS.Create(1.0)
        return gtsam.noiseModel_Robust.Create(robust, model)

    def update_factor_graph(self, keyframe=None):
        if keyframe:
            self.keyframes.append(keyframe)

        self.isam.update(self.graph, self.values)
        self.graph.resize(0)
        self.values.clear()

        # Update all trajectory
        values = self.isam.calculateEstimate()
        for x in range(values.size()):
            pose = values.atPose2(X(x))
            self.keyframes[x].update(pose)

        # Only update latest cov
        cov = self.isam.marginalCovariance(X(values.size() - 1))
        self.keyframes[-1].update(pose, cov)

        for ret in self.nssm_queue:
            ret.source_pose = self.keyframes[ret.source_key].pose
            ret.target_pose = self.keyframes[ret.target_key].pose
            if ret.inserted:
                ret.estimated_transform = ret.target_pose.between(ret.source_pose)

    def verify_pcm(self, queue):
        """
        Get the pairwise consistent measurements.
        
          Consistency of two measurements Tz_{il}, Tz_{jk} is defined as
          Tz_{jk}^{-1} x (T_{j} x T_{ij}^{-1} x Tz_{il} T_{lk}), l < k

          nssm_{il}           nssm_{jk}
          -------             -------
          |     |             |     |
          | i <-|-------------|-- j |
          | |   |             |   | |
          | v   |             |   v |
          | l --|-------------|-> k |
          -------             -------
        """
        if len(queue) < self.min_pcm:
            return []

        G = defaultdict(list)
        for (a, ret_il), (b, ret_jk) in combinations(zip(range(len(queue)), queue), 2):
            pi = ret_il.target_pose
            pj = ret_jk.target_pose
            pil = ret_il.estimated_transform
            plk = ret_il.source_pose.between(ret_jk.source_pose)
            pjk1 = ret_jk.estimated_transform
            pjk2 = pj.between(pi.compose(pil).compose(plk))

            error = gtsam.Pose2.Logmap(pjk1.between(pjk2))
            md = error.dot(np.linalg.inv(ret_jk.cov)).dot(error)
            # chi2.ppf(0.99, 3) = 11.34
            if md < 11.34:
                G[a].append(b)
                G[b].append(a)

        maximal_cliques = list(self.find_cliques(G))

        if not maximal_cliques:
            return []
        maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
        if len(maximum_clique) < self.min_pcm:
            return []

        return maximum_clique

    def find_cliques(self, G):
        """Returns all maximal cliques in an undirected graph.
        """
        if len(G) == 0:
            return

        adj = {u: {v for v in G[u] if v != u} for u in G}
        Q = [None]

        subg = set(G)
        cand = set(G)
        u = max(subg, key=lambda u: len(cand & adj[u]))
        ext_u = cand - adj[u]
        stack = []

        try:
            while True:
                if ext_u:
                    q = ext_u.pop()
                    cand.remove(q)
                    Q[-1] = q
                    adj_q = adj[q]
                    subg_q = subg & adj_q
                    if not subg_q:
                        yield Q[:]
                    else:
                        cand_q = cand & adj_q
                        if cand_q:
                            stack.append((subg, cand, ext_u))
                            Q.append(None)
                            subg = subg_q
                            cand = cand_q
                            u = max(subg, key=lambda u: len(cand & adj[u]))
                            ext_u = cand - adj[u]
                else:
                    Q.pop()
                    subg, cand, ext_u = stack.pop()
        except IndexError:
            pass
