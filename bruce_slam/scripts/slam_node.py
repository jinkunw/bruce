#!/usr/bin/env python
import matplotlib

matplotlib.use("Agg")

import threading
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree

import tf
import rospy
from message_filters import ApproximateTimeSynchronizer
from message_filters import Cache, Subscriber
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import PointCloud2

# For exploration services
from bruce_msgs.srv import PredictSLAMUpdate, PredictSLAMUpdateResponse
from bruce_msgs.msg import ISAM2Update
from bruce_msgs.srv import GetOccupancyMap, GetOccupancyMapRequest

from bruce_slam.utils.io import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import *
from bruce_slam.slam import SLAM, Keyframe
from bruce_slam import pcl


class SLAMNode(SLAM):
    def __init__(self):
        super(SLAMNode, self).__init__()

        self.enable_slam = True

        self.pz_samples = 30
        self.pz_detection_rate = 0.5

        self.lock = threading.RLock()

    def init_node(self, ns="~"):
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_duration = rospy.Duration(self.keyframe_duration)
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        self.enable_slam = rospy.get_param(ns + "enable_slam", True)

        self.prior_sigmas = rospy.get_param(ns + "prior_sigmas")
        self.odom_sigmas = rospy.get_param(ns + "odom_sigmas")
        self.icp_odom_sigmas = rospy.get_param(ns + "icp_odom_sigmas")

        self.point_resolution = rospy.get_param(ns + "point_resolution")

        self.ssm_params.min_points = rospy.get_param(ns + "ssm/min_points")
        self.ssm_params.max_translation = rospy.get_param(ns + "ssm/max_translation")
        self.ssm_params.max_rotation = rospy.get_param(ns + "ssm/max_rotation")
        self.ssm_params.target_frames = rospy.get_param(ns + "ssm/target_frames")

        self.nssm_params.min_st_sep = rospy.get_param(ns + "nssm/min_st_sep")
        self.nssm_params.min_points = rospy.get_param(ns + "nssm/min_points")
        self.nssm_params.max_translation = rospy.get_param(ns + "nssm/max_translation")
        self.nssm_params.max_rotation = rospy.get_param(ns + "nssm/max_rotation")
        self.nssm_params.source_frames = rospy.get_param(ns + "nssm/source_frames")
        self.nssm_params.cov_samples = rospy.get_param(ns + "nssm/cov_samples")

        self.pz_samples = rospy.get_param(ns + "pz_samples")
        self.pz_detection_rate = rospy.get_param(ns + "pz_detection_rate")

        self.pcm_queue_size = rospy.get_param(ns + "pcm_queue_size")
        self.min_pcm = rospy.get_param(ns + "min_pcm")

        self.feature_odom_sync_max_delay = 0.5
        self.sonar_sub = rospy.Subscriber(
            SONAR_TOPIC, OculusPing, self.sonar_callback, queue_size=1
        )
        self.feature_sub = Subscriber(SONAR_FEATURE_TOPIC, PointCloud2)
        self.odom_sub = Subscriber(LOCALIZATION_ODOM_TOPIC, Odometry)
        self.fo_ts = ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub], 20, self.feature_odom_sync_max_delay
        )
        self.fo_ts.registerCallback(self.fo_callback)

        # self.occ_sub = rospy.Subscriber(
        #     MAPPING_OCCUPANCY_TOPIC, OccupancyGrid, self.occ_callback, queue_size=1
        # )

        self.pose_pub = rospy.Publisher(
            SLAM_POSE_TOPIC, PoseWithCovarianceStamped, queue_size=10
        )
        self.odom_pub = rospy.Publisher(SLAM_ODOM_TOPIC, Odometry, queue_size=10)
        self.traj_pub = rospy.Publisher(
            SLAM_TRAJ_TOPIC, PointCloud2, queue_size=1, latch=True
        )
        self.constraint_pub = rospy.Publisher(
            SLAM_CONSTRAINT_TOPIC, Marker, queue_size=1, latch=True
        )
        self.cloud_pub = rospy.Publisher(
            SLAM_CLOUD_TOPIC, PointCloud2, queue_size=1, latch=True
        )
        self.slam_update_pub = rospy.Publisher(
            SLAM_ISAM2_TOPIC, ISAM2Update, queue_size=5, latch=True
        )
        self.tf = tf.TransformBroadcaster()
        self.predict_slam_update_srv = rospy.Service(
            SLAM_PREDICT_SLAM_UPDATE_SERVICE,
            PredictSLAMUpdate,
            self.predict_slam_update_handler,
        )
        self.get_map_client = rospy.ServiceProxy(
            MAPPING_GET_MAP_SERVICE, GetOccupancyMap
        )
        # self.get_map_client.wait_for_service()

        icp_config = rospy.get_param(ns + "icp_config")
        self.icp.loadFromYaml(icp_config)

        self.configure()
        loginfo("SLAM node is initialized")

    def get_common_factors(self, req_key):
        """
        Get common factors and initials for path candidates
        since last slam update.
        """
        common_graph = gtsam.NonlinearFactorGraph()
        common_values = gtsam.Values()
        graph = self.isam.getFactorsUnsafe()
        for i in range(graph.size()):
            factor = graph.at(i)
            keys = factor.keys()
            if keys.size() != 2:
                continue
            key1, key2 = keys.at(0), keys.at(1)
            if gtsam.symbolIndex(key1) > req_key or gtsam.symbolIndex(key2) > req_key:
                common_graph.add(factor)
            if gtsam.symbolIndex(key1) > req_key and not common_values.exists(key1):
                common_values.insert(key1, self.isam.calculateEstimatePose2(key1))
            if gtsam.symbolIndex(key2) > req_key and not common_values.exists(key2):
                common_values.insert(key2, self.isam.calculateEstimatePose2(key2))
        return common_graph, common_values

    def prune_path(self, path):
        """
        Search for the nearest pose in path to current pose.
        The plan will not start from the beginning.
        """
        while len(path.poses) >= 2:
            pose0 = pose322(r2g(path.poses[0].pose))
            pose1 = pose322(r2g(path.poses[1].pose))
            d0 = np.linalg.norm(g2n(pose0.between(self.current_frame.pose)))
            d1 = np.linalg.norm(g2n(pose1.between(self.current_frame.pose)))
            if d1 < d0:
                path.poses.pop(0)
            else:
                break
        return path

    def get_keyframes(self, path):
        current_key = self.current_key
        current_pose = self.current_keyframe.pose
        current_cov = self.current_keyframe.cov
        for pose_msg in path.poses:
            pose = pose322(r2g(pose_msg.pose))
            odom = current_pose.between(pose)
            translation = odom.translation().norm()
            rotation = abs(odom.theta())

            if (
                translation > self.keyframe_translation
                or rotation > self.keyframe_rotation
            ):
                cov = self.propagate_covariance(
                    current_pose, current_cov, pose, self.odom_sigmas
                )
                # Return keyframe information
                yield (
                    current_key,  # key
                    odom,  # odometry from previous pose
                    cov,  # predicted covariance
                    pose,  # pose
                    pose_msg,  # ROS pose msg
                )
            else:
                continue

            current_key += 1
            current_pose = pose
            current_cov = cov

    ###################################################################
    def predict_measurements(self, pose, map_tree):
        # Find points that the robot can observe at pose
        # First search for points within max range
        center = np.array([pose.x(), pose.y()])
        idx = map_tree.query_ball_point(center, self.oculus.max_range, eps=0.1)
        if len(idx) == 0:
            return []

        idx = np.array(idx)
        # Second search for points within horizontal aperture
        points_in_FOV = map_tree.data[idx] - center
        bearings = np.arctan2(points_in_FOV[:, 1], points_in_FOV[:, 0]) - pose.theta()
        bearings = np.arctan2(np.sin(bearings), np.cos(bearings))
        sel = np.abs(bearings) < self.oculus.horizontal_aperture / 2
        return idx[sel]

    def get_measurement_probability(self, pose, cov, map_tree):
        return self.pz_detection_rate

        success = 1
        for _ in range(self.pz_samples):
            s = self.sample_pose(pose, cov)
            idx = self.predict_measurements(s, map_tree)
            if len(idx) > self.nssm_params.min_points / self.pz_detection_rate:
                success += 1
        return success / (1.0 + self.pz_samples)

    def predict_sm(
        self, current_key, odom, pose, cov, map_tree, map_keys, graph, values
    ):
        idx = self.predict_measurements(pose, map_tree)
        # Use icp_model if the robot can observe enough points
        if len(idx) > self.ssm_params.min_points / self.pz_detection_rate:
            model = self.icp_odom_model
        else:
            model = self.odom_model

        factor = gtsam.BetweenFactorPose2(
            X(current_key - 1), X(current_key), odom, model
        )
        graph.add(factor)
        values.insert(X(current_key), pose)

        # Check if nonsequential factor exists
        if len(idx) < self.nssm_params.min_points / self.pz_detection_rate:
            return

        keys, counts = np.unique(np.int32(map_keys[idx]), return_counts=True)
        # Find target key
        matched = np.argmax(counts)
        matched_key = keys[matched]
        if current_key - matched_key > self.nssm_params.min_st_sep:
            pose1 = self.keyframes[matched_key].pose
            odom = pose1.between(pose)

            # Scale noise model based on the probability of
            # obtaining the measurements
            prob = self.get_measurement_probability(pose, cov, map_tree)
            factor = gtsam.BetweenFactorPose2(
                X(matched_key), X(current_key), odom, self.scale_icp_odom_model(prob)
            )
            graph.add(factor)

    ###################################################################

    def propagate_covariance(self, pose1, cov1, pose2, sigmas):
        """
        H = [R2^T*R1 rot(-90)*R2^T*(t1 - t2)]
            [   0              1            ]
        P' = H*P*H^T + Q (local)

        """
        H = np.identity(3, np.float32)
        R1, t1 = pose1.rotation().matrix(), pose1.translation().vector()
        R2, t2 = pose2.rotation().matrix(), pose2.translation().vector()
        H[:2, :2] = R2.T.dot(R1)
        H[:2, 2] = np.array([[0, 1], [-1, 0]]).dot(R2.T).dot(t1 - t2)
        cov2 = H.dot(cov1).dot(H.T) + np.diag(sigmas) ** 2
        return cov2

    def scale_icp_odom_model(self, prob):
        """
        p * d^T * R^{-1} * d = d^T * (1/p * R)^{-1} * d
        """
        return self.create_noise_model(
            np.sqrt(1.0 / prob) * np.array(self.icp_odom_sigmas)
        )

    def predict_slam_update_handler(self, req):
        resp = PredictSLAMUpdateResponse()
        with self.lock:
            resp.keyframes, resp.isam2_updates = self.predict_slam_update(
                req.key, req.paths, req.return_isam2_update
            )

        return resp

    def predict_slam_update(self, key, paths, return_isam2_update):
        isam2_updates = []
        path_keyframes = []
        if return_isam2_update:
            common_graph, common_values = self.get_common_factors(key)
            # Points and tree are in global frame
            map_points, map_keys = self.get_points(return_keys=True)
            map_tree = KDTree(map_points)

        for path in paths:
            path = self.prune_path(path)
            if return_isam2_update:
                new_graph = gtsam.NonlinearFactorGraph(common_graph)
                new_values = gtsam.Values(common_values)

            # Add keyframes from the plan
            keyframes = Path()
            cov = self.current_keyframe.cov
            for current_key, odom, cov, pose, pose_msg in self.get_keyframes(path):
                keyframes.poses.append(pose_msg)

                if return_isam2_update:
                    self.predict_sm(
                        current_key,
                        odom,
                        pose,
                        cov,
                        map_tree,
                        map_keys,
                        new_graph,
                        new_values,
                    )

            if return_isam2_update:
                isam2_update = ISAM2Update()
                isam2_update.graph = gtsam.serializeNonlinearFactorGraph(new_graph)
                isam2_update.values = gtsam.serializeValues(new_values)
                isam2_updates.append(isam2_update)
            path_keyframes.append(keyframes)

        return path_keyframes, isam2_updates

    @add_lock
    def sonar_callback(self, ping):
        """
        Subscribe once to configure Oculus property.
        Assume sonar configuration doesn't change much.
        """
        self.oculus.configure(ping)
        self.sonar_sub.unregister()

    # @add_lock
    # def occ_callback(self, occ_msg):
    #     x0 = occ_msg.info.origin.position.x
    #     y0 = occ_msg.info.origin.position.y
    #     width = occ_msg.info.width
    #     height = occ_msg.info.height
    #     resolution = occ_msg.info.resolution
    #     occ_arr = np.array(occ_msg.data).reshape(height, width)
    #     occ_arr[occ_arr < 0] = 50
    #     occ_arr = occ_arr / 100.0
    #     self.occ = x0, y0, resolution, occ_arr

    def get_map(self, frames, resolution=None):
        """
        Get map from map server in mapping node.
        TODO It's better to integrate mapping into SLAM node
        """
        self.lock.acquire()
        req = GetOccupancyMapRequest()
        req.frames = frames
        req.resolution = -1 if resolution is None else resolution
        try:
            resp = self.get_map_client.call(req)
        except rospy.ServiceException as e:
            logerror("Failed to call get_map service {}".format(e))
            raise

        x0 = resp.occ.info.origin.position.x
        y0 = resp.occ.info.origin.position.y
        width = resp.occ.info.width
        height = resp.occ.info.height
        resolution = resp.occ.info.resolution
        occ_arr = np.array(resp.occ.data).reshape(height, width)
        occ_arr[occ_arr < 0] = 50
        occ_arr = np.clip(occ_arr / 100.0, 0.0, 1.0)
        self.lock.release()
        return x0, y0, resolution, occ_arr

    @add_lock
    def fo_callback(self, feature_msg, odom_msg):
        self.lock.acquire()
        time = feature_msg.header.stamp
        dr_pose3 = r2g(odom_msg.pose.pose)

        frame = Keyframe(False, time, dr_pose3)
        points = r2n(feature_msg)[:, :2].astype(np.float32)
        if len(points) and np.isnan(points[0, 0]):
            # In case feature extraction is skipped in this frame.
            frame.status = False
        else:
            frame.status = self.is_keyframe(frame)
        frame.twist = odom_msg.twist.twist

        if self.keyframes:
            dr_odom = self.current_keyframe.dr_pose.between(frame.dr_pose)
            pose = self.current_keyframe.pose.compose(dr_odom)
            frame.update(pose)

        if frame.status:
            frame.points = points

            if not self.keyframes:
                self.add_prior(frame)
            else:
                self.add_sequential_scan_matching(frame)

            self.update_factor_graph(frame)

            if self.enable_slam and self.add_nonsequential_scan_matching():
                self.update_factor_graph()

        self.current_frame = frame
        self.publish_all()
        self.lock.release()

    def publish_all(self):
        if not self.keyframes:
            return

        self.publish_pose()
        if self.current_frame.status:
            self.publish_trajectory()
            self.publish_constraint()
            self.publish_point_cloud()
            self.publish_slam_update()

    def publish_pose(self):
        """
        Append dead reckoning from Localization to SLAM estimate to achieve realtime TF.
        """
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.current_frame.time
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose = g2r(self.current_frame.pose3)

        cov = 1e-4 * np.identity(6, np.float32)
        # FIXME Use cov in current_frame
        cov[np.ix_((0, 1, 5), (0, 1, 5))] = self.current_keyframe.transf_cov
        pose_msg.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(pose_msg)

        o2m = self.current_frame.pose3.compose(self.current_frame.dr_pose3.inverse())
        o2m = g2r(o2m)
        p = o2m.position
        q = o2m.orientation
        self.tf.sendTransform(
            (p.x, p.y, p.z),
            [q.x, q.y, q.z, q.w],
            self.current_frame.time,
            "odom",
            "map",
        )

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose.pose
        odom_msg.child_frame_id = "base_link"
        odom_msg.twist.twist = self.current_frame.twist
        self.odom_pub.publish(odom_msg)

    def publish_constraint(self):
        """
        Publish constraints between poses in the factor graph,
        either sequential or non-sequential.
        """
        links = []
        for x, kf in enumerate(self.keyframes[1:], 1):
            p1 = self.keyframes[x - 1].pose.x(), self.keyframes[x - 1].pose.y()
            p2 = self.keyframes[x].pose.x(), self.keyframes[x].pose.y()
            links.append((p1, p2, "green"))

            for k, _ in self.keyframes[x].constraints:
                p0 = self.keyframes[k].pose.x(), self.keyframes[k].pose.y()
                links.append((p0, p2, "red"))

        if links:
            link_msg = ros_constraints(links)
            link_msg.header.stamp = self.current_keyframe.time
            self.constraint_pub.publish(link_msg)

    def publish_trajectory(self):
        """
        Publish 3D trajectory as point cloud in [x, y, z, roll, pitch, yaw, index] format.
        """
        poses = np.array([g2n(kf.pose3) for kf in self.keyframes])
        traj_msg = ros_colorline_trajectory(poses)
        traj_msg.header.stamp = self.current_keyframe.time
        traj_msg.header.frame_id = "map"
        self.traj_pub.publish(traj_msg)

    def publish_point_cloud(self):
        """
        Publish downsampled 3D point cloud with z = 0.
        The last column represents keyframe index at which the point is observed.
        """
        all_points = [np.zeros((0, 2), np.float32)]
        all_keys = []
        for key in range(len(self.keyframes)):
            pose = self.keyframes[key].pose
            transf_points = self.keyframes[key].transf_points
            all_points.append(transf_points)
            all_keys.append(key * np.ones((len(transf_points), 1)))

        all_points = np.concatenate(all_points)
        all_keys = np.concatenate(all_keys)
        sampled_points, sampled_keys = pcl.downsample(
            all_points, all_keys, self.point_resolution
        )
        sampled_xyzi = np.c_[sampled_points, np.zeros_like(sampled_keys), sampled_keys]
        if len(sampled_xyzi) == 0:
            return

        if self.save_fig:
            plt.figure()
            plt.scatter(
                sampled_xyzi[:, 0], sampled_xyzi[:, 1], c=sampled_xyzi[:, 3], s=1
            )
            plt.axis("equal")
            plt.gca().invert_yaxis()
            plt.savefig("step-{}-map.png".format(self.current_key - 1), dpi=100)
            plt.close("all")

        cloud_msg = n2r(sampled_xyzi, "PointCloudXYZI")
        cloud_msg.header.stamp = self.current_keyframe.time
        cloud_msg.header.frame_id = "map"
        self.cloud_pub.publish(cloud_msg)

    def publish_slam_update(self):
        """
        Publish the entire ISAM2 instance for exploration server.
        So BayesTree isn't built from scratch.
        """
        update_msg = ISAM2Update()
        update_msg.header.stamp = self.current_keyframe.time
        update_msg.key = self.current_key - 1
        update_msg.isam2 = gtsam.serializeISAM2(self.isam)
        self.slam_update_pub.publish(update_msg)


def offline(args):
    from rosgraph_msgs.msg import Clock
    from localization_node import LocalizationNode
    from feature_extraction_node import FeatureExtractionNode
    from mapping_node import MappingNode
    from bruce_slam.utils import io

    io.offline = True
    node.save_fig = False
    node.save_data = False

    loc_node = LocalizationNode()
    loc_node.init_node(SLAM_NS + "localization/")
    fe_node = FeatureExtractionNode()
    fe_node.init_node(SLAM_NS + "feature_extraction/")
    mp_node = MappingNode()
    mp_node.init_node(SLAM_NS + "mapping/")
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)

    for topic, msg in read_bag(args.file, args.start, args.duration, progress=True):
        while not rospy.is_shutdown():
            if callback_lock_event.wait(1.0):
                break
        if rospy.is_shutdown():
            break

        if topic == IMU_TOPIC:
            loc_node.imu_sub.callback(msg)
        elif topic == DVL_TOPIC:
            loc_node.dvl_sub.callback(msg)
        elif topic == DEPTH_TOPIC:
            loc_node.depth_sub.callback(msg)
        elif topic == SONAR_TOPIC:
            fe_node.sonar_sub.callback(msg)
            if node.sonar_sub.callback:
                node.sonar_sub.callback(msg)
            mp_node.sonar_sub.callback(msg)
        clock_pub.publish(Clock(msg.header.stamp))

        # Publish map to world so we can visualize all in a z-down frame in rviz.
        node.tf.sendTransform((0, 0, 0), [1, 0, 0, 0], msg.header.stamp, "map", "world")

    # Save trajectory and point cloud
    import os
    import datetime

    stamp = datetime.datetime.now()
    name = os.path.basename(args.file).split(".")[0]

    np.savez(
        "run-{}@{}.npz".format(name, stamp),
        poses=node.get_states(),
        points=np.c_[node.get_points(return_keys=True)],
    )


if __name__ == "__main__":
    rospy.init_node("slam", log_level=rospy.INFO)

    node = SLAMNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("Start online slam...")
        rospy.spin()
    else:
        loginfo("Start offline slam...")
        offline(args)
