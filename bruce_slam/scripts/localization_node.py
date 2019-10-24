#!/usr/bin/env python
import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer
from message_filters import Cache, Subscriber
import gtsam

from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.io import *
from bruce_slam.utils.visualization import ros_colorline_trajectory


class LocalizationNode(object):
    def __init__(self):
        self.pose = None
        self.prev_time = None
        self.prev_vel = None
        self.keyframes = []

        # Force yaw at origin to be aligned with x axis
        self.imu_yaw0 = None
        self.imu_pose = [0, 0, 0, -np.pi / 2, 0, 0]
        self.imu_rot = None
        self.dvl_max_velocity = 0.3

        # Create a new key pose when
        # - |ti - tj| > min_duration and
        # - |xi - xj| > max_translation or
        # - |ri - rj| > max_rotation
        self.keyframe_duration = None
        self.keyframe_translation = None
        self.keyframe_rotation = None

        self.dvl_error_timer = 0.0

    def init_node(self, ns="~"):
        self.imu_pose = rospy.get_param(ns + "imu_pose")
        self.imu_pose = n2g(self.imu_pose, "Pose3")
        self.imu_rot = self.imu_pose.rotation()

        # Parameters for Node
        self.dvl_max_velocity = rospy.get_param(ns + "dvl_max_velocity")
        self.keyframe_duration = rospy.get_param(ns + "keyframe_duration")
        self.keyframe_translation = rospy.get_param(ns + "keyframe_translation")
        self.keyframe_rotation = rospy.get_param(ns + "keyframe_rotation")

        # Subscribers and caches
        self.imu_sub = Subscriber(IMU_TOPIC, Imu)
        self.dvl_sub = Subscriber(DVL_TOPIC, DVL)
        self.depth_sub = Subscriber(DEPTH_TOPIC, Depth)
        self.depth_cache = Cache(self.depth_sub, 1)

        # Use point cloud for visualization
        self.traj_pub = rospy.Publisher(
            LOCALIZATION_TRAJ_TOPIC, PointCloud2, queue_size=10
        )
        self.odom_pub = rospy.Publisher(
            LOCALIZATION_ODOM_TOPIC, Odometry, queue_size=10
        )

        # Sync
        self.ts = ApproximateTimeSynchronizer([self.imu_sub, self.dvl_sub], 200, 0.1)
        self.ts.registerCallback(self.callback)

        self.tf = tf.TransformBroadcaster()

        loginfo("Localization node is initialized")

    def callback(self, imu_msg, dvl_msg):
        depth_msg = self.depth_cache.getLast()
        if depth_msg is None:
            return
        dd_delay = (depth_msg.header.stamp - dvl_msg.header.stamp).to_sec()
        if abs(dd_delay) > 1.0:
            logdebug("Missing depth message for {}".format(dd_delay))

        rot = r2g(imu_msg.orientation)
        # nRb = nRs * bRs^-1
        rot = rot.compose(self.imu_rot.inverse())

        if self.imu_yaw0 is None:
            self.imu_yaw0 = rot.yaw()
        rot = gtsam.Rot3.Ypr(rot.yaw() - self.imu_yaw0, rot.pitch(), rot.roll())

        vel = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])
        if np.any(np.abs(vel) > self.dvl_max_velocity):
            if self.pose:
                self.dvl_error_timer += (dvl_msg.header.stamp - self.prev_time).to_sec()
                if self.dvl_error_timer > 5.0:
                    logwarn(
                        "DVL velocity ({:.1f}, {:.1f}, {:.1f}) exceeds max velocity {:.1f} for {:.1f} secs.".format(
                            vel[0],
                            vel[1],
                            vel[2],
                            self.dvl_max_velocity,
                            self.dvl_error_timer,
                        )
                    )
                vel = self.prev_vel
            else:
                return
        else:
            self.dvl_error_timer = 0.0

        if self.pose:
            dt = (dvl_msg.header.stamp - self.prev_time).to_sec()
            dv = (vel + self.prev_vel) * 0.5
            trans = dv * dt

            local_point = gtsam.Point2(trans[0], trans[1])
            pose2 = gtsam.Pose2(
                self.pose.x(), self.pose.y(), self.pose.rotation().yaw()
            )
            point = pose2.transform_from(local_point)

            self.pose = gtsam.Pose3(
                rot, gtsam.Point3(point.x(), point.y(), depth_msg.depth)
            )
        else:
            self.pose = gtsam.Pose3(rot, gtsam.Point3(0, 0, depth_msg.depth))

        self.prev_time = dvl_msg.header.stamp
        self.prev_vel = vel
        omega = imu_msg.angular_velocity
        omega = np.array([omega.x, omega.y, omega.z])
        self.prev_omega = self.imu_rot.matrix().dot(omega)

        new_keyframe = False
        if not self.keyframes:
            new_keyframe = True
        else:
            duration = self.prev_time.to_sec() - self.keyframes[-1][0]
            if duration > self.keyframe_duration:
                odom = self.keyframes[-1][1].between(self.pose)
                odom = g2n(odom)
                translation = np.linalg.norm(odom[:3])
                rotation = abs(odom[-1])

                if (
                    translation > self.keyframe_translation
                    or rotation > self.keyframe_rotation
                ):
                    new_keyframe = True

        if new_keyframe:
            self.keyframes.append((self.prev_time.to_sec(), self.pose))
        self.publish_pose(new_keyframe)

    def publish_pose(self, publish_traj=False):
        if self.pose is None:
            return

        header = rospy.Header()
        header.stamp = self.prev_time
        header.frame_id = "odom"

        odom_msg = Odometry()
        odom_msg.header = header
        # pose in odom frame
        odom_msg.pose.pose = g2r(self.pose)
        # twist in local frame
        odom_msg.child_frame_id = "base_link"
        # Local planer behaves worse
        # odom_msg.twist.twist.linear.x = self.prev_vel[0]
        # odom_msg.twist.twist.linear.y = self.prev_vel[1]
        # odom_msg.twist.twist.linear.z = self.prev_vel[2]
        # odom_msg.twist.twist.angular.x = self.prev_omega[0]
        # odom_msg.twist.twist.angular.y = self.prev_omega[1]
        # odom_msg.twist.twist.angular.z = self.prev_omega[2]
        odom_msg.twist.twist.linear.x = 0
        odom_msg.twist.twist.linear.y = 0
        odom_msg.twist.twist.linear.z = 0
        odom_msg.twist.twist.angular.x = 0
        odom_msg.twist.twist.angular.y = 0
        odom_msg.twist.twist.angular.z = 0
        self.odom_pub.publish(odom_msg)

        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        self.tf.sendTransform(
            (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), header.stamp, "base_link", "odom"
        )

        if publish_traj:
            traj = np.array([g2n(pose) for _, pose in self.keyframes])
            traj_msg = ros_colorline_trajectory(traj)
            traj_msg.header = header
            self.traj_pub.publish(traj_msg)


def offline(args):
    from rosgraph_msgs.msg import Clock
    from bruce_slam.utils import io

    io.offline = True

    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)
    for topic, msg in read_bag(args.file, args.start, args.duration):
        while not rospy.is_shutdown():
            if callback_lock_event.wait(1.0):
                break
        if rospy.is_shutdown():
            break

        if topic == IMU_TOPIC:
            node.imu_sub.callback(msg)
        elif topic == DVL_TOPIC:
            node.dvl_sub.callback(msg)
        elif topic == DEPTH_TOPIC:
            node.depth_sub.callback(msg)

        clock_pub.publish(Clock(msg.header.stamp))


if __name__ == "__main__":
    rospy.init_node("localization", log_level=rospy.INFO)

    node = LocalizationNode()
    node.init_node()

    args, _ = common_parser().parse_known_args()
    if not args.file:
        loginfo("Start online localization...")
        rospy.spin()
    else:
        loginfo("Start offline localization...")
        offline(args)
