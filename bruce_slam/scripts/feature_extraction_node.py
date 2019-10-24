#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, Image

from bruce_slam.utils.io import *
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import apply_custom_colormap
from bruce_slam.feature import FeatureExtraction
from bruce_slam import pcl


class FeatureExtractionNode(FeatureExtraction):
    def __init__(self):
        super(FeatureExtractionNode, self).__init__()

        self.colormap = "RdBu_r"
        self.pub_rect = True
        self.resolution = 0.5
        self.outlier_filter_radius = 1.0
        self.outlier_filter_min_points = 5
        self.skip = 5

        # for offline visualization
        self.feature_img = None

    def init_node(self, ns="~"):
        self.Ntc = rospy.get_param(ns + "CFAR/Ntc")
        self.Ngc = rospy.get_param(ns + "CFAR/Ngc")
        self.Pfa = rospy.get_param(ns + "CFAR/Pfa")
        self.rank = rospy.get_param(ns + "CFAR/rank")
        self.alg = rospy.get_param(ns + "CFAR/alg", "SOCA")

        self.threshold = rospy.get_param(ns + "filter/threshold")
        self.resolution = rospy.get_param(ns + "filter/resolution")
        self.outlier_filter_radius = rospy.get_param(ns + "filter/radius")
        self.outlier_filter_min_points = rospy.get_param(ns + "filter/min_points")
        self.skip = rospy.get_param(ns + "filter/skip")

        self.coordinates = rospy.get_param(
            ns + "visualization/coordinates", "cartesian"
        )
        self.radius = rospy.get_param(ns + "visualization/radius")
        self.color = rospy.get_param(ns + "visualization/color")

        self.sonar_sub = rospy.Subscriber(
            SONAR_TOPIC, OculusPing, self.callback, queue_size=10
        )

        self.feature_pub = rospy.Publisher(
            SONAR_FEATURE_TOPIC, PointCloud2, queue_size=10
        )

        self.feature_img_pub = rospy.Publisher(
            SONAR_FEATURE_IMG_TOPIC, Image, queue_size=10
        )

        self.configure()
        loginfo("Sonar feature extraction node is initialized")

    @add_lock
    def callback(self, ping):
        if ping.ping_id % self.skip != 0:
            self.feature_img = None
            # Don't extract features in every frame.
            # But we still need empty point cloud for synchronization in SLAM node.
            nan = np.array([[np.nan, np.nan]])
            self.publish_features(ping, nan)
            return

        # Detected targets in row/col
        locs = self.detect(ping)
        # Detected targets in local x/y
        points = self.to_points(locs)

        if len(points) and self.resolution > 0:
            points = pcl.downsample(points, self.resolution)

        if self.outlier_filter_min_points > 1 and len(points) > 0:
            # points = pcl.density_filter(points, 5, self.min_density, 1000)
            points = pcl.remove_outlier(
                points, self.outlier_filter_radius, self.outlier_filter_min_points
            )

        self.publish_features(ping, points)
        self.publish_feature_image(ping, locs)

    def publish_features(self, ping, points):
        points = np.c_[points, np.zeros(len(points))]
        feature_msg = n2r(points, "PointCloudXYZ")
        feature_msg.header.stamp = ping.header.stamp
        feature_msg.header.frame_id = "base_link"
        self.feature_pub.publish(feature_msg)

    def publish_feature_image(self, ping, locs):
        img = r2n(ping.ping)

        if self.coordinates == "cartesian":
            rect_img = self.oculus.remap(img=img)
            rect_img = apply_custom_colormap(rect_img, self.colormap)
            rect_locs = self.remap(locs)
            self.feature_img = self.draw_features(
                rect_img, rect_locs, self.radius, self.color
            )
        else:
            img = apply_custom_colormap(img, self.colormap)
            self.feature_img = self.draw_features(img, locs, self.radius, self.color)

        img_msg = n2r(self.feature_img, "ImageBGR")
        img_msg.header.stamp = ping.header.stamp
        self.feature_img_pub.publish(img_msg)

    def draw_features(self, img, locs, radius=2, color=(0, 255, 0)):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r, c in locs:
            r, c = int(round(r)), int(round(c))
            img = cv2.circle(img, (c, r), radius, color)
        return img


def offline(args):
    from rosgraph_msgs.msg import Clock
    from bruce_slam.utils import io

    io.offline = True
    cimgs = []
    logwarn("Press s to save image")

    clock_pub = rospy.Publisher("/clock", Clock, queue_size=100)
    for topic, msg in read_bag(args.file, args.start, args.duration):
        while not rospy.is_shutdown():
            if callback_lock_event.wait(1.0):
                break
        if rospy.is_shutdown():
            break

        if topic == SONAR_TOPIC:
            node.sonar_sub.callback(msg)
            if node.feature_img is not None:
                cv2.imshow("feature image", node.feature_img)
                key = cv2.waitKey()
                if key == 27:
                    break
                elif key == ord("s"):
                    loginfo("Save current frame")
                    cimgs.append(node.cimg)
                else:
                    continue

        clock_pub.publish(Clock(msg.header.stamp))

    if cimgs:
        np.savez("cimgs.npz", cimgs=cimgs)


if __name__ == "__main__":
    rospy.init_node("feature_extraction_node", log_level=rospy.INFO)

    node = FeatureExtractionNode()
    node.init_node()

    parser = common_parser()
    args, _ = parser.parse_known_args()
    if not args.file:
        loginfo("Start online sonar feature extraction...")
        rospy.spin()
    else:
        loginfo("Start offline sonar feature extraction...")
        offline(args)
