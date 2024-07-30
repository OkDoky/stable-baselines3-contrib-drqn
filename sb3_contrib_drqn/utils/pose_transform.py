import rospy
import tf
from geometry_msgs.msg import Pose2D, PoseStamped, Point

def Pose2D2PoseStamped(pose: Pose2D, frame_id: str):
    assert isinstance(pose, Pose2D)
    result = PoseStamped()
    result.header.stamp = rospy.get_rostime()
    result.header.frame_id = frame_id
    result.pose.position.x = pose.x
    result.pose.position.y = pose.y
    quat = tf.transformations.quaternion_from_euler(0, 0, pose.theta)
    result.pose.orientation.x = quat[0]
    result.pose.orientation.y = quat[1]
    result.pose.orientation.z = quat[2]
    result.pose.orientation.w = quat[3]
    return result

def PoseStamped2Pose2D(pose: PoseStamped):
    assert isinstance(pose, PoseStamped)
    result = Pose2D()
    result.x = pose.pose.position.x
    result.y = pose.pose.position.y
    quat = (pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w)
    result.theta = tf.transformations.euler_from_quaternion(quat)[2]
    return result

def Point2Pose2D(pose: Point):
    assert isinstance(pose, Point)
    result = Pose2D()
    result.x = pose.x
    result.y = pose.y
    return result

def Pose2D2Point(pose: Pose2D):
    assert isinstance(pose, Pose2D)
    result = Point()
    result.x = pose.x
    result.y = pose.y
    return result

def PoseStamped2Point(pose: PoseStamped):
    assert isinstance(pose, PoseStamped)
    result = Point()
    result.x = pose.pose.position.x
    result.y = pose.pose.position.y
    return result