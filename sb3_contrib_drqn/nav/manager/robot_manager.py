
from typing import Union
import rospy
import math
import random
import tf
from geometry_msgs.msg import PoseStamped, Pose2D, Point
from flatland_msgs.srv import MoveModelRequest, MoveModel
from astar_ros_msgs.srv import GetPlan, GetPlanRequest
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from sb3_contrib_drqn.utils.pose_transform import Pose2D2PoseStamped, PoseStamped2Pose2D, Pose2D2Point, PoseStamped2Point
from sb3_contrib_drqn.cfg.config import Config

class RobotManager:
    def __init__(self, ns, model_name, map_manager):
        # set default values
        self.ns = ns
        self.model_name = ns + "/" + model_name
        footprint = eval(rospy.get_param("%s/dmvm/FootprintProcessor/footprint"%ns, "[]"))
        if not isinstance(footprint, list):
            footprint = list(footprint)
        if len(footprint):
            max_height = max([abs(point[1]) for point in footprint])
            max_width = max([abs(point[0]) for point in footprint])
            self.robot_radius = math.hypot(max_height, max_width)
        else:
            self.robot_radius = 0.105  ## default burger robot
        self.map_frame = rospy.get_param("%s/global_planner/costmap/global_frame"%ns, "map")
        self.map_manager = map_manager
        
        # init publisher
        self.pubs = {}
        # self.pubs["move_robot"] = rospy.Publisher("%s/move_model"%ns, MoveModelMsg, queue_size=1)
        self.pubs["goal"] = rospy.Publisher("%s/goal"%ns, PoseStamped, queue_size=1)
        self.pubs["path"] = rospy.Publisher("%s/global_planner/planner/plan"%ns, Path, queue_size=1)

        # init clients
        self.clients = {}
        self.clients["move_robot"] = rospy.ServiceProxy("%s/move_model"%ns, MoveModel)
        self.clients["get_plan"] = rospy.ServiceProxy("%s/get_astar_plan"%ns, GetPlan)

        ## only for test
        test_mode = rospy.get_param("/test_mode", False)
        if test_mode:
            rospy.Service("%s/reset_robot"%ns, Empty, self.resetCallback)



    def generateSpawnAndGoalPose(self, forbidden_zone):
        safety_radius = self.robot_radius + Config.RobotConfig.RobotSafeDist
        spawn_pose = self.getRandomPose(safety_radius, forbidden_zone)
        goal_pose = self.getRandomPose(safety_radius, [*forbidden_zone, spawn_pose])
        return Pose2D(*spawn_pose), Pose2D(*goal_pose)

    def getRandomPose(self, safety_radius, forbidden_zone):
        return self.map_manager.get_random_pos_on_map(safety_radius, forbidden_zone)

    def getStaticPose(self, pos: Union[Pose2D, PoseStamped], output_type: str="PoseStamped"):
        assert isinstance(pos, Union[Pose2D, PoseStamped])
        
        if output_type == "PoseStamped":
            if isinstance(pos, Pose2D):
                static_pose = \
                    Pose2D2PoseStamped(pose=pos, frame_id=self.map_frame)
            else:
                static_pose = pos
                static_pose.header.stamp = rospy.get_rostime()
                static_pose.header.frame_id = self.map_frame
        
        elif output_type == "Pose2D":
            if isinstance(pos, Pose2D):
                static_pose = pos
            else:
                static_pose = Pose2D2PoseStamped(pose=pos)

        return static_pose
    
    def reset(self, forbidden_zone=[]):
        spawn_pos, goal_pos = self.generateSpawnAndGoalPose(forbidden_zone)
        self.moveRobot(spawn_pos)
        self.setGoal(Pose2D2Point(spawn_pos), Pose2D2Point(goal_pos))

    def moveRobot(self, pos: Union[Pose2D, PoseStamped]):
        assert isinstance(pos, Union[Pose2D, PoseStamped])

        ## for service call
        req = MoveModelRequest()
        req.name = self.model_name

        if isinstance(pos, Pose2D):
            req.pose = pos
            res = self.clients["move_robot"].call(req)
        else:
            req.pose = PoseStamped2Pose2D(pos)
            res = self.clients["move_robot"].call(req)
        if res.success:
            rospy.logwarn("[RobotManager] success to move robot.. try to get new plan")
        else:
            rospy.logwarn("[RobotManager] failed to move robot.., try move robot again..")
            self.moveRobot(pos)

    def setGoal(self, start: Point, goal: Point):
        assert isinstance(goal, Point)
        assert isinstance(start, Point)
        # if isinstance(goal, Pose2D):
        #     self.pubs["goal"].publish(Pose2D2PoseStamped(goal, self.map_frame))
        # else:
        #     self.pubs["goal"].publish(goal)
        req = GetPlanRequest()
        req.start = start
        req.goal = goal
        res = self.clients["get_plan"].call(req)
        self.pubs["path"].publish(res.path)

    def resetCallback(self, req):
        self.reset()
        return