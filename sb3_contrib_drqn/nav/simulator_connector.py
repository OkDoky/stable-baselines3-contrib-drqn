
from collections import deque
import traceback
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sb3_contrib_drqn.nav.reward_functions import RewardHandler

Actions = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 0],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
]

class SimulatorHandler:
    def __init__(self, ns: str):
        self.ns = ns
        rospy.init_node("imulator_handler_node")
        # subscriber for observations
        self.subs = []
        self.subs.append(rospy.Subscriber("data_map", OccupancyGrid, self.data_map_callback))
        self.subs.append(rospy.Subscriber("feedback_vel", Twist, self.feedback_callback))
        
        # publisher for action
        self.pubs = {}
        self.pubs["cmd_vel"] = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        
        # variables
        self.lin_acc = 0.2
        self.data_map = deque(maxlen=2)
        self.current_velocity = Twist()
        
        # reward handler
        self.rh = RewardHandler()
        
    def send_action(self, action: np.ndarray):
        ## action setup 
        ## 0 ~ 9, 
        # 0 : {lin acc : -1, ang vel : -1}, 
        # 1 : {lin acc : -1, ang vel :  0}, 
        # 2 : {lin acc : -1, ang vel :  1}, 
        # 3 : {lin acc :  0, ang vel : -1}, 
        # 4 : {lin acc :  0, ang vel :  0}, 
        # 5 : {lin acc :  0, ang vel :  1}, 
        # 6 : {lin acc :  1, ang vel : -1}, 
        # 7 : {lin acc :  1, ang vel :  0}, 
        # 8 : {lin acc :  1, ang vel :  1}, 
        # 9 : break
 
        def action_parser(action):
            if action == 9: # break
                return Twist()
            lin_vel = (Actions[action][0] * self.lin_acc) + self.current_velocity.linear.x
            twist = Twist()
            twist.linear.x = lin_vel
            twist.angular.z = Actions[action][1]
            return twist
        vel = action_parser(action)
        self.pubs["cmd_vel"].publish(vel)
    
    def get_transitions(self):
        obs = self.get_observation()
        reward, done, info = self.get_reward(obs)
        return obs, reward, done, info
    
    def get_observation(self):
        try:
            obs = self.data_map.pop()
            return obs
        except IndexError:
            # rospy.logwarn("[SimulatorHandler] dm queue is empty..")
            return np.zeros((200, 200))
        except Exception:
            rospy.logwarn("[SimulatorHandler] some exception, %s"%traceback.format_exc())
            return np.zeros((200, 200))
    
    def get_reward(self, obs):
        return self.rh.get_rewards(obs)
        
    def reset(self):
        self._stop_action()
        self._reset_world()
        self._reset_reward()
    
    def data_map_callback(self, msg):
        self.data_map.append(np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width))

    def feedback_callback(self, msg):
        self.current_velocity = msg
    
    def _stop_action(self):
        self.pubs["cmd_vel"].publish(Twist())
    
    def _reset_world(self):
        """
        reset robot position in gazebo, amcl
        random position for robot
        """
        pass
    
    def _reset_reward(self):
        self.rh.reset_rewards()
    