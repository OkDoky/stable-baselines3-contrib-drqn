import math
import rospy
import copy
import numpy as np
from sb3_contrib_drqn.nav.conditions import ResultCondition
from sb3_contrib_drqn.nav.data_map_config import DMConfig

def get_mask(targets):
    mask = 0
    for target in targets:
        mask |= (1 << target)
    return int(mask)

class RewardHandler:
    total_reward: float = 0.0
    config = DMConfig()
    def __init__(self, cx=100, cy=100):
        self.function_list = [self.step_count, 
                              self.collision_check, 
                              self.goal_reached, 
                              self.approach_goal,
                            #   self.safety_check, 
                            #   self.cross_track_error_check,
                              ]
        self.footprint_value = self.config.data_map_infos.padded_footprint
        self.obstacle_value = self.config.data_map_infos.obstacle
        self.velocity_value = self.config.data_map_infos.velocity
        self.path_value = self.config.data_map_infos.path
        self.goal_value = self.config.data_map_infos.goal
        self.center = [cx, cy]
        self.last_cell_dist = 5.0
        self.current_cell_dist = 5.0
        
    def get_rewards(self, obs):
        rewards = 0.0
        done = False
        info = {}
        for func in self.function_list:
            reward, done, info = func(obs)
            rewards += reward
            if done: break
        self.total_reward += rewards
        return self.total_reward, done, info
    
    def reset_rewards(self):
        self.total_reward = 0.0
        self.last_cell_dist = 5.0
    
    def collision_check(self, obs):
        reward = 0.0
        mask = get_mask([self.obstacle_value, self.footprint_value])
        obs = obs.astype(int)
        if np.any(np.bitwise_and(obs, mask) == mask):
            reward = self.config.reward_panalty.collision
            rospy.logwarn("[RewardHandler] collision, done")
            return reward, True, {"result_condition": ResultCondition.COLLISION, "is_success": False}
        return reward, False, {}

    def step_count(self, obs):
        reward = self.config.reward_panalty.step
        return reward, False, {}

    def goal_reached(self, obs):
        reward = 0.0
        mask = get_mask([self.goal_value])
        obs = obs.astype(int)
        goal_indices = np.where(np.bitwise_and(obs, mask) == mask)
        if not len(goal_indices[0]):
            return reward, False, {}
        cell_dist = math.hypot(goal_indices[0] - self.center[0], goal_indices[1] - self.center[1])
        if cell_dist < self.config.data_map_infos.goal_threshold:
            reward = self.config.reward_panalty.goal
            rospy.logwarn("[RewardHandler] goal reached, done")
            return reward, True, {"result_condition": ResultCondition.SUCCESS, "is_success": True}
        return reward, False, {}
    
    def approach_goal(self, obs):
        reward = 0.0
        mask = get_mask([self.path_value])
        obs = obs.astype(int)
        cell_dist = np.sum(np.bitwise_and(obs, mask) == mask)
        reward = round((self.last_cell_dist - cell_dist) * self.config.data_map_infos.map_resolution, 2) * 10
        self.last_cell_dist = cell_dist
        return reward, False, {}

    def safety_check(self, obs):
        reward = 0.0
        return reward, False, {}

    def cross_track_error_check(self, obs):
        reward = 0.0
        return reward, False, {}

if __name__ == "__main__":
    test_data = np.zeros((10, 10))
    test_data[4][4] = 32
    
    test_data[4][4] = 40
    test_data[4][5] = 8
    test_data[4][6] = 8
    test_data[5][6] = 8
    test_data[6][6] = 8
    test_data[6][5] = 8
    test_data[6][4] = 8
    test_data[5][4] = 8

    # test_data[4][4] = 9
    
    rh = RewardHandler(cx=5, cy=5)
    reward = rh.get_rewards(np.array(test_data, dtype=np.uint8))
    print("test date reward : ", reward)