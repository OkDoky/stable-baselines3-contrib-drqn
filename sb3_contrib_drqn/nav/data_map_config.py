
class DMConfig:
    class data_map_infos:
        obstacle = 0  ## 2^0
        path = 1  ## 2^1
        footprint = 2  ## 2^2
        padded_footprint = 3  ## 2^3
        velocity = 4  ## 2^4
        goal = 5  ## 2^5
        
        map_resolution = 0.05
        goal_threshold = int(0.5 / map_resolution)
        
    class reward_panalty:
        collision = -30
        goal = 30
        step = -0.1