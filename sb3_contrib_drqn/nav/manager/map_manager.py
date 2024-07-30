import numpy as np
import random
import math
import traceback
import rospy
from map_distance_server.srv import GetDistanceMap
from astar_ros_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse
from shapely.geometry import Point, Polygon

class MapManager:
    """
    The map manager manages the static map
    and is used to get new goal, robot and
    obstacle positions.
    """

    def __init__(self, map: GetDistanceMap,ns):
        self.update_map(map)
        self.astar_client = rospy.ServiceProxy("get_astar_plan", GetPlan)

    def update_map(self, map: GetDistanceMap):
        self.map = map
        self.map_with_distances = np.reshape(
            self.map.data, 
            (self.map.info.height, self.map.info.width)
        )
        self.origin = map.info.origin.position

    def get_random_pos_on_map(self, safe_dist, forbidden_zones=[]):

        """
        This function is used by the robot manager and
        obstacles manager to get new positions for both
        robot and obstalces.
        The function will choose a position at random
        and then validate the position. If the position
        is not valid a new position is chosen. When
        no valid position is found after 100 retries
        an error is thrown.
        Args:
            safe_dist: minimal distance to the next
                obstacles for calculated positons
            forbidden_zones: Array of (x, y, radius),
                describing circles on the map. New
                position should not lie on forbidden
                zones e.g. the circles.
                x, y and radius are in meters
        Returns:
            A tuple with three elements: x, y, theta
        """
        # safe_dist is in meters so at first calc safe dist to distance on
        # map -> resolution of map is m / cell -> safe_dist in cells is
        # safe_dist / resolution
        safe_dist_in_cells = math.ceil(safe_dist / self.map.info.resolution) + 1
        forbidden_zones_in_cells = list(
            map(
                lambda point: [math.ceil(p / self.map.info.resolution) for p in point],
                forbidden_zones
            )
        )

        # Now get index of all cells were dist is > safe_dist_in_cells
        possible_cells = list(np.array(np.where(self.map_with_distances > safe_dist_in_cells)).transpose())

        assert len(possible_cells) > 0, "No cells available"

        # The position should not lie in the forbidden zones and keep the safe 
        # dist to these zones as well. We could remove all cells here but since
        # we only need one position and the amount of cells can get very high
        # we just pick positions at random and check if the distance to all 
        # forbidden zones is high enough

        while len(possible_cells) >= 0:
            if len(possible_cells) == 0:
                raise Exception("can't find any non-occupied spaces")

            # Select a random cell
            x, y = possible_cells.pop(random.randint(0, len(possible_cells) - 1))

            # Check if valid
            if self._is_pos_valid(x, y, safe_dist_in_cells, forbidden_zones_in_cells):
                break

        theta = random.uniform(-math.pi, math.pi)
  
        return (
            round(y * self.map.info.resolution , 3), 
            round(x * self.map.info.resolution , 3),
            theta
        )
    
    def _is_pos_valid(self, x, y, safe_dist, forbidden_zones):

        if len(forbidden_zones) == 0:
            return True

        for p in forbidden_zones:
                f_x, f_y, radius = p

                dist = math.floor(math.sqrt(
                    (x - f_x) ** 2 + (y - f_y) ** 2
                ))

                if dist > safe_dist + radius:
                    return True

        return False
   
    def get_start_and_goal_amg_sector(self, sectors, use_random=True, start_idx=0, end_idx=1): ## robot only spawn random
        start_goal = []
        if use_random:
            sector_indexes = random.sample(range(0, len(sectors)), 2)
        else:
            sector_indexes = [start_idx, end_idx]
        for sector_index in sector_indexes:
            start_goal.append(self.get_random_point_in_polygon(Polygon(sectors[sector_index])))
        return start_goal

    def get_random_point_in_polygon(self, polygon):
        x_min, y_min, x_max, y_max = polygon.bounds
        while True:
            point = Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))
            if polygon.contains(point):
                return point
            
    def get_pedsim_plan(self, startgoal, n):
        startgoal_req = GetPlanRequest()
        startgoal_req.start.x = startgoal[0].x
        startgoal_req.start.y = startgoal[0].y
        startgoal_req.start.z = 0
        startgoal_req.goal.x = startgoal[1].x
        startgoal_req.goal.y = startgoal[1].y
        startgoal_req.goal.z = 0
        path_res = self.astar_client(startgoal_req)

        path_list = [[pose_stamped.pose.position.x, pose_stamped.pose.position.y] for pose_stamped in path_res.path.poses]
        sampled_path_list = path_list[::n]
        waypoints = sampled_path_list + sampled_path_list[-2::-1]

        return waypoints