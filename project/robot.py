import numpy as np

class Robot():
    def __init__(self, pos, env, robot_id):
        self.pos = np.array(pos, dtype=float)
        self.env = env
        self.robot_id = robot_id
        
        # Kinematic constraints
        self.max_speed = 3.0
        self.repulsion_radius = 60.0 # Distance to start avoiding other robots
        self.min_target_dist = 100.0 # Minimum distance to the target
        
    # --- Artificial Potential Field algorithm ---
    def step(self):
        # Get the estimated target position from the environment
        target_est = self.env.get_estimated_target()
        if target_est is None:
            return
            
        # Target constraint (stay on the self.min_target_dist)
        vec_to_target = target_est - self.pos
        dist_to_target = np.linalg.norm(vec_to_target)
        
        v_target = np.array([0.0, 0.0])
        if dist_to_target > 0:
            # Error is positive if too far, negative if too close.
            error_t = dist_to_target - self.min_target_dist
            v_target = (vec_to_target / dist_to_target) * error_t * 0.1
            
        # Distance-Based Formation Control
        v_formation = np.array([0.0, 0.0])
        desired_peer_dist = self.min_target_dist * 1.732 # An equilateral triangle inside a circle has side lenght of exactly the radius of the circle, R * sqrt(3) and sqrt(3) = 1,732
        for i, other_pos in enumerate(self.env.robot_pos):
            if i == self.robot_id:
                continue
            
            vec_to_peer = np.array(other_pos) - self.pos
            dist_p = np.linalg.norm(vec_to_peer)
            
            if dist_p > 0:
                error_p = dist_p - desired_peer_dist
                v_formation += (vec_to_peer / dist_p) * error_p * 0.05
                
        # Combine velocities
        velocity = v_target + v_formation
        
        # Cap max speed
        if np.linalg.norm(velocity) > self.max_speed:
            velocity = (velocity / np.linalg.norm(velocity)) * self.max_speed
            
        self.pos += velocity
        
        # Sync to environment
        self.env.robot_pos[self.robot_id] = tuple(self.pos)
        

        #self.received_robots = []
        #self.received_targets = []
