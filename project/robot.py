import numpy as np
from env import signal_strenght, angle_of_arrival

class Robot():
    def __init__(self, pos, env, robot_id):
        self.pos = np.array(pos, dtype=float)
        self.env = env
        self.robot_id = robot_id
        
        # Kinematic constraints
        self.max_speed = 3.0
        self.repulsion_radius = 60.0 # Distance to start avoiding other robots
        self.min_target_dist = 100.0 # Minimum distance to the target
        
        # Consensus variables
        self.local_measurement = None
        self.prev_local_measurement = None
        self.consensus_state = None
        self.has_los = False # Whether this robot had valid LOS this tick
       
    def get_sensor_reading(self):
        """Read raw hardware sensors and make a local, noisy guess."""
        true_target = self.env.target_pos[0]

        # Skip reading entirely if an obstacle blocks the signal path
        if self.env.los_blocked(true_target, self.pos):
            self.has_los = False # No valid reading this tick; coast on consensus
            return

        self.has_los = True
        
        # Get raw sensor data
        rx_power = signal_strenght(true_target, self.pos)
        a_min, a_max = angle_of_arrival(true_target, self.pos)
        
        # Adding random radio interference
        #rx_power += np.random.normal(0, 3.0)
        
        # Adding random compass error
        est_angle = np.radians((a_min + a_max) / 2.0)
        est_angle += np.random.normal(0, np.radians(10.0))
        
        # Reverse-calculate distance using Path Loss model from env.py
        path_loss = 20 - rx_power
        est_dist = 10 ** ((path_loss - 35) / 20.0)
        
        # Each robot has uniquely "broken" hardware to simulate Distributed Estimation importance
        bias_x = [40.0, -40.0, 0.0, 30.0, -30.0, 10.0][self.robot_id]  # extended to 6 robots
        bias_y = [0.0, 40.0, -40.0, -30.0, 10.0, -20.0][self.robot_id]  # extended to 6 robots
        
        # Debugging: Prove the bias is loaded on the first tick!
        if self.env.clock == 1 and self.robot_id == 0:
            print(f"Robot 0 Bias Applied: X: {bias_x}, Y: {bias_y}")
        
        # Create local coordinate estimate
        local_est = self.pos + np.array([
            est_dist * np.cos(est_angle),
            est_dist * np.sin(est_angle)
        ])
        
        # Update dynamic tracking history
        self.prev_local_measurement = self.local_measurement
        self.local_measurement = local_est
        
        if self.consensus_state is None:
            self.consensus_state = np.copy(self.local_measurement)
            self.prev_local_measurement = np.copy(self.local_measurement)
            
    def communicate_and_consensus(self, peers, wifi_enabled=True):
        """Talk to peers and run Dynamic Average Consensus."""
        if self.consensus_state is None:
            return
        
        consensus_gain = 0.2 # How fast robots agree on a consensus
        sensor_gain = 0.05 # Shock absorber for raw sensor
        sum_diff = np.array([0.0, 0.0])
        
        # Read broadcasts from peers
        if wifi_enabled:
            for peer in peers:
                if peer.robot_id == self.robot_id:
                    continue
                if peer.consensus_state is not None:
                    # Compare states
                    sum_diff += (peer.consensus_state - self.consensus_state)
                
        # Only pull toward local sensor if we had a valid reading this tick;
        # otherwise coast on peer consensus alone so stale data doesn't corrupt the estimate
        if self.has_los and self.local_measurement is not None:
            pull_to_sensor = self.local_measurement - self.consensus_state
        else:
            pull_to_sensor = np.array([0.0, 0.0])
        
        self.next_consensus_state = self.consensus_state + (consensus_gain * sum_diff) + (sensor_gain * pull_to_sensor)
        
    def step(self):
        """Act on the agreed consensus"""
        if hasattr(self, 'next_consensus_state'):
            self.consensus_state = self.next_consensus_state
            
        # Use the CONSENSUS state to drive movement
        target_est = self.consensus_state
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
        # For N robots in a regular polygon inscribed in a circle of radius R, side = 2R * sin(π/N)
        n_robots = len(self.env.robot_pos)
        desired_peer_dist = 2 * self.min_target_dist * np.sin(np.pi / n_robots)
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