from env import Env
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt

def make_robot_positions(n, radius=150.0, y_offset=-150.0):
    """Space N robots evenly along the bottom of the arena."""
    if n == 1:
        return [(0.0, y_offset)]
    return [(radius * np.cos(np.pi + np.pi * i / (n - 1)), y_offset) for i in range(n)]

def main():
    N_ROBOTS = 3  # Change to 3–6

    robot_pos = make_robot_positions(N_ROBOTS)
    target_pos = [(0.0, 150.0)]
    obstacles = [
        ((60, 80), (60, 220)),  # vertical wall right of target; clears all start positions
        ((-50, 89), (-50, 150)),
    ]

    env = Env(robot_pos, target_pos, obstacles)
    
    robots = [Robot(robot_pos[i], env, i) for i in range(N_ROBOTS)]

    convergence_threshold = 5.0  # metres; max spread between robots' consensus states to count as converged
    last_logged_estimate = None
    log_interval = 50  # ticks between periodic log lines

    try:
        while True:
            # Phase 1: All robots take their local sensor readings
            for robot in robots:
                robot.get_sensor_reading()
            
            # Phase 2: All robots exchange data and run Consensus Math
            for robot in robots:
                robot.communicate_and_consensus(robots)
                
            # Phase 3: All robots update their physical positions
            for robot in robots:
                robot.step()
            
            # Extract beliefs for visualization
            local_ests = [robot.local_measurement for robot in robots if robot.local_measurement is not None]
            consensus_ests = [robot.consensus_state for robot in robots if robot.consensus_state is not None]
            
            env.draw_beliefs(local_ests, consensus_ests)
            
            # Log target location estimate
            states = [r.consensus_state for r in robots if r.consensus_state is not None]
            if len(states) == len(robots):
                spread = max(np.linalg.norm(states[i] - states[j])
                             for i in range(len(states))
                             for j in range(i + 1, len(states)))
                
                mean_estimate = np.mean(states, axis=0)

                if spread < convergence_threshold:
                    # Only log when estimate has moved more than 1 m since last log
                    if last_logged_estimate is None or np.linalg.norm(mean_estimate - last_logged_estimate) > 1.0:
                        print(f"[t={env.clock:>5}] Target location converged: "
                              f"x={mean_estimate[0]:>7.2f}, y={mean_estimate[1]:>7.2f} m  "
                              f"(spread={spread:.2f} m)")
                        last_logged_estimate = mean_estimate.copy()
                elif env.clock % log_interval == 0:
                    # Periodic update while still converging
                    print(f"[t={env.clock:>5}] Converging...  "
                          f"estimate: x={mean_estimate[0]:>7.2f}, y={mean_estimate[1]:>7.2f} m  "
                          f"(spread={spread:.2f} m)")

            # Update the environment and visuals
            env.tick()
            
    except KeyboardInterrupt:
        print("Simulation stopped.")
        if last_logged_estimate is not None:
            print(f"Last converged estimate: x={last_logged_estimate[0]:.2f}, y={last_logged_estimate[1]:.2f} m")
        plt.close()

if __name__ == "__main__":
    main()