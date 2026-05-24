import numpy as np
import matplotlib.pyplot as plt
from env import Env
from robot import Robot

def make_robot_positions(n, radius=150.0, y_offset=-150.0):
    """Space N robots evenly along the bottom of the arena."""
    if n == 1:
        return [(0.0, y_offset)]
    return [(radius * np.cos(np.pi + np.pi * i / (n - 1)), y_offset) for i in range(n)]

def run_test(wifi_on, steps=300):
    N_ROBOTS = 2  # Change to 3–6

    robot_pos = make_robot_positions(N_ROBOTS)
    target_pos = [(0.0, 150.0)]
    obstacles = [
        ((60, 80), (60, 150)), # vertical wall right of target; clears all start positions
        ((-50, 89), (-50, 150)),  
    ]
    
    # Initialize the environment
    env = Env(robot_pos, target_pos, obstacles)
    
    # Close the live visualizer window so the test runs instantly
    plt.close(env.fig)

    robots = [Robot(robot_pos[i], env, i) for i in range(N_ROBOTS)]

    error_history = []

    for _ in range(steps):
        # Phase 1
        for robot in robots:
            robot.get_sensor_reading()

        # Phase 2 (Pass the wifi state!)
        for robot in robots:
            robot.communicate_and_consensus(robots, wifi_enabled=wifi_on)

        # Phase 3
        for robot in robots:
            robot.step()

        # Record the Estimation Error of Robot 0
        if robots[0].consensus_state is not None:
            true_pos = np.array(target_pos[0])
            est_pos = robots[0].consensus_state
            
            # Calculate distance between real target and estimated target
            error_distance = np.linalg.norm(true_pos - est_pos)
            error_history.append(error_distance)

    return error_history

def main():
    print("Running Simulation with WiFi ON (Consensus Active)...")
    error_wifi_on = run_test(wifi_on=True)
    
    print("Running Simulation with WiFi OFF (Consensus Disabled)...")
    error_wifi_off = run_test(wifi_on=False)

    # --- Plot the Results ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(error_wifi_off, label="No Communication (Raw Sensors)", color='red', alpha=0.7)
    plt.plot(error_wifi_on, label="Consensus Network Active", color='blue', linewidth=2)
    
    plt.title("Target Estimation Error Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Steps", fontsize=12)
    plt.ylabel("Estimation Error (Meters)", fontsize=12)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()