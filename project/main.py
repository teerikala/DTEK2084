from env import Env
from robot import Robot
import matplotlib.pyplot as plt

def main():
    robot_pos = [(-150.0, -150.0), (150.0, -150.0), (0.0, -150.0)]
    target_pos = [(0.0, 150.0)]

    env = Env(robot_pos, target_pos)
    
    robots = [
        Robot(robot_pos[0], env, 0),
        Robot(robot_pos[1], env, 1),
        Robot(robot_pos[2], env, 2)
    ]
    
    #robot1 = Robot(robot_pos[0], env)
    #robot2 = Robot(robot_pos[1], env)
    #robot3 = Robot(robot_pos[2], env)

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
            
            # Update the environment and visuals
            env.tick()
            
    except KeyboardInterrupt:
        print("Simulation stopped.")
        plt.close()

if __name__ == "__main__":
    main()

