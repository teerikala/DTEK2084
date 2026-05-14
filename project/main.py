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
            # First, move all robots
            for robot in robots:
                robot.step()
        
            # Update the environment and visuals
            env.tick()
    except KeyboardInterrupt:
        print("Simulation stopped.")
        plt.close()

if __name__ == "__main__":
    main()

