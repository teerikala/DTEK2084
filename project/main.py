from env import Env
from robot import Robot


def main():
    robot_pos = [(-150.0, -150.0), (150.0, -150.0), (0.0, -150.0)]
    target_pos = [(0.0, 150.0)]

    env = Env(robot_pos, target_pos)
    
    robot1 = Robot(robot_pos[0], env)
    robot2 = Robot(robot_pos[1], env)
    robot3 = Robot(robot_pos[2], env)

    while True:
        env.tick()


if __name__ == "__main__":
    main()

