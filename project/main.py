from env import Env
from robot import Robot


def main():
    env = Env()
    robot = Robot()

    while True:
        env.tick()

if __name__ == "__main__":
    main()

